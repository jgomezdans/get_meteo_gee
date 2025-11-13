"""Core public API and adapters to GEE/xee.

This module exposes ``get_meteo_data`` and hides the details of fetching
hourly ERA5-Land + DEM. Two small adapter functions are the only places
that touch xee/GEE directly. Everything else (validation, aggregation,
units, tz handling) is framework-agnostic.
"""

from __future__ import annotations

from typing import Iterable

import ee  # type: ignore
import pandas as pd
import xarray as xr

from .sources import ERA5L_ID, DEM_DEFAULT
from .models import MODELS
from .utils import (
    is_site,
    parse_site,
    parse_bbox,
    bbox_size_km,
    year_bounds_utc,
)


# ------------------------- Public entry point -------------------------------


def get_meteo_data(
    loc: Iterable[float],
    year: int,
    model: str,
    ee_project: str,
    *,
    tz: str = "UTC",
    variables: tuple[str, ...] | None = None,
    dem_asset: str = DEM_DEFAULT,
    ee_opt_url: str | None = "https://earthengine-highvolume.googleapis.com",
) -> pd.DataFrame | xr.Dataset:
    """Fetch ERA5-Land hourly via xee/GEE and return model-specific output.

    Parameters
    ----------
    loc
        (lat, lon) for a site, or (min_lon, min_lat, max_lon, max_lat) for
        a bounding box. Sites use nearest-neighbour sampling; NO SPATIAL
        BUFFERING!.
    year
        Calendar year (UTC bounds Jan 1 00:00 to Jan 1 next year 00:00).
    model
        Either "db" (hourly D&B) or "wofost" (daily aggregates).
    tz
        IANA timezone for WOFOST daily aggregation windows. Ignored for D&B.
        Default is "UTC".
    variables
        Optional subset of variables to fetch (must be a subset of the
        model's required bands). Most users should leave this as None.
    dem_asset
        DEM asset to use. Defaults to "COPERNICUS/DEM/GLO30". The optional
        fallback is "USGS/GTOPO30".

    Returns
    -------
    pandas.DataFrame or xarray.Dataset
        Site returns a DataFrame. BBox returns a Dataset.

    Notes
    -----
    This function delegates GEE/xee access to small adapters. In the future,
    we might be able to support other data backends other than GEE, which of
    course belongs to a nasty capitalist company. The rest of the pipeline is
    pure pandas/xarray.
    """
    model = model.lower().strip()
    if model not in MODELS:
        msg = f"Unknown model '{model}'. Use 'db' or 'wofost'."
        raise ValueError(msg)
    mspec = MODELS[model]

    req = set(mspec.required_bands)
    if variables is not None:
        subset = set(variables)
        if not subset.issubset(req):
            msg = "variables must be a subset of the model's required bands."
            raise ValueError(msg)
        bands = tuple(v for v in mspec.required_bands if v in subset)
    else:
        bands = mspec.required_bands

    start, end = year_bounds_utc(year)

    ee_init_kwargs = {}
    if ee_project is not None:
        ee_init_kwargs["project"] = ee_project
    if ee_opt_url is not None:
        ee_init_kwargs["opt_url"] = ee_opt_url

    if is_site(loc):
        lat, lon = parse_site(loc)
        # Fetch hourly site timeseries and DEM scalar
        df_hourly = _xee_fetch_site_hourly(
            collection_id=ERA5L_ID,
            bands=bands,
            lat=lat,
            lon=lon,
            start=start,
            end=end,
            ee_init_kwargs=ee_init_kwargs,
        )
        dem_value = _xee_fetch_site_dem(
            dem_asset=dem_asset,
            lat=lat,
            lon=lon,
            ee_init_kwargs=ee_init_kwargs,
        )
        return mspec.postprocess_site(df_hourly, dem_value, tz)

    # BBox path
    min_lon, min_lat, max_lon, max_lat = parse_bbox(loc)
    w_km, h_km = bbox_size_km(min_lon, min_lat, max_lon, max_lat)
    if w_km > 200.0 or h_km > 200.0:
        msg = (
            "bbox must be <= 200 km in width and height. "
            f"Got {w_km:.1f} x {h_km:.1f} km."
        )
        raise ValueError(msg)

    ds_hourly = _xee_fetch_bbox_hourly(
        collection_id=ERA5L_ID,
        bands=bands,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        start=start,
        end=end,
        ee_init_kwargs=ee_init_kwargs,
    )
    dem_grid = _xee_fetch_bbox_dem(
        dem_asset=dem_asset,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        ee_init_kwargs=ee_init_kwargs,
    )
    return mspec.postprocess_area(ds_hourly, dem_grid, tz)


# ------------------------- xee adapters -----------------------


def _ee_open_dataset(
    *,
    src: "ee.ImageCollection | ee.Image",
    geometry: "ee.Geometry",
    bands: tuple[str, ...] | None = None,
    scale: float = 0.1,
    projection: "ee.Projection | None" = None,
    ee_init_kwargs: dict | None = None,
) -> "xr.Dataset":
    """Open an EE ImageCollection/Image as xarray.Dataset.

    Parameters
    ----------
    src
        EE ImageCollection (hourly) or Image (DEM).
    geometry
        ee.Geometry.Point for site, ee.Geometry.Rectangle for bbox.
    bands
        Optional band subset to select on the collection/image.
    scale
        Pixel size in degrees. 0.1 ~ 10 km for ERA5-Land.
    projection
        EE projection. For ERA5-Land use ic.first().select(0).projection().

    Returns
    -------
    xarray.Dataset
        Dataset with dims including 'time' for collections.
    """
    import ee  # type: ignore

    try:
        ee.Initialize(**(ee_init_kwargs or {}))
    except Exception:
        raise RuntimeError("Please initialise Earth Engine API before use.")

    obj = src
    if bands:
        obj = obj.select(list(bands))

    ds = xr.open_dataset(
        obj,
        engine="ee",
        projection=projection,
        geometry=geometry,
        scale=scale,
    )
    return ds


def _xee_fetch_site_hourly(
    *,
    collection_id: str,
    bands: tuple[str, ...],
    lat: float,
    lon: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    ee_init_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Nearest-neighbour hourly site series → pandas.DataFrame.

    Reuses _ee_open_dataset with a Point geometry, squeezes spatial dims,
    ensures UTC hourly continuity.
    """
    import ee  # type: ignore

    try:
        ee.Initialize(**(ee_init_kwargs or {}))
    except Exception:
        raise RuntimeError("Please initialise Earth Engine API before use.")

    ic = ee.ImageCollection(collection_id).filterDate(
        start.isoformat(), end.isoformat()
    )
    proj = ic.first().select(0).projection()
    geom = ee.Geometry.Point(float(lon), float(lat))

    ds = _ee_open_dataset(
        src=ic,
        geometry=geom,
        bands=bands,
        scale=0.1,
        projection=proj,
        ee_init_kwargs=ee_init_kwargs,
    )

    # Squeeze 1x1 lat/lon to get a pure time series
    for d in ("lat", "lon"):
        if d in ds.dims and ds.sizes.get(d, 1) == 1:
            ds = ds.squeeze(d, drop=True)

    # Make time tz-aware UTC and continuous hourly
    t = pd.DatetimeIndex(ds.indexes["time"])
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    full = pd.date_range(start, end, freq="h", inclusive="left").tz_convert(
        "UTC"
    )
    ds = ds.assign_coords(time=t).reindex(time=full)

    df = ds.to_dataframe().reset_index().set_index("time").sort_index()
    return df


def _xee_fetch_site_dem(
    *,
    dem_asset: str,
    lat: float,
    lon: float,
    ee_init_kwargs: dict | None = None,
) -> float:
    """Nearest-neighbour DEM height at a point (meters)."""

    img = ee.Image(dem_asset)
    geom = ee.Geometry.Point(float(lon), float(lat))

    ds = _ee_open_dataset(
        src=img, geometry=geom, scale=0.1, ee_init_kwargs=ee_init_kwargs
    )

    if not ds.data_vars:
        raise RuntimeError("DEM dataset returned no variables.")
    band = next(iter(ds.data_vars))
    da = ds[band]
    for d in ("lat", "lon"):
        if d in da.dims and da.sizes.get(d, 1) == 1:
            da = da.squeeze(d, drop=True)
    return float(da.values)


def _xee_fetch_bbox_hourly(
    *,
    collection_id: str,
    bands: tuple[str, ...],
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    ee_init_kwargs: dict | None = None,
) -> "xr.Dataset":
    """Hourly bbox cube → xarray.Dataset on native ERA5 grid."""
    import ee  # type: ignore

    try:
        ee.Initialize(**(ee_init_kwargs or {}))
    except Exception:
        raise RuntimeError("Please initialise Earth Engine API before use.")

    ic = ee.ImageCollection(collection_id).filterDate(
        start.isoformat(), end.isoformat()
    )
    proj = ic.first().select(0).projection()
    geom = ee.Geometry.Rectangle(
        float(min_lon), float(min_lat), float(max_lon), float(max_lat)
    )

    ds = _ee_open_dataset(
        src=ic,
        geometry=geom,
        bands=bands,
        scale=0.1,
        projection=proj,
        ee_init_kwargs=ee_init_kwargs,
    )

    # Sort lat/lon ascending for consistency
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")

    # Ensure UTC hourly continuity
    t = pd.DatetimeIndex(ds.indexes["time"])
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    full = pd.date_range(start, end, freq="H", inclusive="left").tz_convert(
        "UTC"
    )
    ds = ds.assign_coords(time=t).reindex(time=full)
    return ds


def _xee_fetch_bbox_dem(
    *,
    dem_asset: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    ee_init_kwargs: dict | None = None,
) -> "xr.DataArray":
    """Static DEM grid over bbox (lat, lon)."""

    img = ee.Image(dem_asset)
    geom = ee.Geometry.Rectangle(
        float(min_lon), float(min_lat), float(max_lon), float(max_lat)
    )

    ds = _ee_open_dataset(
        src=img, geometry=geom, scale=0.1, ee_init_kwargs=ee_init_kwargs
    )

    if not ds.data_vars:
        raise RuntimeError("DEM dataset returned no variables.")
    band = next(iter(ds.data_vars))
    da = ds[band].rename("DEM")
    if "lat" in da.coords:
        da = da.sortby("lat")
    if "lon" in da.coords:
        da = da.sortby("lon")
    da.attrs["units"] = "m"
    return da
