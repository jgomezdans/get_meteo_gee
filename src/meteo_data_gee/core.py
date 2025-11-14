"""Core public API and adapters to GEE/xee.

This module exposes ``get_meteo_data`` and hides the details of fetching
hourly ERA5-Land + DEM. Two small adapter functions are the only places
that touch xee/GEE directly. Everything else (validation, aggregation,
units, tz handling) is framework-agnostic.
"""

from __future__ import annotations

import logging
from typing import Iterable

import ee  # type: ignore
import numpy as np
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


# Basic configuration for logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Define log message format
)

# Create a logger object
logger = logging.getLogger(
    __name__
)  # Use __name__ to get a logger named after the current module


def _dem_image_and_band(dem_asset: str) -> tuple["ee.Image", str]:
    """Return a single-image mosaic and the band name for a DEM asset."""
    import ee  # type: ignore

    ic = ee.ImageCollection(dem_asset)
    # Map asset → band name
    if "COPERNICUS/DEM" in dem_asset:
        band = "DEM"
    elif "USGS/GTOPO30" in dem_asset:
        band = "elevation"
    else:
        # Try to infer the first band of the first image if unknown
        first = ee.Image(ic.first())
        band = ee.List(first.bandNames()).get(0).getInfo()

    # Mosaic all tiles into a single Image, then select the band
    img = ic.select([band]).mosaic()
    return img, band


# ------------------------- Public entry point -------------------------------


# def get_meteo_data(
#     loc: Iterable[float],
#     model: str,
#     ee_project: str,
#     *,
#     year: int | None = None,
#     start: pd.Timestamp | None = None,
#     end: pd.Timestamp | None = None,
#     tz: str = "UTC",
#     variables: tuple[str, ...] | None = None,
#     dem_asset: str = DEM_DEFAULT,
#     ee_opt_url: str | None = (
#         "https://earthengine-highvolume.googleapis.com"
#     ),
# ) -> pd.DataFrame | xr.Dataset:
#     """Fetch meteo / fire danger data via xee/GEE and postprocess by model.

#     Parameters
#     ----------
#     loc
#         (lat, lon) for a site, or (min_lon, min_lat, max_lon, max_lat) for
#         a bounding box. Sites use nearest-neighbour sampling; NO SPATIAL
#         BUFFERING!.
#     year
#         Calendar year (UTC bounds Jan 1 00:00 to Jan 1 next year 00:00).
#     model
#         One of:
#         - "db"        : hourly D&B indices (ERA5-Land hourly).
#         - "wofost"    : daily aggregates derived from ERA5-Land hourly.
#         - "firedanger": daily fire danger indices (CEMS Fire).
#     tz
#         IANA timezone for daily aggregation windows where relevant
#         (e.g. WOFOST). Ignored for pure-hourly models and for models
#         that are already daily (FireDanger).
#     variables
#         Optional subset of variables to fetch (must be a subset of the
#         model's required bands). Most users should leave this as None.
#     dem_asset
#         DEM asset to use. Defaults to "COPERNICUS/DEM/GLO30". The optional
#         fallback is "USGS/GTOPO30".

#     Returns
#     -------
#     pandas.DataFrame or xarray.Dataset
#         Site returns a DataFrame. BBox returns a Dataset.

#     Notes
#     -----
#     This function delegates GEE/xee access to small adapters. Models may
#     use different collections (e.g. ERA5-Land hourly vs CEMS Fire daily)
#     and different temporal logic (hourly vs daily).
#     """
#     if year is not None and (start is not None or end is not None):
#         msg = "Specify either year or start/end, not both."
#         raise ValueError(msg)
#     if year is not None:
#         start, end = year_bounds_utc(year)
#     elif start is None or end is None:
#         msg = "Must specify either year or both start and end."
#         raise ValueError(msg)

#         # Coerce to pandas Timestamp
#         start = pd.to_datetime(start)
#         end = pd.to_datetime(end)
        
#     # Normalise to UTC tz-aware timestamps
#     if start.tzinfo is None or start.tzinfo.utcoffset(start) is None:
#         start = start.tz_localize("UTC")
#         end = end.tz_localize("UTC")
#     else:
#         start = start.tz_convert("UTC")
#         end = end.tz_convert("UTC")

#     logger.debug(
#         "Fetching meteo data for loc=%s, model=%s, start=%s, end=%s, tz=%s",
#         loc,
#         model,
#         start,
#         end,
#         tz,
#     )
#     model_key = model.lower().strip()

#     if model_key not in MODELS:
#         msg = (
#             f"Unknown model '{model}'. Use 'db', 'wofost', or 'firedanger'."
#         )
#         raise ValueError(msg)
#     mspec = MODELS[model_key]

#     # Collection and temporal frequency come from the model spec.
#     collection_id = getattr(mspec, "collection_id", ERA5L_ID)
#     frequency = getattr(mspec, "frequency", "hourly")

#     req = set(mspec.required_bands)
#     if variables is not None:
#         subset = set(variables)
#         if not subset.issubset(req):
#             msg = "variables must be a subset of the model's required bands."
#             raise ValueError(msg)
#         bands = tuple(v for v in mspec.required_bands if v in subset)
#     else:
#         bands = mspec.required_bands

#     if not bands:
#         raise ValueError("No bands to fetch (empty selection).")

#     ee_init_kwargs: dict[str, object] = {}
#     if ee_project is not None:
#         ee_init_kwargs["project"] = ee_project
#     if ee_opt_url is not None:
#         ee_init_kwargs["opt_url"] = ee_opt_url

#     # ------------------------- Site path ---------------------------------
#     if is_site(loc):
#         lat, lon = parse_site(loc)
#         logger.info(
#             "Fetching site data at lat=%.4f, lon=%.4f (freq=%s)",
#             lat,
#             lon,
#             frequency,
#         )

#         if frequency == "hourly":
#             # ERA5-Land-style hourly collections (db, wofost)
#             df_hourly = _xee_fetch_site_hourly(
#                 collection_id=collection_id,
#                 bands=bands,
#                 lat=lat,
#                 lon=lon,
#                 start=start,
#                 end=end,
#                 ee_init_kwargs=ee_init_kwargs,
#             )
#             logger.info("Got hourly data with %d timestamps", len(df_hourly))
#             logger.info(
#                 "Fetching site DEM at lat=%.4f, lon=%.4f", lat, lon
#             )
#             dem_value = _xee_fetch_site_dem(
#                 dem_asset=dem_asset,
#                 lat=lat,
#                 lon=lon,
#                 ee_init_kwargs=ee_init_kwargs,
#             )
#             logger.info("Got DEM value %.2f m", dem_value)
#             return mspec.postprocess_site(df_hourly, dem_value, tz)

#         # Daily collections (e.g. FireDanger)
#         ds_daily = _xee_fetch_site_daily(
#             collection_id=collection_id,
#             bands=bands,
#             lat=lat,
#             lon=lon,
#             start=start,
#             end=end,
#             ee_init_kwargs=ee_init_kwargs,
#         )
#         logger.info(
#             "Got daily data with %d timestamps", ds_daily.sizes.get("time", 0)
#         )
#         logger.info("Fetching site DEM at lat=%.4f, lon=%.4f", lat, lon)
#         dem_value = _xee_fetch_site_dem(
#             dem_asset=dem_asset,
#             lat=lat,
#             lon=lon,
#             ee_init_kwargs=ee_init_kwargs,
#         )
#         logger.info("Got DEM value %.2f m", dem_value)
#         return mspec.postprocess_site(ds_daily, dem_value, tz)

#     # ------------------------- BBox path ---------------------------------
#     min_lon, min_lat, max_lon, max_lat = parse_bbox(loc)
#     w_km, h_km = bbox_size_km(min_lon, min_lat, max_lon, max_lat)
#     if w_km > 200.0 or h_km > 200.0:
#         msg = (
#             "bbox must be <= 200 km in width and height. "
#             f"Got {w_km:.1f} x {h_km:.1f} km."
#         )
#         raise ValueError(msg)

#     logger.info(
#         "Fetching bbox data [%f,%f,%f,%f] (freq=%s)",
#         min_lon,
#         min_lat,
#         max_lon,
#         max_lat,
#         frequency,
#     )

#     if frequency == "hourly":
#         ds_hourly = _xee_fetch_bbox_hourly(
#             collection_id=collection_id,
#             bands=bands,
#             min_lon=min_lon,
#             min_lat=min_lat,
#             max_lon=max_lon,
#             max_lat=max_lat,
#             start=start,
#             end=end,
#             ee_init_kwargs=ee_init_kwargs,
#         )
#         logger.info(
#             "Got hourly bbox data with %d timestamps, "
#             "%d lat, %d lon",
#             ds_hourly.sizes.get("time", 0),
#             ds_hourly.sizes.get("lat", 0),
#             ds_hourly.sizes.get("lon", 0),
#         )
#         dem_grid = _xee_fetch_bbox_dem_on_grid(
#             dem_asset=dem_asset,
#             lats=ds_hourly["lat"].values,
#             lons=ds_hourly["lon"].values,
#             ee_init_kwargs=ee_init_kwargs,
#         )
#         return mspec.postprocess_area(ds_hourly, dem_grid, tz)

#     # Daily collections (e.g. FireDanger)
#     ds_daily = _xee_fetch_bbox_daily(
#         collection_id=collection_id,
#         bands=bands,
#         min_lon=min_lon,
#         min_lat=min_lat,
#         max_lon=max_lon,
#         max_lat=max_lat,
#         start=start,
#         end=end,
#         ee_init_kwargs=ee_init_kwargs,
#     )
#     logger.info(
#         "Got daily bbox data with %d timestamps, "
#         "%d lat, %d lon",
#         ds_daily.sizes.get("time", 0),
#         ds_daily.sizes.get("lat", 0),
#         ds_daily.sizes.get("lon", 0),
#     )
#     dem_grid = _xee_fetch_bbox_dem_on_grid(
#         dem_asset=dem_asset,
#         lats=ds_daily["lat"].values,
#         lons=ds_daily["lon"].values,
#         ee_init_kwargs=ee_init_kwargs,
#     )
#     return mspec.postprocess_area(ds_daily, dem_grid, tz)
def get_meteo_data(
    loc: Iterable[float],
    model: str,
    ee_project: str,
    *,
    year: int | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    tz: str = "UTC",
    variables: tuple[str, ...] | None = None,
    dem_asset: str = DEM_DEFAULT,
    ee_opt_url: str | None = (
        "https://earthengine-highvolume.googleapis.com"
    ),
) -> pd.DataFrame | xr.Dataset:
    """Fetch meteo / fire danger data via xee/GEE and postprocess by model.

    Parameters
    ----------
    loc
        (lat, lon) for a site, or (min_lon, min_lat, max_lon, max_lat) for
        a bounding box. Sites use nearest-neighbour sampling; NO SPATIAL
        BUFFERING!.
    year
        Calendar year (UTC bounds Jan 1 00:00 to Jan 1 next year 00:00).
    model
        One of:
        - "db"        : hourly D&B indices (ERA5-Land hourly).
        - "wofost"    : daily aggregates derived from ERA5-Land hourly.
        - "firedanger": daily fire danger indices (CEMS Fire).
    tz
        IANA timezone for daily aggregation windows where relevant.
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
    """
    # ------------------------------------------------------------------
    # Resolve time window: either year OR explicit start/end.
    # Allow strings or Timestamps for start/end.
    # ------------------------------------------------------------------
    if year is not None and (start is not None or end is not None):
        msg = "Specify either year or start/end, not both."
        raise ValueError(msg)

    if year is not None:
        start, end = year_bounds_utc(year)
    else:
        if start is None or end is None:
            msg = "Must specify either year or both start and end."
            raise ValueError(msg)
        # Coerce to pandas Timestamp
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

    # Normalise to UTC tz-aware timestamps
    if start.tzinfo is None or start.tzinfo.utcoffset(start) is None:
        start = start.tz_localize("UTC")
        end = end.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
        end = end.tz_convert("UTC")

    logger.debug(
        "Fetching meteo data for loc=%s, model=%s, start=%s, end=%s, tz=%s",
        loc,
        model,
        start,
        end,
        tz,
    )
    model_key = model.lower().strip()

    if model_key not in MODELS:
        msg = (
            f"Unknown model '{model}'. Use 'db', 'wofost', or 'firedanger'."
        )
        raise ValueError(msg)
    mspec = MODELS[model_key]

    # Collection and temporal frequency come from the model spec.
    collection_id = getattr(mspec, "collection_id", ERA5L_ID)
    frequency = getattr(mspec, "frequency", "hourly")

    req = set(mspec.required_bands)
    if variables is not None:
        subset = set(variables)
        if not subset.issubset(req):
            msg = "variables must be a subset of the model's required bands."
            raise ValueError(msg)
        bands = tuple(v for v in mspec.required_bands if v in subset)
    else:
        bands = mspec.required_bands

    if not bands:
        raise ValueError("No bands to fetch (empty selection).")

    ee_init_kwargs: dict[str, object] = {}
    if ee_project is not None:
        ee_init_kwargs["project"] = ee_project
    if ee_opt_url is not None:
        ee_init_kwargs["opt_url"] = ee_opt_url

    # ------------------------- Site path ---------------------------------
    if is_site(loc):
        lat, lon = parse_site(loc)
        logger.info(
            "Fetching site data at lat=%.4f, lon=%.4f (freq=%s)",
            lat,
            lon,
            frequency,
        )

        if frequency == "hourly":
            df_hourly = _xee_fetch_site_hourly(
                collection_id=collection_id,
                bands=bands,
                lat=lat,
                lon=lon,
                start=start,
                end=end,
                ee_init_kwargs=ee_init_kwargs,
            )
            logger.info("Got hourly data with %d timestamps", len(df_hourly))
            logger.info(
                "Fetching site DEM at lat=%.4f, lon=%.4f", lat, lon
            )
            dem_value = _xee_fetch_site_dem(
                dem_asset=dem_asset,
                lat=lat,
                lon=lon,
                ee_init_kwargs=ee_init_kwargs,
            )
            logger.info("Got DEM value %.2f m", dem_value)
            return mspec.postprocess_site(df_hourly, dem_value, tz)

        ds_daily = _xee_fetch_site_daily(
            collection_id=collection_id,
            bands=bands,
            lat=lat,
            lon=lon,
            start=start,
            end=end,
            ee_init_kwargs=ee_init_kwargs,
        )
        logger.info(
            "Got daily data with %d timestamps",
            ds_daily.sizes.get("time", 0),
        )
        logger.info("Fetching site DEM at lat=%.4f, lon=%.4f", lat, lon)
        dem_value = _xee_fetch_site_dem(
            dem_asset=dem_asset,
            lat=lat,
            lon=lon,
            ee_init_kwargs=ee_init_kwargs,
        )
        logger.info("Got DEM value %.2f m", dem_value)
        return mspec.postprocess_site(ds_daily, dem_value, tz)

    # ------------------------- BBox path ---------------------------------
    min_lon, min_lat, max_lon, max_lat = parse_bbox(loc)
    w_km, h_km = bbox_size_km(min_lon, min_lat, max_lon, max_lat)
    if w_km > 200.0 or h_km > 200.0:
        msg = (
            "bbox must be <= 200 km in width and height. "
            f"Got {w_km:.1f} x {h_km:.1f} km."
        )
        raise ValueError(msg)

    logger.info(
        "Fetching bbox data [%f,%f,%f,%f] (freq=%s)",
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        frequency,
    )

    if frequency == "hourly":
        ds_hourly = _xee_fetch_bbox_hourly(
            collection_id=collection_id,
            bands=bands,
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            start=start,
            end=end,
            ee_init_kwargs=ee_init_kwargs,
        )
        logger.info(
            "Got hourly bbox data with %d timestamps, %d lat, %d lon",
            ds_hourly.sizes.get("time", 0),
            ds_hourly.sizes.get("lat", 0),
            ds_hourly.sizes.get("lon", 0),
        )
        dem_grid = _xee_fetch_bbox_dem_on_grid(
            dem_asset=dem_asset,
            lats=ds_hourly["lat"].values,
            lons=ds_hourly["lon"].values,
            ee_init_kwargs=ee_init_kwargs,
        )
        return mspec.postprocess_area(ds_hourly, dem_grid, tz)

    ds_daily = _xee_fetch_bbox_daily(
        collection_id=collection_id,
        bands=bands,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        start=start,
        end=end,
        ee_init_kwargs=ee_init_kwargs,
    )
    logger.info(
        "Got daily bbox data with %d timestamps, %d lat, %d lon",
        ds_daily.sizes.get("time", 0),
        ds_daily.sizes.get("lat", 0),
        ds_daily.sizes.get("lon", 0),
    )
    dem_grid = _xee_fetch_bbox_dem_on_grid(
        dem_asset=dem_asset,
        lats=ds_daily["lat"].values,
        lons=ds_daily["lon"].values,
        ee_init_kwargs=ee_init_kwargs,
    )
    return mspec.postprocess_area(ds_daily, dem_grid, tz)


# ------------------------- xee adapters -----------------------


def _ee_open_dataset(
    *,
    src: "str | ee.ImageCollection | ee.Image",
    geometry: "ee.Geometry",
    bands: tuple[str, ...] | None = None,
    scale: float = 0.1,
    projection: "ee.Projection | None" = None,
    ee_init_kwargs: dict | None = None,
) -> "xr.Dataset":
    """Open an EE ImageCollection/Image/asset-id as xarray.Dataset.

    We prefer passing an ImageCollection (even for DEM) and select bands
    server-side. If `src` is a string, it's treated as an EE asset id.
    """
    import ee  # type: ignore
    import xarray as xr

    try:
        ee.Initialize(**(ee_init_kwargs or {}))
    except Exception:
        raise RuntimeError("Please initialise Earth Engine API before use.")

    if isinstance(src, str):
        obj = ee.ImageCollection(src)
    else:
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

    if bands:
        keep = [b for b in bands if b in ds.data_vars]
        ds = ds[keep] if keep else ds

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
    """Nearest-neighbour hourly site series → pandas.DataFrame."""
    import ee  # type: ignore

    try:
        ee.Initialize(**(ee_init_kwargs or {}))
    except Exception:
        raise RuntimeError("Please initialise Earth Engine API before use.")

    ic = (
        ee.ImageCollection(collection_id)
        .filterDate(start.isoformat(), end.isoformat())
        .select(list(bands))
    )  # <-- server-side subset
    proj = ic.first().select(0).projection()
    geom = ee.Geometry.Point(float(lon), float(lat))

    ds = _ee_open_dataset(
        src=ic,
        geometry=geom,
        bands=bands,  # <-- client-side guard
        scale=0.1,
        projection=proj,
        ee_init_kwargs=ee_init_kwargs,
    )

    for d in ("lat", "lon"):
        if d in ds.dims and ds.sizes.get(d, 1) == 1:
            ds = ds.squeeze(d, drop=True)

    t = pd.DatetimeIndex(ds.indexes["time"])
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    full = pd.date_range(start, end, freq="h", inclusive="left").tz_convert(
        "UTC"
    )
    ds = ds.assign_coords(time=t).reindex(time=full)

    df = ds.to_dataframe().reset_index().set_index("time").sort_index()

    if bands:
        df = df[[b for b in bands if b in df.columns]]

    return df


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

    ic = (
        ee.ImageCollection(collection_id)
        .filterDate(start.isoformat(), end.isoformat())
        .select(list(bands))
    )  # server-side subset
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

    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")

    t = pd.DatetimeIndex(ds.indexes["time"])
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    full = pd.date_range(start, end, freq="h", inclusive="left").tz_convert(
        "UTC"
    )
    ds = ds.assign_coords(time=t).reindex(time=full)

    # Final trim to requested bands (defensive)
    if bands:
        keep = [b for b in bands if b in ds.data_vars]
        ds = ds[keep] if keep else ds

    return ds


def _xee_fetch_site_dem(
    *,
    dem_asset: str,
    lat: float,
    lon: float,
    ee_init_kwargs: dict | None = None,
) -> float:
    """Nearest-neighbour DEM height at a point (meters).

    Fast path: sample the mosaicked DEM Image at the point using EE API.
    Avoids passing a huge ImageCollection to xee.
    """
    import ee  # type: ignore

    try:
        ee.Initialize(**(ee_init_kwargs or {}))
    except Exception:
        raise RuntimeError("Please initialise Earth Engine API before use.")

    img, band = _dem_image_and_band(dem_asset)
    geom = ee.Geometry.Point(float(lon), float(lat))

    # Use a small scale; DEM native ~30m for GLO30. Nearest by default.
    val = (
        img.sample(region=geom, scale=30, numPixels=1)
        .first()
        .get(band)
        .getInfo()
    )
    return float(val)


def _xee_fetch_bbox_dem_on_grid(
    *,
    dem_asset: str,
    lats: np.ndarray,
    lons: np.ndarray,
    ee_init_kwargs: dict | None = None,
) -> "xr.DataArray":
    import ee  # type: ignore
    import numpy as np
    import xarray as xr

    try:
        ee.Initialize(**(ee_init_kwargs or {}))
    except Exception:
        raise RuntimeError("Please initialise Earth Engine API before use.")

    # Mosaic DEM and choose band
    ic = ee.ImageCollection(dem_asset)
    if "COPERNICUS/DEM" in dem_asset:
        band = "DEM"
    elif "USGS/GTOPO30" in dem_asset:
        band = "elevation"
    else:
        first = ee.Image(ic.first())
        band = ee.List(first.bandNames()).get(0).getInfo()
    img = ic.select([band]).mosaic()

    # Build FeatureCollection of grid centres with (i=lat_idx, j=lon_idx)
    lat_arr = np.asarray(lats)
    lon_arr = np.asarray(lons)
    ny, nx = lat_arr.size, lon_arr.size

    feats = []
    for i, lat in enumerate(lat_arr):
        for j, lon in enumerate(lon_arr):
            geom = ee.Geometry.Point(float(lon), float(lat))
            feats.append(ee.Feature(geom, {"i": int(i), "j": int(j)}))
    fc = ee.FeatureCollection(feats)

    # >>> Preserve i/j: use sampleRegions, not sample
    samp = img.sampleRegions(
        collection=fc,
        properties=["i", "j"],
        scale=30,  # ~30 m for GLO30
        geometries=False,
    )

    # Bring results client-side
    recs_py = samp.getInfo()["features"]  # list of Features with properties

    # Reassemble to (lat, lon) grid
    arr = np.full((ny, nx), np.nan, dtype=float)
    for f in recs_py:
        props = f["properties"]
        i = int(props["i"])
        j = int(props["j"])
        v = props.get(band)
        if v is not None:
            arr[i, j] = float(v)

    da = xr.DataArray(
        arr,
        coords={"lat": lat_arr, "lon": lon_arr},
        dims=("lat", "lon"),
        name="DEM_m",
    )
    da.attrs["units"] = "m"
    return da

def _xee_fetch_site_daily(
    *,
    collection_id: str,
    bands: tuple[str, ...],
    lat: float,
    lon: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    ee_init_kwargs: dict | None = None,
) -> xr.Dataset:
    """Nearest-neighbour daily site series → xarray.Dataset.

    Similar to _xee_fetch_site_hourly but for daily collections:
    - no hourly reindex
    - just tidy the time coord.
    """
    import ee  # type: ignore

    try:
        ee.Initialize(**(ee_init_kwargs or {}))
    except Exception:
        raise RuntimeError("Please initialise Earth Engine API before use.")

    ic = (
        ee.ImageCollection(collection_id)
        .filterDate(start.isoformat(), end.isoformat())
        .select(list(bands))
    )
    proj = ic.first().select(0).projection()
    geom = ee.Geometry.Point(float(lon), float(lat))

    ds = _ee_open_dataset(
        src=ic,
        geometry=geom,
        bands=bands,
        scale=0.25,  # FireDanger native ~0.25 deg
        projection=proj,
        ee_init_kwargs=ee_init_kwargs,
    )

    for d in ("lat", "lon"):
        if d in ds.dims and ds.sizes.get(d, 1) == 1:
            ds = ds.squeeze(d, drop=True)

    # Tidy time coord
    t = pd.to_datetime(ds.indexes["time"])
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    ds = ds.assign_coords(time=t)
    return ds


def _xee_fetch_bbox_daily(
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
) -> xr.Dataset:
    """Daily bbox cube → xarray.Dataset for daily collections."""
    import ee  # type: ignore

    try:
        ee.Initialize(**(ee_init_kwargs or {}))
    except Exception:
        raise RuntimeError("Please initialise Earth Engine API before use.")

    ic = (
        ee.ImageCollection(collection_id)
        .filterDate(start.isoformat(), end.isoformat())
        .select(list(bands))
    )
    proj = ic.first().select(0).projection()
    geom = ee.Geometry.Rectangle(
        float(min_lon), float(min_lat), float(max_lon), float(max_lat)
    )

    ds = _ee_open_dataset(
        src=ic,
        geometry=geom,
        bands=bands,
        scale=0.25,  # native resolution
        projection=proj,
        ee_init_kwargs=ee_init_kwargs,
    )

    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")

    t = pd.to_datetime(ds.indexes["time"])
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    ds = ds.assign_coords(time=t)
    return ds
