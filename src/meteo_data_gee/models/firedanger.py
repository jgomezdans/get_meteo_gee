# src/meteo_data_gee/models/firedanger.py

from __future__ import annotations

import pandas as pd
import xarray as xr

from ..sources import FIREDANGER_BANDS, FIREDANGER_ID


class FireDangerSpec:
    """Model spec for Fire Danger indices (CEMS/ERA5-based).

    Data are already daily on input, so no temporal aggregation is done
    here. We simply tidy the time axis and add DEM where applicable.
    """

    name = "firedanger"
    required_bands: tuple[str, ...] = FIREDANGER_BANDS
    collection_id: str = FIREDANGER_ID
    frequency: str = "daily"  # used by `core` to pick fetcher

    # ------------------------------------------------------------------
    # site: Dataset (time, lat, lon=1) -> DataFrame (daily) + DEM
    # ------------------------------------------------------------------
    def postprocess_site(
        self, ds_daily: xr.Dataset, dem_value: float, tz: str
    ) -> pd.DataFrame:
        """Convert site Dataset to daily DataFrame and append DEM.

        Parameters
        ----------
        ds_daily : xarray.Dataset
            Daily dataset from FireDanger collection for a single site.
            For point sampling via xee, this is typically 1D on 'time'
            only (no 'lat'/'lon' dims).
        dem_value : float
            DEM height (m) at site.
        tz : str
            Unused for FireDanger (data already daily); kept for API
            compatibility.

        Returns
        -------
        pandas.DataFrame
            Daily FireDanger indices with a 'DEM_m' column. Index is
            tz-naive daily datetime (labelled 'DAY').
        """
        ds = ds_daily.copy()

        # Ensure a clean daily DatetimeIndex (tz-naive)
        idx = pd.to_datetime(ds.indexes["time"])
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        ds = ds.assign_coords(time=idx)

        # Convert to DataFrame, index by time
        df = ds.to_dataframe().reset_index().set_index("time").sort_index()

        # Make index a pure date (daily) and name it 'DAY'
        df.index = df.index.normalize()
        df.index.name = "DAY"

        # Append DEM
        df["DEM_m"] = float(dem_value)
        return df


    # ------------------------------------------------------------------
    # bbox: Dataset (time, lat, lon) -> Dataset + DEM grid
    # ------------------------------------------------------------------
    def postprocess_area(
        self, ds_daily: xr.Dataset, dem_grid: xr.DataArray, tz: str
    ) -> xr.Dataset:
        """Attach DEM to daily FireDanger cube.

        Parameters
        ----------
        ds_daily : xarray.Dataset
            Daily FireDanger dataset (time, lat, lon).
        dem_grid : xarray.DataArray
            DEM_m (lat, lon) aligned to the same grid.
        tz : str
            Unused for FireDanger (data already daily).

        Returns
        -------
        xarray.Dataset
            Same as input, with an added 'DEM_m' variable.
        """
        ds = ds_daily.copy()

        # Normalise time: tz-naive daily datetime
        idx = pd.to_datetime(ds.indexes["time"])
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        ds = ds.assign_coords(time=idx)

        # Attach DEM, broadcasting along time
        dem_aligned = dem_grid.rename("DEM_m")
        ds["DEM_m"] = dem_aligned
        return ds
