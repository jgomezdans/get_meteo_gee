"""D&B model: hourly outputs with ERA5-Land units (strictly hourly)."""

from __future__ import annotations

import pandas as pd
import xarray as xr

from ..sources import BANDS_COMMON, REQUIRED_BANDS


class DBSpec:
    """D&B hourly model spec.

    Outputs remain hourly. Units are preserved as in ERA5-Land except
    precipitation which is converted from m/hr to mm/hr. DEM is added.
    """

    name = "db"
    required_bands = REQUIRED_BANDS["db"]

    # Column naming used in outputs
    COLS = {
        BANDS_COMMON["t2m"]: "T",  # K
        BANDS_COMMON["soilT3"]: "SoilT",  # K
        BANDS_COMMON["sw"]: "Rsw",  # J m-2 hr-1
        BANDS_COMMON["lw"]: "LW downwelling",  # J m-2 hr-1
        BANDS_COMMON["tp"]: "P",  # mm hr-1 after conv
        BANDS_COMMON["u10"]: "U10",  # m s-1
        BANDS_COMMON["v10"]: "V10",  # m s-1
    }

    @staticmethod
    def _rename_and_units_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=DBSpec.COLS)
        if "P" in df.columns:
            df["P"] = df["P"] * 1000.0  # m/hr -> mm/hr
        return df

    @staticmethod
    def _rename_and_units_ds(ds: xr.Dataset) -> xr.Dataset:
        rename = {k: v for k, v in DBSpec.COLS.items() if k in ds}
        ds = ds.rename(rename)
        if "P" in ds:
            ds["P"] = ds["P"] * 1000.0
            ds["P"].attrs["units"] = "mm hr-1"
        return ds

    def postprocess_site(
        self, df_hourly: pd.DataFrame, dem_value: float, tz: str
    ) -> pd.DataFrame:
        """Rename columns, convert precip units, attach DEM."""
        out = self._rename_and_units_df(df_hourly.copy())
        out["DEM"] = float(dem_value)
        out.index.name = "time"
        return out

    def postprocess_area(
        self, ds_hourly: xr.Dataset, dem_grid: xr.DataArray, tz: str
    ) -> xr.Dataset:
        """Rename variables, convert precip units, attach DEM grid."""
        out = self._rename_and_units_ds(ds_hourly.copy())
        dem_grid.name = "DEM"
        dem_grid.attrs["units"] = "m"
        out = out.assign(DEM=dem_grid)
        return out
