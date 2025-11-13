"""WOFOST model: daily aggregates with tz-aware day windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..sources import BANDS_COMMON, REQUIRED_BANDS
from ..utils import dewpoint_to_vap_kpa, to_local_calendar_day_index


class WofostSpec:
    """WOFOST daily model spec.

    Aggregates hourly ERA5-Land to daily with local calendar days defined
    by the given tz (IANA). Produces standard WOFOST-like columns/vars,
    including dual irradiance units. Timezone normalisation is important if
    considering areas far away from UTC.
    """

    name = "wofost"
    required_bands = REQUIRED_BANDS["wofost"]

    def _agg_site(self, df: pd.DataFrame, tz: str) -> pd.DataFrame:
        """Aggregate a site's hourly dataframe to daily WOFOST outputs."""
        # Build helpers
        idx_utc = pd.DatetimeIndex(df.index)
        day_idx = to_local_calendar_day_index(idx_utc, tz)

        # Precompute pieces
        t2m = df[BANDS_COMMON["t2m"]].to_numpy()
        td2m = df[BANDS_COMMON["td2m"]].to_numpy()
        sw = df[BANDS_COMMON["sw"]].to_numpy()
        tp_mm = df[BANDS_COMMON["tp"]].to_numpy() * 1000.0
        u10 = df[BANDS_COMMON["u10"]].to_numpy()
        v10 = df[BANDS_COMMON["v10"]].to_numpy()

        wind = np.sqrt(u10 * u10 + v10 * v10)
        vap = dewpoint_to_vap_kpa(td2m)

        dfh = pd.DataFrame(
            {
                "DAY": day_idx,
                "T_K": t2m,
                "SW": sw,
                "RAIN_mm": tp_mm,
                "VAP_kPa": vap,
                "WIND_ms": wind,
            },
            index=idx_utc,
        )

        # Group by local calendar day
        grp = dfh.groupby("DAY", sort=True, as_index=True, dropna=False)

        tmax_c = grp["T_K"].max() - 273.15
        tmin_c = grp["T_K"].min() - 273.15
        irrad_j_m2 = grp["SW"].sum()
        rain_mm = grp["RAIN_mm"].sum()
        vap_kpa = grp["VAP_kPa"].mean()
        wind_ms = grp["WIND_ms"].mean()

        out = pd.DataFrame(
            {
                "IRRAD_J_cm2_day": irrad_j_m2 / 10000.0,
                "IRRAD_W_m2_day": irrad_j_m2 / 86400.0,
                "TMIN_C": tmin_c,
                "TMAX_C": tmax_c,
                "VAP_kPa": vap_kpa,
                "WIND_ms": wind_ms,
                "RAIN_mm": rain_mm,
            }
        )
        out.index.name = "DAY"
        return out

    def _agg_area(self, ds: xr.Dataset, tz: str) -> xr.Dataset:
        """Aggregate a bbox hourly dataset to daily along local days.

        We convert the coordinate "time" from UTC to the given tz, then
        groupby its local calendar date. xarray handles broadcasting.
        """
        ds_local = ds.copy()

        # xarray: convert the datetime coordinate to the target tz
        # and add a "day" index for grouping.
        time_local = ds_local.indexes["time"].tz_convert(tz)
        day = xr.DataArray(
            pd.DatetimeIndex(time_local.normalize()),
            dims=("time",),
            name="day",
        )

        # Scalars
        tmax_c = ds_local[BANDS_COMMON["t2m"]].groupby(day).max() - 273.15
        tmin_c = ds_local[BANDS_COMMON["t2m"]].groupby(day).min() - 273.15

        irrad_j_m2 = ds_local[BANDS_COMMON["sw"]].groupby(day).sum()
        rain_mm = ds_local[BANDS_COMMON["tp"]].groupby(day).sum() * 1000.0

        u10 = ds_local[BANDS_COMMON["u10"]]
        v10 = ds_local[BANDS_COMMON["v10"]]
        wind = xr.apply_ufunc(
            np.hypot, u10, v10, dask="parallelized", keep_attrs=True
        )
        wind_mean = wind.groupby(day).mean()

        vap_kpa = xr.apply_ufunc(
            dewpoint_to_vap_kpa,
            ds_local[BANDS_COMMON["td2m"]],
            dask="parallelized",
            keep_attrs=False,
        )
        vap_daily = vap_kpa.groupby(day).mean()

        ds_out = xr.Dataset(
            {
                "IRRAD_J_cm2_day": irrad_j_m2 / 10000.0,
                "IRRAD_W_m2_day": irrad_j_m2 / 86400.0,
                "TMIN_C": tmin_c,
                "TMAX_C": tmax_c,
                "VAP_kPa": vap_daily,
                "WIND_ms": wind_mean,
                "RAIN_mm": rain_mm,
            }
        )
        ds_out = (
            ds_out.rename({"group": "DAY"})
            if "group" in ds_out.dims
            else ds_out
        )
        ds_out = ds_out.assign_coords(
            DAY=ds_out.indexes["IRRAD_J_cm2_day"].tz_localize(None)
        )
        return ds_out

    def postprocess_site(
        self, df_hourly: pd.DataFrame, dem_value: float, tz: str
    ) -> pd.DataFrame:
        """Aggregate to daily and append DEM."""
        daily = self._agg_site(df_hourly, tz)
        daily["DEM_m"] = float(dem_value)
        return daily

    def postprocess_area(
        self, ds_hourly: xr.Dataset, dem_grid: xr.DataArray, tz: str
    ) -> xr.Dataset:
        """Aggregate to daily and append DEM grid."""
        daily = self._agg_area(ds_hourly, tz)
        dem_grid.name = "DEM_m"
        daily = daily.assign(DEM_m=dem_grid)
        return daily
