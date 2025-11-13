"""Base model spec: required bands and postprocessing hooks."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd
import xarray as xr


@runtime_checkable
class ModelSpec(Protocol):
    """Protocol for model handlers.

    A model defines which hourly ERA5-Land bands are required and how to
    postprocess the hourly site/bbox data into its final shape.
    """

    name: str
    required_bands: tuple[str, ...]

    def postprocess_site(
        self, df_hourly: pd.DataFrame, dem_value: float, tz: str
    ) -> pd.DataFrame:
        """Transform a site hourly dataframe into model output."""

    def postprocess_area(
        self, ds_hourly: xr.Dataset, dem_grid: xr.DataArray, tz: str
    ) -> xr.Dataset:
        """Transform an area hourly dataset into model output."""

