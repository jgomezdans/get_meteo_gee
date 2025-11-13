"""Utilities: validation, geometry, time helpers, and unit converters."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Iterable

import numpy as np
import pandas as pd


def is_site(loc: Iterable[float]) -> bool:
    """Return True if ``loc`` looks like a site (lat, lon)."""
    try:
        seq = tuple(loc)
    except TypeError as exc:  # not iterable
        msg = "loc must be an iterable of floats."
        raise ValueError(msg) from exc
    if len(seq) == 2:
        return True
    if len(seq) == 4:
        return False
    msg = "loc must be (lat, lon) or (min_lon, min_lat, max_lon, max_lat)."
    raise ValueError(msg)


def parse_site(loc: Iterable[float]) -> tuple[float, float]:
    """Parse site into (lat, lon) floats with quick sanity checks."""
    lat, lon = (float(loc[0]), float(loc[1]))
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        msg = "lat/lon out of bounds."
        raise ValueError(msg)
    return lat, lon


def parse_bbox(loc: Iterable[float]) -> tuple[float, float, float, float]:
    """Parse bbox into (min_lon, min_lat, max_lon, max_lat) with checks."""
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in loc)
    if not (-180 <= min_lon < max_lon <= 180):
        msg = "bbox longitudes invalid."
        raise ValueError(msg)
    if not (-90 <= min_lat < max_lat <= 90):
        msg = "bbox latitudes invalid."
        raise ValueError(msg)
    return min_lon, min_lat, max_lon, max_lat


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def bbox_size_km(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> tuple[float, float]:
    """Approximate width and height of a bbox in km."""
    w = haversine_km(min_lat, min_lon, min_lat, max_lon)
    h = haversine_km(min_lat, min_lon, max_lat, min_lon)
    return w, h


def year_bounds_utc(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC start and end bounds for full calendar year."""
    start = pd.Timestamp(datetime(year, 1, 1, tzinfo=timezone.utc))
    end = pd.Timestamp(datetime(year + 1, 1, 1, tzinfo=timezone.utc))
    return start, end


def to_local_calendar_day_index(
    times_utc: pd.DatetimeIndex, tz: str
) -> pd.DatetimeIndex:
    """Convert UTC hourly index to local daily index for grouping.

    Parameters
    ----------
    times_utc
        Hourly, tz-aware UTC timestamps.
    tz
        IANA timezone name. If "UTC", no conversion.

    Returns
    -------
    DatetimeIndex
        Localized timestamps truncated to date (normalized).
    """
    idx_local = times_utc.tz_convert(tz)
    return idx_local.normalize()


def dewpoint_to_vap_kpa(tdew_K: np.ndarray) -> np.ndarray:
    """Convert dewpoint temperature (K) to vapor pressure (kPa).

    Uses the standard FAO-like formulation:
    ea = 0.6108 * exp(17.27 * T / (T + 237.3)) with T in degC.
    """
    t_c = tdew_K - 273.15
    num = 17.27 * t_c
    den = t_c + 237.3
    return 0.6108 * np.exp(num / den)
