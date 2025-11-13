"""Data source and band registry for ERA5-Land and DEM.

This module centralizes Earth Engine dataset identifiers and band names
used by models. Keeping this mapping here makes it easy to extend or
swap sources later with minimal changes elsewhere.
"""

ERA5L_ID = "ECMWF/ERA5_LAND/HOURLY"

# Default DEM per your decision. GTOPO30 is available as an option.
DEM_DEFAULT = "COPERNICUS/DEM/GLO30"
DEM_FALLBACK = "USGS/GTOPO30"

# Canonical band names we use throughout the package.
BANDS_COMMON = {
    "t2m": "temperature_2m",  # K
    "td2m": "dewpoint_temperature_2m",  # K
    "sw": "surface_solar_radiation_downwards_hourly",  # J m-2 hr-1
    "lw": "surface_thermal_radiation_downwards",  # J m-2 hr-1
    "tp": "total_precipitation_hourly",  # m hr-1
    "u10": "u_component_of_wind_10m",  # m s-1
    "v10": "v_component_of_wind_10m",  # m s-1
    "soilT3": "soil_temperature_level_3",  # K
}

# Per-model band requirements (only fetch what we need).
REQUIRED_BANDS = {
    "db": (
        BANDS_COMMON["t2m"],
        BANDS_COMMON["soilT3"],
        BANDS_COMMON["sw"],
        BANDS_COMMON["lw"],
        BANDS_COMMON["tp"],
        BANDS_COMMON["u10"],
        BANDS_COMMON["v10"],
    ),
    "wofost": (
        BANDS_COMMON["t2m"],
        BANDS_COMMON["td2m"],
        BANDS_COMMON["sw"],
        BANDS_COMMON["tp"],
        BANDS_COMMON["u10"],
        BANDS_COMMON["v10"],
    ),
}
