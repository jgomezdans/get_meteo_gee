# meteo_data_gee

`meteo_data_gee` is a Python package to fetch and pre-process
meteorological variables from **ERA5-Land** and **DEM products** directly
from **Google Earth Engine (GEE)** using the [`xee`](https://github.com/xgcm/xee)
backend. The package returns `xarray.Dataset` or `pandas.DataFrame`
objects suitable for use in crop and ecosystem models such as
**WOFOST**, **D&B**, and others.

It is designed to work with both **site-level (point)** and
**area-level (bounding box)** queries, automatically handling:
- Variable selection (2 m temperature, dewpoint, precipitation, etc.)
- Temporal aggregation (hourly, daily)
- Local timezone conversion (this is important if you want to compare to local meteo stations that are not in UTC)
- Downwelling shortwave and longwave fluxes
- DEM sampling (COPERNICUS DEM GLO30 by default; GTOPO30 optional)

---

## Features

- Fetch ERA5-Land hourly meteorological data from GEE
- Aggregate to daily values following WOFOST conventions
- Automatic computation of:
  - `TMIN_C`, `TMAX_C`
  - Daily `RAIN_mm`
  - `VAP_kPa` from 2 m dewpoint
  - Mean daily wind speed
  - Daily shortwave radiation (J cm⁻² day⁻¹ and W m⁻² day⁻¹)
- Fetch DEM elevation at site or for bounding boxes
- Supports both **nearest-neighbour (site)** and **area grid (bbox)**
- It should be possible to extend to other sources of data that might not be in GEE in the future
- You can easily add new models by defining variable mappings and aggregation rules.
---

## Installation

`meteo_data_gee` requires Python ≥ 3.11 and access to an authenticated
Google Earth Engine account.

Clone and install in editable mode:

```bash
git clone https://github.com/jgomezdans/meteo_data_gee.git
cd meteo_data_gee
pip install -e .
```

The package uses `numpy`, `pandas`,`xee`, `xarray`, and `earthengine-api`, which will be
installed automatically.

You must initialise your GEE project credentials before use:

```bash
earthengine authenticate
```

---

## Usage

###  Site-level meteorology

**Note**: replace `ee-yourproject` with your actual GEE project name.

```python
from meteo_data_gee import get_meteo_data

loc = {"lat": 51.8089, "lon": -0.3566}
start, end = "2020-06-01", "2020-06-05"

df = get_meteo_data(
    loc,
    model="wofost",
    start=start,
    end=end,
    ee_project="ee-yourproject",
)
print(df.head())
```

Output: a daily `pandas.DataFrame` with standardised columns:

| DAY | TMIN_C | TMAX_C | VAP_kPa | WIND_ms | RAIN_mm | IRRAD_W_m2_day | DEM_m |
|-----|---------|---------|----------|----------|----------|----------------|--------|

---

### Bounding box aggregation

```python
bbox = {
    "min_lon": -0.7,
    "min_lat": 51.5,
    "max_lon": -0.1,
    "max_lat": 52.0,
}

ds = get_meteo_data(
    bbox,
    model="wofost",
    start=start,
    end=end,
    ee_project="ee-yourproject",
)

print(ds)
```

This returns an `xarray.Dataset` with daily aggregated fields across
the requested area.

---

## Notes

- Default DEM: `COPERNICUS/DEM/GLO30`
- Optional DEM: `USGS/GTOPO30`
- Default timezone: UTC
- Maximum bounding box size: **< 200 × 200 km**
- For WOFOST Vapour Pressure applications, mean daily aggregation is used.
- Longwave radiation corresponds to the ERA5-Land variable
  `surface_thermal_radiation_downwards`.

---



## License

GNU Affero General Public License v3.0
Developed by José Gómez-Dans (King's College London).
