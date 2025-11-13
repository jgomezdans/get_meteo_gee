"""Model registry for meteo_data_gee."""

from .db import DBSpec
from .wofost import WofostSpec

MODELS = {
    "db": DBSpec(),
    "wofost": WofostSpec(),
}

