"""Model registry for meteo_data_gee."""

from .db import DBSpec
from .wofost import WofostSpec
from .firedanger import FireDangerSpec

MODELS = {
    "db": DBSpec(),
    "wofost": WofostSpec(),
    "firedanger": FireDangerSpec(),

}

