import os
from enum import Enum


class ErrorFlag(Enum):
    NO_ERROR = 0
    ERROR = -1
    FOUND_BY_SHORTEN = 0.5
    NO_SOLUTION = 1
    TIMEOUT = 2
    INVALID_PLAN = 3


PROJECT_PATH = ".."
PAL_PATH = f"{PROJECT_PATH}/pal"
DEFUALT_DOMAIN_PATH = f"{PAL_PATH}/available_tests/pogo_nonov.json"
EVALUATION_DOMAINS_PATH = [
    f"{PAL_PATH}/available_tests/pogo_nov_lvl-0_type-1.json",
    f"{PAL_PATH}/available_tests/pogo_nov_lvl-0_type-2.json",
    f"{PAL_PATH}/available_tests/pogo_nov_lvl-1_type-1.json",
    f"{PAL_PATH}/available_tests/pogo_nov_lvl-2_type-1b.json",
    f"{PAL_PATH}/available_tests/pogo_nov_lvl-3_type-1.json",
]
ENHSP_PATH = f"{PROJECT_PATH}/ENHSP"
METRIC_FF_PATH = f"{PROJECT_PATH}/METRIC_FF"
NYX_PATH = f"{PROJECT_PATH}/nyx_base"
NSAM_PATH = f"{PROJECT_PATH}/sam_learning"

os.environ["CONVEX_HULL_ERROR_PATH"] = "temp_files/ch_error.txt"
os.environ["EPSILON"] = "0.1"

VALIDATOR_DIRECTORY = f"{PROJECT_PATH}/VAL"
