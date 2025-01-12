import os

PAL_PATH = "/home/benjamin/Projects/pal"
DEFUALT_DOMAIN_PATH = f"{PAL_PATH}/available_tests/pogo_nonov.json"
EVALUATION_DOMAINS_PATH = [
    f"{PAL_PATH}/available_tests/pogo_nov_lvl-0_type-1.json",
    f"{PAL_PATH}/available_tests/pogo_nov_lvl-0_type-2.json",
    f"{PAL_PATH}/available_tests/pogo_nov_lvl-1_type-1.json",
    f"{PAL_PATH}/available_tests/pogo_nov_lvl-2_type-1b.json",
    f"{PAL_PATH}/available_tests/pogo_nov_lvl-3_type-1.json",
]
ENHSP_PATH = "/home/benjamin/Projects/ENHSP"
METRIC_FF_PATH = "/home/benjamin/Projects/METRIC_FF"
NYX_PATH = "/home/benjamin/Projects/nyx_base"
NSAM_PATH = "/home/benjamin/Projects/numeric-sam"

os.environ["CONVEX_HULL_ERROR_PATH"] = "temp_files/ch_error.txt"
os.environ["EPSILON"] = "0.1"

VALIDATOR_DIRECTORY = "/home/benjamin/Projects/VAL"
