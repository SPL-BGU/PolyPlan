import config as CONFIG
import logging
import os
import subprocess
from pathlib import Path

VALID_PLAN = "Plan valid"
INAPPLICABLE_PLAN = "Plan failed to execute"
PLAN_APPLICABLE = "Plan executed successfully"
GOAL_NOT_REACHED = "Goal not satisfied"

MAX_RUNNING_TIME = 60

logger = logging.getLogger(__name__)


def run_validate_script(
    domain_file_path: Path, problem_file_path: Path, solution_file_path: Path
) -> bool:
    """Validates that the plan for the input problem.

    :param domain_file_path: the path to the domain file.
    :param problem_file_path: the path to the problem file.
    :param solution_file_path: the path to the solution file.
    :return: True if plan is valid, False otherwise."""

    original_dir = os.getcwd()
    os.chdir(CONFIG.VALIDATOR_DIRECTORY)
    logger.info("Running VAL to validate the plan's correctness.")
    validation_file_path = domain_file_path.parent / "validation_log.txt"
    run_command = (
        f"./Validate -v -t 0.1 {domain_file_path} {problem_file_path} "
        f"{solution_file_path} > {validation_file_path}"
    )
    try:
        subprocess.check_output(run_command, shell=True)

    except subprocess.CalledProcessError as e:
        logger.error(f"VAL returned status code {e.returncode}.")

    logger.info("Finished validating the solution file.")
    os.chdir(original_dir)

    with open(validation_file_path, "r") as validation_file:
        validation_file_content = validation_file.read()
        if VALID_PLAN in validation_file_content:
            return True
    return False
