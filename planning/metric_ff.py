import config as CONFIG
from config import ErrorFlag
import subprocess

import os
from pathlib import Path


class MetricFF:
    """
    Create MetricFF for polycraft
    Where METRIC_FF_PATH must be updated in the config.py file in order to work
    """

    def __init__(self):
        self.path = CONFIG.METRIC_FF_PATH
        self.error_flag = ErrorFlag.NO_ERROR
        self.explored_states = -1

    def create_plan(
        self, domain: str, problem: str, timeout: int = 60, flag: str = ""
    ) -> list:
        """
        Create a plan for the given domain and problem
        :param domain: the domain file - must be located in the planning folder
        :param problem: the problem file - must be located in the planning folder
        :param timeout: the timeout for the planner in seconds
        """

        self.error_flag = ErrorFlag.NO_ERROR

        domain = Path(domain).absolute()
        problem = Path(problem).absolute()

        # Check if the domain and problem files exist
        if not os.path.exists(domain):
            raise Exception("Domain file not found")
        if not os.path.exists(problem):
            raise Exception("Problem file not found")

        original_dir = os.getcwd()
        os.chdir(self.path)
        cmd = f"./ff -o {domain} -f {problem} -s 0"

        if flag:
            cmd += f" {flag}"

        planner = subprocess.Popen(
            "exec " + cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            planner.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # print(f"Can't find a plan in {timeout} seconds")
            planner.kill()
            self.error_flag = ErrorFlag.TIMEOUT
            return []
        finally:
            os.chdir(original_dir)

        exception_flag = None
        for exception_flag in planner.stderr:
            break
        if exception_flag:
            planner.kill()
            self.error_flag = ErrorFlag.ERROR
            raise Exception(f"unknowned error for {domain} {problem}")

        plan = []
        for line in planner.stdout:
            if "found legal plan as follows" in str(line):
                line = str(planner.stdout.readline())
                while "b'\\n'" not in line:
                    try:
                        start = line.index(":") + 2
                        line = line[start:-3]
                        plan.append(f"({line.lower()})")
                        # print(line)
                    except:
                        pass
                    line = str(planner.stdout.readline())
                break
            elif any(
                phrase in str(line)
                for phrase in [
                    "unsolvable",
                    "goal not fulfilled",
                    "No plan will solve it",
                ]
            ):
                # print("Problem unsolvable")
                self.error_flag = ErrorFlag.NO_SOLUTION
                break

        planner.kill()

        return plan
