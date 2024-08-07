import config as CONFIG
import subprocess

import os
from pathlib import Path


class NYX:
    """
    Create MetricFF for polycraft
    Where NYX_PATH must be updated in the config.py file in order to work
    """

    def __init__(self):
        self.path = CONFIG.NYX_PATH
        self.error_flag = 0  # -1: error, 0: no error, 1: no solution, 2: timeout
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

        self.error_flag = 0

        domain = Path(domain).absolute()
        problem = Path(problem).absolute()

        # Check if the domain and problem files exist
        if not os.path.exists(domain):
            raise Exception("Domain file not found")
        if not os.path.exists(problem):
            raise Exception("Problem file not found")

        original_dir = os.getcwd()
        os.chdir(self.path)

        cmd = f"python3.8 nyx.py {domain} {problem}"
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
            self.error_flag = 2
            return []
        finally:
            os.chdir(original_dir)

        exception_flag = None
        for exception_flag in planner.stderr:
            break
        if exception_flag:
            planner.kill()
            self.error_flag = -1
            raise Exception(f"unknowned error for {domain} {problem}")

        plan = []

        for line in planner.stdout:
            if "explored states" in str(line):
                str_line = str(line)
                index = str_line.index("explored states:") + len("explored states:")
                self.explored_states = int(str_line[index:-3])
            if "===== Plan ======================================" in str(line):
                line = str(planner.stdout.readline())
                while "b'\\n'" not in line:
                    line = str(planner.stdout.readline())
                line = str(planner.stdout.readline())
                while "b'\\n'" not in line:
                    try:
                        start = line.index("\\t") + 3
                        end = line.index("\\t[") - 1
                        line = line[start:end]
                        plan.append(f"({line.lower()})")
                        # print(line)
                    except:
                        pass
                    line = str(planner.stdout.readline())
                break
            elif "No Plan Found!" in str(line):
                # print("Problem unsolvable")
                self.error_flag = 1
                break

        planner.kill()

        return plan
