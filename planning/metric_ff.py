import config as CONFIG
import subprocess

import os


class MetricFF:
    """
    Create MetricFF for polycraft
    Where METRIC_FF_PATH must be updated in the config.py file in order to work
    """

    def __init__(self):
        self.path = CONFIG.METRIC_FF_PATH

    def create_plan(self, domain: str, problem: str, timeout: int = 60) -> list:
        """
        Create a plan for the given domain and problem
        :param domain: the domain file - must be located in the planning folder
        :param problem: the problem file - must be located in the planning folder
        :param timeout: the timeout for the planner in seconds
        """

        # Check if the domain and problem files exist
        if not os.path.exists(domain):
            raise Exception("Domain file not found")
        if not os.path.exists(problem):
            raise Exception("Problem file not found")

        original_dir = os.getcwd()
        os.chdir(self.path)
        cmd = f"./ff -o {domain} -f {problem} -s 0"

        planner = subprocess.Popen(
            "exec " + cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            planner.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"Can't find a plan in {timeout} seconds")
            planner.kill()
            return []

        os.chdir(original_dir)

        exception_flag = False
        for line in planner.stderr:
            exception_flag = True
            if "not reachable" in str(line):
                exception_flag = False
                break
        if exception_flag:
            planner.kill()
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
            elif "unsolvable" in str(line):
                # print("Problem unsolvable")
                break

        planner.kill()

        return plan
