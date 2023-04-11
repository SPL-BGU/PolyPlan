import config as CONFIG
import subprocess
import os


class ENHSP:
    """
    Create ENHSP for polycraft
    Where ENHSP_PLANNER_PATH must be updated in the config.py file in order to work
    """

    def __init__(self):
        self.path = CONFIG.ENHSP_PLANNER_PATH

    def create_plan(self, domain, problem) -> list:
        """
        Create a plan for the given domain and problem
        :param domain: the domain file - must be located in the planning folder
        :param problem: the problem file - must be located in the planning folder
        """

        cmd = f"java -jar {self.path}/enhsp.jar -o {domain} -f {problem} -planner opt-hrmax"

        planner = subprocess.Popen(
            "exec " + cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        plan = []
        for line in planner.stdout:
            if "Problem Solved" in str(line):
                line = str(planner.stdout.readline())
                while "Plan-Length" not in line:
                    try:
                        start = line.index("(") + 1
                        end = line.index(")")
                        if line[end - 1] == " ":
                            end -= 1
                        line = line[start:end]
                        plan.append(line)
                        # print(line)
                    except:
                        pass
                    line = str(planner.stdout.readline())
                break

        planner.kill()

        return plan
