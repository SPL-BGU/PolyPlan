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

    def create_plan(self) -> list:
        """
        Create a plan for the given domain and problem
        """

        # f"java -jar enhsp.jar -o basic_minecraft_domain.pddl -f basic_minecraft_problem.pddl"
        cmd = f"java -jar {self.path}/enhsp.jar -o {os. getcwd()}/planning/basic_minecraft_domain.pddl -f {os. getcwd()}/planning/basic_minecraft_problem.pddl"

        planner = subprocess.Popen(
            "exec " + cmd,
            shell=True,
            stdout=subprocess.PIPE,
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
