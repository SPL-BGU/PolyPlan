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

        # Check java version
        java_version = subprocess.check_output(
            ["java", "-version"], stderr=subprocess.STDOUT
        ).decode("utf-8")
        start_index = java_version.index('"') + 1
        end_index = java_version.index('"', start_index)
        java_version = int(java_version[start_index:end_index].split(".")[0])
        if java_version < 15:
            raise Exception("Please use JAVA 15 or higher")

    def create_plan(self, domain, problem) -> list:
        """
        Create a plan for the given domain and problem
        :param domain: the domain file - must be located in the planning folder
        :param problem: the problem file - must be located in the planning folder
        """

        # Check if the domain and problem files exist
        if not os.path.exists(domain):
            raise Exception("Domain file not found")
        if not os.path.exists(problem):
            raise Exception("Problem file not found")

        cmd = f"java -jar {self.path}/enhsp.jar -o {domain} -f {problem} -planner opt-hrmax"

        planner = subprocess.Popen(
            "exec " + cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        for line in planner.stderr:
            planner.kill()
            raise Exception("unknowned error")

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
            elif "Problem unsolvable" in str(line):
                # print("Problem unsolvable")
                break

        planner.kill()

        return plan
