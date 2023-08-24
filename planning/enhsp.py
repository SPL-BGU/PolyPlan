import config as CONFIG
import subprocess
import os


class ENHSP:
    """
    Create ENHSP for polycraft
    Where ENHSP_PATH must be updated in the config.py file in order to work
    """

    def __init__(self):
        self.path = CONFIG.ENHSP_PATH
        self.error_flag = 0  # -1: error, 0: no error, 1: no solution, 2: timeout

        # Check java version
        java_version = subprocess.check_output(
            ["java", "-version"], stderr=subprocess.STDOUT
        ).decode("utf-8")
        start_index = java_version.index('"') + 1
        end_index = java_version.index('"', start_index)
        java_version = int(java_version[start_index:end_index].split(".")[0])
        if java_version < 15:
            raise Exception("Please use JAVA 15 or higher")

    def create_plan(
        self,
        domain: str,
        problem: str,
        planner: str = "opt-hrmax",
        tolerance: int = 0.01,
        timeout: int = 60,
    ) -> list:
        """
        Create a plan for the given domain and problem
        :param domain: the domain file - must be located in the planning folder
        :param problem: the problem file - must be located in the planning folder
        :param planner: the planner to use - default is opt-hrmax, sat option is sat-hmrphj
        :param tolerance: the tolerance for the planner - default is 0.01
        :param timeout: the timeout for the planner in seconds
        """
        self.error_flag = 0

        # Check if the domain and problem files exist
        if not os.path.exists(domain):
            raise Exception("Domain file not found")
        if not os.path.exists(problem):
            raise Exception("Problem file not found")

        cmd = f"java -jar {self.path}/enhsp.jar -o {domain} -f {problem} -planner {planner} -tolerance {tolerance}"

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
            self.error_flag = 2
            return []

        exception_flag = False
        for line in planner.stderr:
            exception_flag = True
            if "Goal is not reachable" in str(line):
                exception_flag = False
                self.error_flag = 1
                break
        if exception_flag:
            planner.kill()
            self.error_flag = -1
            raise Exception(f"unknowned error for {domain} {problem}")

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
                        plan.append(f"({line.lower()})")
                        # print(line)
                    except:
                        pass
                    line = str(planner.stdout.readline())
                break
            elif "Problem unsolvable" in str(line) or "Unsolvable" in str(line):
                # print("Problem unsolvable")
                self.error_flag = 1
                break

        planner.kill()

        return plan
