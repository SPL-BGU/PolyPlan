from pathlib import Path
from tqdm import tqdm

from envs import BasicMinecraft as minecraft
from agents import FixedScriptAgent

from planning.metric_ff import MetricFF
from planning.enhsp import ENHSP

from utils import ProblemGenerator


def generate_solutions(output_directory_path: Path, problem_index: int):
    domain = "/home/benjamin/Projects/PolyPlan/planning/basic_minecraft_domain.pddl"
    problem = f"{output_directory_path}/basic_map_instance_{problem_index}.pddl"

    for solver in [ENHSP, MetricFF]:
        planner = solver()
        plan = planner.create_plan(domain, problem)
        if len(plan) < 30:
            break

    if len(plan) == 0:
        return False

    filename = f"{output_directory_path}/basic_map_instance_{problem_index}.solution"
    with open(filename, "w") as file:
        for item in plan:
            file.write(str(item) + "\n")
    return True


def valid_problem(output_directory_path: Path, problem_index: int):
    found_plan = generate_solutions(output_directory_path, problem_index)
    if not found_plan:
        return False

    env = minecraft(start_pal=False, keep_alive=True)

    domain_path = f"{output_directory_path}/map_instance_{problem_index}.json"
    env.set_domain(domain_path)

    filename = f"{output_directory_path}/basic_map_instance_{problem_index}.solution"
    fixed_script_agent = FixedScriptAgent(env, filename=filename, human_readable=False)

    env.reset()
    reward = 0
    while not env.done:
        fixed_script_agent.act()
        reward += env.reward
    env.close()

    if reward == 0:
        return False
    return True


if __name__ == "__main__":
    output_directory_path = f"/home/benjamin/Projects/PolyPlan/dataset/basic"

    generator = ProblemGenerator(None)
    generator.generate_basic_problems(1000)

    for i in tqdm(range(1000)):
        success = valid_problem(output_directory_path, i)
        if not success:
            raise Exception(f"No solution found for problem {i}")
