from polycraft_gym_env import PolycraftGymEnv
from agents.learning_agent import LearningAgent
from agents.fixed_script_agent import FixedScriptAgent


def main():
    # start the polycraft environment and keep it open
    env = PolycraftGymEnv(
        visually=True, start_pal=True, keep_alive=True
    )  # after the first run for running again you need to change start_pal to False
    env.reset()  # reset the environment
    env.close()

    fixed_script_agent = FixedScriptAgent("my_script.txt")
    learning_agent = LearningAgent(fixed_script_agent)

    learning_agent.open_connection()  # open the connection to the Polycraft server

    for _ in range(32):  # 32 is the number of commands in my_script.txt
        action = learning_agent.act()
        print(action)

    learning_agent.export_trajectory()  # export the trajectory to a file name "expert_trajectory.json"
    learning_agent.close_connection()  # close the connection to the simulator


if __name__ == "__main__":
    main()
