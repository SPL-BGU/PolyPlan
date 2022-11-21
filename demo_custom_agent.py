from polycraft_gym_env import PolycraftGymEnv
from agents.learning_agent import LearningAgent
from agents.fixed_script_agent import FixedScriptAgent


def main():
    # start the polycraft environment
    env = PolycraftGymEnv(visually=True, start_pal=True, keep_alive=False)

    fixed_script_agent = FixedScriptAgent(env, "my_script.txt")

    recording = True

    if recording:
        learning_agent = LearningAgent(env, fixed_script_agent)

        learning_agent.record_trajectory(episodes=2)
        learning_agent.export_trajectory()  # export the trajectory to a file name "expert_trajectory.pkl"
    else:
        env.reset()  # reset the environment

        for _ in range(30):  # 30 is the number of commands in my_script.txt
            action = fixed_script_agent.act()
            print(action)

    env.close()  # close the environment


if __name__ == "__main__":
    main()
