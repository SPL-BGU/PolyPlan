from envs import PolycraftGymEnv as BasicMinecraft
from agents import LearningAgent, FixedScriptAgent


def main():
    # start the polycraft environment
    env = BasicMinecraft(visually=True, start_pal=True, keep_alive=False)

    fixed_script_agent = FixedScriptAgent(env, "my_script.txt")

    recording = True

    if recording:
        learning_agent = LearningAgent(env, fixed_script_agent)

        learning_agent.record_trajectory()
        learning_agent.export_trajectory()  # export the trajectory to a file name "expert_trajectory.pkl"
    else:
        env.reset()  # reset the environment

        for _ in range(13):  # 13 is the number of commands in my_script.txt
            fixed_script_agent.act()

    env.close()  # close the environment


if __name__ == "__main__":
    main()
