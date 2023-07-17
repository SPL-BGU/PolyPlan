from envs import BasicMinecraft, IntermediateMinecraft, AdvancedMinecraft
from agents import LearningAgent, FixedScriptAgent


def main():
    env_index = 0  # 0: BasicMinecraft, 1: IntermediateMinecraft, 2: AdvancedMinecraft
    minecraft = [BasicMinecraft, IntermediateMinecraft, AdvancedMinecraft][env_index]

    # start the polycraft environment
    env = minecraft(visually=True, start_pal=True, keep_alive=False)

    if env_index == 0:
        filename = "agents/scripts/macro_actions_script.txt"
    elif env_index == 1:
        filename = "agents/scripts/intermediate_actions_script.txt"
    else:  # env_index == 2
        filename = "agents/scripts/advanced_actions_script.txt"

    fixed_script_agent = FixedScriptAgent(env, filename=filename, human_readable=True)

    recording = True
    planning = False

    if recording:
        learning_agent = LearningAgent(env, fixed_script_agent, for_planning=planning)

        learning_agent.record_trajectory()
        learning_agent.export_trajectory()  # export the trajectory to a file name "expert_trajectory.pkl"
    else:
        env.reset()  # reset the environment

        while not env.done:
            fixed_script_agent.act()

    env.close()  # close the environment


if __name__ == "__main__":
    main()
