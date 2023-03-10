from envs import BasicMinecraft, AdvancedMinecraft
from agents import LearningAgent, FixedScriptAgent


def main():

    advanced_actions = False  # set to True to use advanced actions
    minecraft = [BasicMinecraft, AdvancedMinecraft][int(advanced_actions)]

    # start the polycraft environment
    env = minecraft(visually=True, start_pal=True, keep_alive=False)

    if advanced_actions:
        fixed_script_agent = FixedScriptAgent(
            env, filename="agents/scripts/advanced_actions_script.txt"
        )
    else:
        fixed_script_agent = FixedScriptAgent(
            env, filename="agents/scripts/macro_actions_script.txt"
        )

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
