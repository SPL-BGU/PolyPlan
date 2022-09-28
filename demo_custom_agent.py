from Agents.LearningAgent import LearningAgent
from Agents.FixedScriptAgent import FixedScriptAgent


def main():
    fixedScriptAgent = FixedScriptAgent("my_script.txt")
    learningAgent = LearningAgent(fixedScriptAgent)

    learningAgent.open_connection()  # open the connection to the Polycraft server

    for _ in range(28):  # 28 is the number of commands in my_script.txt
        action = learningAgent.act()
        print(action)

    learningAgent.export_trajectory()  # export the trajectory to a file name "expert_trajectory.json"
    learningAgent.close_connection()  # close the connection to the simulator


if __name__ == "__main__":
    main()
