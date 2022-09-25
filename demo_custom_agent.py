from Agents.LearningAgent import LearningAgent
from Agents.FixedScriptAgent import FixedScriptAgent


def main():
    fixedScriptAgent = FixedScriptAgent("MyScript.txt")
    learningAgent = LearningAgent(fixedScriptAgent)

    for _ in range(28):  # 28 is the number of commands in MyScript.txt
        action = learningAgent.act()
        print(action)

    learningAgent._export_trajectory()  # export the trajectory to a file name "learning_agent.json"


if __name__ == "__main__":
    main()
