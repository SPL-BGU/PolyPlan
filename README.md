<h1 align="center">Solving Minecraft Tasks via Model Learning</h2>
<p align="center">
<a href="https://github.com/Search-BGU/PolyPlan/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<a href="https://www.python.org/downloads/release/python-3818/"><img alt="Python Version" src="https://img.shields.io/badge/python-3.8-blue"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# Getting Started

The code has been thoroughly tested on the Ubuntu operating system and is recommended for use with this project. <br />
While the project has not been formally tested, it should function correctly when using WSL on Windows. <br />

## Dependencies
1. Make sure that Python 3.8 is installed and active (via [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) or conda environment).
2. Make sure java8 is installed and active.
3. Follow the installation of [Polycraft (PAL)](https://github.com/StephenGss/PAL/tree/ba9128bd8bd37d79a60c6e94b6670870911c0986).
4. Navigate to the pal directory and run the following command:
```
xvfb-run -s '-screen 0 1280x1024x24' ./gradlew --no-daemon --stacktrace runclient
```
* This will run Polycraft independently in headless mode. Gradle will install any dependencies that the java runtime needs, and eventually a message will appear in the log output Minecraft finished loading, which signifies that Polycraft is ready to use. Exit out of the application.
5. Follow the installation of [Numeric-SAM (N-SAM)](https://github.com/argaman-aloni/sam_learning/tree/55-add-the-ability-to-iterate-over-action-triplets-instead-of-complete-trajectories).
6. pip install all the requirements for this project:
```
python -m pip install -r requirements.txt
```

### Possible issues
1. While installing pal, running the command ./setup_linux_headless.sh may cause an error that can be fixed by running the following command and then try again:
```
sed -i -e 's/\r$//' setup_linux_headless.sh
```
2. If you got an error while installing gym:
```
pip install setuptools==66
```
3. If you still have errors while installing the gym:
```
pip install wheel==0.38.4
```

# Usage

## How to launch your first agent:
1. You need to update all the locations in the config.py file
2. Now you can run the demo agent with the following command: 
```
python demo.py
```
* The demo agent do random actions in order to solve the environment.

## How to use the environment:
1. To run only the Polycraft server you can use the following code: 
```
env = BasicMinecraft(visually=True, keep_alive=True)
env.reset()
env.close()
```
2. To run as a custom RL agent use the following code:
```
env = BasicMinecraft(visually=True)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
env.close()
```
* As used in playground.py
3. If you like to export the expert trajectories for planning or behavioural cloning:
```
python custom_agent.py
```
* The agent makes a wooden pogo from a list of commands (my_script.txt).
4. If you like to train the agent to learn the last k actions of an environment use:
```
env = PolycraftGymEnvKLA(BasicMinecraft, k=1, expert_actions=11, visually=True)
```
* You can update "my_script.txt" as you like, and set expert_actions to num of lines of the file
5. In order to start an planning agent:
```
enhsp = ENHSP()
plan = enhsp.create_plan()
model = FixedScriptAgent(env, script=plan)
```
* You need to update the ENHSP location in the config.py file and change the java version accordingly
6. To see the model learning results run the following command in a shell:
```
tensorboard --logdir logs
```
7. The code of our Hybrid approch can be seen in:
```
agents/exploring_sam.py
agents/hybrid_ppo_model.py
```

## How to reproduce results in the paper
1. Recreate the maps:
```
python constructor.py
```
2. Run the offline RL algorithms:
```
python playground_offline.py
```
3. Run the offline N-SAM algorithm:
```
python playground_nsam.py
```
4. Run the online algorithms:
```
python playground_online.py
```
5. Run the Hybrid algorithm:
```
python playground_online_hybrid.py
```
* Note that the reproduction of the Wooden Sword task is in the equalvine file named with "sword" at the end

# Citations

If you find our work interesting or the repo useful, please consider citing [this paper](https://arxiv.org/abs/2502.13006):
```
@misc{benyamin2025integratingreinforcementlearningaction,
      title={Integrating Reinforcement Learning, Action Model Learning, and Numeric Planning for Tackling Complex Tasks}, 
      author={Yarin Benyamin and Argaman Mordoch and Shahaf S. Shperberg and Roni Stern},
      year={2025},
      eprint={2502.13006},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.13006}, 
}
```
