[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Getting Started

I recommending to use ubuntu as an operating system or using wsl on windows. <br />

## Prerequisites
1. Make sure that Python 3.8 is installed and active (via [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment), conda environment or by using python3.8 instead of python commands).
2. Make sure java8 is installed and active.
3. Follow the installation of [pal](https://github.com/StephenGss/PAL/tree/release_2.0#Installation).
* Note that the last checked working pal version with this project is [release 2.0 commit number 25d9af6](https://github.com/StephenGss/PAL/tree/25d9af6ed6d58693f96eae7927bb0852008afcfe).

### Possible issues
While installing pal, running the command ./setup_linux_headless.sh may cause an error that can be fixed by running the following command:
> sed -i -e 's/\r$//' setup_linux_headless.sh

and then try again.

## Installation
1. Run the following commands in the shell:
> sudo apt-get update <br />
> sudo apt-get upgrade -y <br />
> sudo apt-get -y install xvfb mesa-utils x11-xserver-utils xdotool gosu <br />
> sudo apt-get install zip unzip build-essential -y <br />
> sudo apt-get install unixodbc-dev -y <br />
> sudo apt-get install python3-dev -y <br />
> sudo apt-get install python3-pip -y <br />

2. Navigate to the pal directory and run the following command:
> xvfb-run -s '-screen 0 1280x1024x24' ./gradlew --no-daemon --stacktrace runclient
* This will run Polycraft independently in headless mode. Gradle will install any dependencies that the java runtime needs, and eventually a message will appear in the log output Minecraft finished loading, which signifies that Polycraft is ready to use. Exit out of the application.

3. pip install all the requirements for this project:
> python -m pip install -r requirements.txt


# Usage

## How to launch your agent:
1. You need to update the pal location in the config.py file
2. Now you can run the demo agent with the following command: 
> python demo_custom_agent.py 
* The demo agent makes a wooden pogo from a list of commands (name my_script.txt). Also, you can record the expert trajectory and export it (to a file name expert_trajectory.pkl).

## How to use the environment:
1. To run only the Polycraft server you can use the following code: 
> env = PolycraftGymEnv(visually=True, keep_alive=True) <br />
> env.reset() <br />
> env.close() <br />
* As used in demo_custom_agent.py
2. To run as a custom ai gym environment use the following code:
> env = PolycraftGymEnv(visually=True) <br />
> model = PPO("MlpPolicy", env, verbose=1) <br />
> model.learn(total_timesteps=1000) <br />
> env.close() <br />
* As used in playground.py
3. To see the model learning results run the following command in the shell:
> tensorboard --logdir logs
* Need to run the model first from playground.py
