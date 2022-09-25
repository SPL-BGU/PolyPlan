# Getting Started

I recommending to use ubuntu as an operating system or using wsl on windows. <br />

## Prerequisites
1. Make sure that Python 3.8 is installed and active (via virtual environment or by using python3.8 instead of python/python3 command).
2. Make sure java8 is installed and active.
3. Follow the installation of pal - [this link](https://github.com/StephenGss/PAL/tree/release_2.0#Installation).

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


# Usage

## How to run the demo agent:
1. With the shell go to the pal directory and run the following command: 
> ./gradlew runclient
* This should take a moment and start a Minecraft client self-host on 127.0.0.1:9000.
2. Go to the PolyPlan directory and run the following command (update the path accordingly): 
> python3.8 RunEnvironment.py -domain /locaion/to/your/pal/available_tests/pogo_nonov.json
* This will start your demo environment on the Minecraft client.
3. Then just run the following command: 
> python3.8 demo_agent.py 
* Now you can watch the demo agent make a wooden pogo.

