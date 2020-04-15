# mit_s191_intro_to_deep_learning
A repository to hold same basic guidances on how to set up my working environment
# Needed packages in the system
sudo apt-get install -y abcmidi  # Converts abc files into MIDI
sudo apt-get install -y timidity # Sound renderer to play MIDI files
# How to install a specific python version (Example with python3.8)
1. Update system: sudo apt-get update
2. Install desired python version: sudo apt-get install python3.8
3. Verify that your installation was succesful: which python3.8
    You should read something like :/usr/bin/python3.8
4. Download and install pip:
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3.8 get-pip.py
5. Install virtual environment: python3.8 -m pip install --user virtualenv

# Creating the Environment
6. Go to the Folder project: cd /media/bruno/Extra/Ubuntu-Repository/Personal\ Repo/getting-started-python/
7. Create environment: python3.8 -m venv env

# Using the Environment
8. Activate the environment: source env/bin/activate
9. Do stuff(run programs, install libraries,...)
10. Deactivate: deactivate

---------------------------------------------------------------------------------------------------
# Virtual environment for Anaconda/Conda users
<!-- It Seems that Anaconda has some limitations and only let you use python 3.7 or 2.7 in the UI-->
1.  Install Anaconda:
2.  Create environment in the UI or through:
        conda create -n myenvpython38 python=3.8 
        <!-- The new environment interpreter should be found in /home/bruno/anaconda3/envs/myenvpython38/bin/python3.8 -->
        <!-- Remember when using Spyder to change the interpreter to the one you are using -->
        <!-- If you want to install the spyder files as you create the environment just add a bit at the end -->
        conda create -n myenvpython38 python=3.8 spyder-kernels 
3.  Install libraries and stuff you need.

4.  Export environment into a YML file to save it into repository:
        conda env export --name myenvpython38 > myenvpython38.yml
    This produces the file "/media/bruno/Extra/Ubuntu-Repository/Personal Repo/getting-started-python/myenvpython38.yml"


5.  If in another computer import the enviroment from the YML file "Create an environment from YAML file":
        conda env create --file myenvpython38.yml

6.  If in same computer wanna clone your environment to branch away:
        conda create --clone myenvpython38 --name mynewpy38
7. Delete environment if something goes wrong:
    conda remove --name ENVNAME --all

---------------------------------------------------------------------------------------------------
# New environment msyself Python 3.8 for a new project
conda create --clone myenvpython38 --name project_name
