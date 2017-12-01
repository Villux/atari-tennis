# atari-tennis
RL Agent to play Atari 2600 Tennis

# Installlation instructions 

### Install system packages
- On OSX: 
	- brew install cmake boost boost-python sdl2 swig wget
- Ubuntu:
	- sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

### Create and activate the virtualenv:
- virtualenv -p /usr/bin/python2.(?) "your chosen directory name"
- source "your chosen directory name"/bin/activate

### Install the python packages:
1. Longer way	
	- pip install gym
	- pip install gym[atari]
	- pip install ipdb
    - Install the pytorch for python2.7 and for CUDA 7.5:
        - pip install http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl 
        - pip install torchvision 
        #### if the above command does not work, then you have python 2.7 UCS2, use this command 
        - pip install http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp27-cp27m-manylinux1_x86_64.whl
        
2. Easier way 
	- pip install -r requirement.txt

### Run the code:
- python atari_tennis.py
