# DDPG-Keras-TORCS

This a set of instructions to follow in order to run the Deep Deterministic Policy Gradient
algorithm (DDPG) with Keras to play The Open Racing Car Simulator (TORCS).
The algorithm is explained in this article :

https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html .

You will need Python 2.7​.

Everything will be done in a directory called “ddpg_simulation”. We will assume that you will
put it under your /home/your_username directory. Feel free to set it somewhere else if you
want to.

In order not to mess with your already installed libraries, we will set up a virtual environment
using virtualenv package. First check if it is already installed on your machine by running:
```
virtualenv --version
```
If it’s not already installed, run the following:
```
pip install virtualenv
```
create your Python 2.7 virtual environment:
```
cd ~
```

create the environment
```
virtualenv -p python2 ddpg_simulation
```

enter the directory
```
cd ddpg_simulation
```

run the virtual environment
```
source bin/activate
```

put the requirements.txt file from DDPG-TORCS-simulation folder (the one that contains this
pdf) into ddpg_simulation directory then install them by running:
```
pip2 install -r requirements.txt
```

The requirements file contains libraries that are necessary to run the algorithm and the
simulation like Keras​ to create and manage the neural network and gym​ which is a
collection of environments widely used to run tests on reinforcement learning algorithms.


## Installing gym_torcs
The training is performed on Gym-TORCS, a TORCS environment with Open-AI-gym like
interface. In order to install it, we will follow the steps described here:
https://github.com/ugo-nama-kun/gym_torcs.

Clone the gym-torcs repository as follows:
```
cd to ddpg_simulation
git clone https://github.com/WissamAKRETCHE/DDPG-Keras-TORCS.git
cd gym_torcs
```
Run the following command:

### xautomation
This library enables to programmatically simulate keyboard and mouse use, as well as
manipulate windows. It is used to control the running torcs simulation through the algorithm.

More info at: http://linux.die.net/man/7/xautomation .
```
sudo apt-get install xautomation
```

## Installing TORCS

The Open Racing Car Simulator (TORCS) is a car racing simulator that enables running
tests with pre-programmed AI drivers.
### Plib 1.8.5
First, you will need to install Plib which is a set of libraries for developing games, such as
audio, rendering and control.

To install it, you need to run the following commands:

First, get the requirements by running:
```
sudo apt-get install libglib2.0-dev libgl1-mesa-dev
libglu1-mesa-dev freeglut3-dev libplib-dev libopenal-dev
libalut-dev libxi-dev libxmu-dev libxrender-dev libxrandr-dev
libpng-dev
```
Download plib and untar it in ddpg_simulation/gym_torcs folder:
```
wget http://plib.sourceforge.net/dist/plib-1.8.5.tar.gz
sudo updatedb
plib_folder=$(locate plib-1.8.5.tar.gz)
tar xfvz $plib_folder
```
Install it:
```
cd plib-1.8.5
```
If you run a 64 bit version of Linux, export the following variables (from
http://www.berniw.org/tutorials/robot/torcs/install/plib-install.html) :
```
export CFLAGS="-fPIC"
export CPPFLAGS=$CFLAGS
export CXXFLAGS=$CFLAGS
```
Run the following (for all versions):
```
./configure
sudo make
sudo make install
```

if you run a 64 bit version, reset the flag back:
```
export CFLAGS=
export CPPFLAGS=
export CXXFLAGS=
```
## TORCS

In order to install TORCS, please run the following commands:
```
cd to ddpg_simulation/gym_torcs/vtorcs-RL-color directory
sudo ./configure
sudo make
sudo make install
sudo make datainstall
```

Done !

To check if torcs has been successfully installed, run :
```
sudo torcs
```
Torcs UI should open.

## DDPG

Now you can get the git repository to run DDPG.
```
cd to ddpg_simulation directory
git clone https://github.com/yanpanlau/DDPG-Keras-Torcs.git
cd DDPG-Keras-Torcs
cp *.* ../gym_torcs
cd ../gym_torcs
python2 ddpg.py
```
(Change the flag train_indicator=1 in ddpg.py if you want to train the network)

To stop the virtual environment, run: 
```
deactivate
```

### Note :​
some may have their race view set to “Tracks view” instead of driver’s view. When the
simulation is running, press F2 to check if you are in this situation. If you’re not moving
ahead or the view completely changes, it means that you are. You can do the following to
change torcs configuration:

First, you need to change the settings to be able to control the car in the race:
Launch torcs by running on the terminal: 
```
sudo torcs
```

Select Race > Practice > Configure Race > Accept

Select scr_server1 on the left box, and press (De)Select

Select Player on the right box and press (De)Select

Press Accept, then Accept again

After that, launch a race, fix the view then quit the race:

New Race

Press F2 once the race has started (to set the view to the driver's)

(Press F1 if you want to see the other options)

Press Esc then Abandon Game

This should fix the view issue. Change the player configuration back to scr_server1 (so
ddpg.py will be able to control the car):

In the Practice Menu, select Configure Race > Accept

Select Player on the left box, and press (De)Select

Select scr_server1 on the right box and press (De)Select

Press Accept, then Accept again

Then you can quit torcs: press Esc until you get the Quit? screen, then press Yes, Let’s Quit.
