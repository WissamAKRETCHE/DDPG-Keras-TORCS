# DDPG-Keras-TORCS

This Repository is under development.

In order to run the Deep Deterministic Policy Gradient algorithm (DDPG) with Keras to play The Open Racing Car Simulator (TORCS), please click [here !](https://github.com/WissamAKRETCHE/DDPG-Keras-TORCS/blob/master/installation.md)

The algorithm is explained in this article :

https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html .

Code base adapted from: https://github.com/yanpanlau/DDPG-Keras-Torcs.git

Our model has been trained on different tracks. The figures below show the layouts of some tracks:

#### The Aalborg track:
![Aalborg track](/images/Aalborg.png)

#### The E-track2:
![E-track2](/images/Etrack2_layout.png)

#### The CG track3:
![CG track3](/images/cgtrack3.png)

The first video shows the result of the E-track2, our validation dataset:
![E-track2 results](/images/E-track.gif)

The second video shows the result of the CG track3, our test dataset:
![CG-track3 results](/images/CG-track.gif)


The algorithm seems to perform well ! However, it gets trapped in a local optimum sometimes (we can see the reward curve on the right):

![Local optimum](local_optimum.gif)
