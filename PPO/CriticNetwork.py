import tensorflow as tf
import numpy as np
import gym
import sys


HIDDEN1 = 400
HIDDEN2 = 300
HIDDEN3 = 300


xavier = tf.contrib.layers.xavier_initializer()
bias_const = tf.constant_initializer(0.05)
rand_unif = tf.keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3)
regularizer = tf.contrib.layers.l2_regularizer(scale=5e-4)

def critic(tfs):
	with tf.variable_scope('critic'):
            l1 = tf.layers.dense(tfs, HIDDEN1, activation=None, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
            l1 = tf.nn.relu(l1)

            l2 = tf.layers.dense(l1, HIDDEN2, activation=None, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
            l2 = tf.nn.relu(l2)

            l3 = tf.layers.dense(l2, HIDDEN3, activation=None, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer) 
            l3 = tf.nn.relu(l3)

            v = tf.layers.dense(l3, 1, activation=None, kernel_initializer=rand_unif, bias_initializer=bias_const)
            return v
