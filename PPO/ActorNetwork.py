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

def actor( tfs, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(tfs, HIDDEN1, activation=None, trainable=trainable, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
            l1 = tf.nn.relu(l1)

            l2 = tf.layers.dense(l1, HIDDEN2, activation=None, trainable=trainable, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
            l2 = tf.nn.relu(l2)

            l3 = tf.layers.dense(l2, HIDDEN3, activation=None, trainable=trainable, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
            l3 = tf.nn.relu(l3)

            mu_st = tf.layers.dense(l3, 1, activation=tf.nn.tanh, trainable=trainable, kernel_initializer=rand_unif, bias_initializer=bias_const)
            mu_acc = tf.layers.dense(l3, 1, activation=tf.nn.sigmoid, trainable=trainable, kernel_initializer=rand_unif, bias_initializer=bias_const) 
            mu_br = tf.layers.dense(l3, 1, activation=tf.nn.sigmoid, trainable=trainable, kernel_initializer=rand_unif, bias_initializer=bias_const)

            small = tf.constant(1e-6)
            mu_st = tf.clip_by_value(mu_st,-1.0+small,1.0-small)
            mu_acc = tf.clip_by_value(mu_acc,0.0+small,1.0-small) 
            mu_br = tf.clip_by_value(mu_br,0.0+small,1.0-small)
            mu_br = tf.scalar_mul(0.1,mu_br) # scalar mult
            mu = tf.concat([mu_st, mu_acc, mu_br], axis=1)          

            sigma_st = tf.layers.dense(l3, 1, activation=tf.nn.sigmoid, trainable=trainable, kernel_initializer=rand_unif, bias_initializer=bias_const)
            sigma_acc = tf.layers.dense(l3, 1, activation=tf.nn.sigmoid, trainable=trainable, kernel_initializer=rand_unif, bias_initializer=bias_const)
            sigma_br = tf.layers.dense(l3, 1, activation=tf.nn.sigmoid, trainable=trainable, kernel_initializer=rand_unif, bias_initializer=bias_const)
            sigma_st = tf.scalar_mul(0.2,sigma_st) # scalar mult            
            sigma_acc = tf.scalar_mul(0.2,sigma_acc) # scalar mult 
            sigma_br = tf.scalar_mul(0.05,sigma_br) # scalar mult 
            sigma = tf.concat([sigma_st, sigma_acc, sigma_br], axis=1)          
            sigma = tf.clip_by_value(sigma,0.0+small,1.0-small)

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params