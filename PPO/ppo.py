import tensorflow as tf
import numpy as np
import gym
import sys
from ActorNetwork import *
from CriticNetwork import *

class PPO(object):

    def __init__(self, sess, S_DIM, A_DIM, A_LR, C_LR, A_UPDATE_STEPS, C_UPDATE_STEPS, METHOD):
        self.sess = sess
        self.S_DIM = S_DIM
        self.A_DIM = A_DIM
        self.A_LR = A_LR
        self.C_LR = C_LR
        self.A_UPDATE_STEPS = A_UPDATE_STEPS
        self.C_UPDATE_STEPS = C_UPDATE_STEPS
        self.METHOD = METHOD

        # tf placeholders
        self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.tfa = tf.placeholder(tf.float32, [None, self.A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.tflam = tf.placeholder(tf.float32, None, 'lambda')


        # critic
        with tf.variable_scope('critic'):

            self.v = critic(self.tfs)          
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)

        # actor
        self.pi, self.pi_params = actor(self.tfs, 'pi', trainable=True)
        self.oldpi, self.oldpi_params = actor(self.tfs, 'oldpi', trainable=False)

        self.pi_mean = self.pi.mean()
        self.pi_sigma = self.pi.stddev()

 
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(self.pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_params, self.oldpi_params)]
        
        with tf.variable_scope('loss'):
            self.ratio = tf.exp(self.pi.log_prob(self.tfa) - self.oldpi.log_prob(self.tfa))

            if self.METHOD['name'] == 'kl_pen':
                kl = tf.distributions.kl_divergence(self.oldpi, self.pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))

            else:   # clipping method
                self.clipped_ratio = tf.clip_by_value(self.ratio, 1.-self.METHOD['epsilon'], 1.+self.METHOD['epsilon'])
                self.aloss = -tf.reduce_mean(tf.minimum(self.ratio*self.tfadv, self.clipped_ratio*self.tfadv))

                # entropy loss
                entropy = -tf.reduce_sum(self.pi.prob(self.tfa) * tf.log(tf.clip_by_value(self.pi.prob(self.tfa),1e-10,1.0)),axis=1)
                entropy = tf.reduce_mean(entropy,axis=0)    
                self.aloss -= 0.001 * entropy


        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)

   
    def screen_out(self, s, a, r):

        print("ratio: ", self.sess.run(self.ratio, {self.tfs: s, self.tfa: a}))
        print("clipped_ratio: ", self.sess.run(self.clipped_ratio, {self.tfs: s, self.tfa: a}))

        print("mu: ", self.sess.run(self.pi_mean, {self.tfs: s, self.tfa: a}))
        print("sigma: ", self.sess.run(self.pi_sigma, {self.tfs: s, self.tfa: a}))
        
        print("sample action: ", self.sess.run(self.sample_op, {self.tfs: s}))

     

    def update(self, s, a, r):
      
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})

        # update actor
        if self.METHOD['name'] == 'kl_pen':
            for _ in range(self.A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: self.METHOD['lam']})

            if kl < self.METHOD['kl_target'] / 1.5:  # adaptive lambda
                self.METHOD['lam'] /= 2

            elif kl > self.METHOD['kl_target'] * 1.5:
                self.METHOD['lam'] *= 2
            self.METHOD['lam'] = np.clip(self.METHOD['lam'], 1e-4, 10)    # sometimes explode

        else:   # clipping method
            for _ in range(self.A_UPDATE_STEPS):
               self.sess.run(self.atrain_op, feed_dict={self.tfs: s, self.tfa: a, self.tfadv: adv})
                 
        # update critic
        for _ in range(self.C_UPDATE_STEPS):
           self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) 
     

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})
        return a[0]

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        vv = self.sess.run(self.v, {self.tfs: s})
        return vv[0,0]
