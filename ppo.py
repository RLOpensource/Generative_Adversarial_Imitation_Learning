import tensorflow as tf
import numpy as np

import gym
import copy

from tensorboardX import SummaryWriter

def get_gaes(rewards, dones, values, next_values, gamma, lamda, normalize):
    deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
    deltas = np.stack(deltas)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

    target = gaes + values
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    return gaes, target

def mlp(x, hidden_sizes=[32], activation=tf.nn.relu, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi, logp_all

def mlp_actor_critic(x, a, hidden_sizes=(256, 256), activation=tf.nn.relu, output_activation=None, action_space=None):

    with tf.variable_scope('pi'):
        pi, logp, logp_pi, logp_all = mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, v, logp_all

class PPO:
    def __init__(self, sess, state_size, action_dim):
        self.sess = sess

        self.ppo_eps = 0.2
        self.pi_lr = 0.001
        self.v_lr = 0.001
        self.epoch = 3
        self.state_size = state_size
        self.action_dim = action_dim
        
        self.s_ph = tf.placeholder(tf.float32, [None, self.state_size])
        self.a_ph = tf.placeholder(tf.int32, shape=[None])
        self.logp_old_ph = tf.placeholder(tf.float32, shape=[None])
        self.adv_ph = tf.placeholder(tf.float32, shape=[None])
        self.target_ph = tf.placeholder(tf.float32, shape=[None])

        self.all_phs = [self.s_ph, self.a_ph, self.target_ph, self.adv_ph, self.logp_old_ph]

        self.pi, self.logp, self.logp_pi, self.v, self.logp_all = mlp_actor_critic(
            x=self.s_ph, a=self.a_ph, activation=tf.nn.relu, output_activation=None, action_space=self.action_dim)

        self.ratio = tf.exp(self.logp - self.logp_old_ph)
        self.clipped_ratio = tf.clip_by_value(self.ratio, clip_value_min=1 - self.ppo_eps, clip_value_max=1 + self.ppo_eps)
        self.min = tf.minimum(tf.multiply(self.adv_ph, self.clipped_ratio), tf.multiply(self.adv_ph, self.ratio))
        self.entropy = -tf.reduce_mean(-tf.exp(self.logp_all) * self.logp_all) * 0.01
        self.pi_loss = -tf.reduce_mean(self.min) + self.entropy
        self.v_loss = tf.reduce_mean((self.target_ph - self.v) ** 2)

        self.train_pi = tf.train.AdamOptimizer(self.pi_lr).minimize(self.pi_loss)
        self.train_v = tf.train.AdamOptimizer(self.v_lr).minimize(self.v_loss)

        self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)
        self.approx_ent = tf.reduce_mean(-self.logp)

        self.sess.run(tf.global_variables_initializer())

    def update(self, state, action, target, adv, logp_old):
        zip_ph = [state, action, target, adv, logp_old]
        inputs = {k:v for k,v in zip(self.all_phs, zip_ph)}
        value_loss, kl, ent = 0, 0, 0
        for i in range(self.epoch):
            _, _, v_loss, approxkl, approxent = self.sess.run([self.train_pi, self.train_v, self.v_loss, self.approx_kl, self.approx_ent], feed_dict=inputs)
            value_loss += v_loss
            kl += approxkl
            ent += approxent
        return value_loss, kl, ent

    def get_action(self, state):
        a, v, logp_t = self.sess.run([self.pi, self.v, self.logp_pi], feed_dict={self.s_ph: [state]})
        return a[0], v[0], logp_t[0]