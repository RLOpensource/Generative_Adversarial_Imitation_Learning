import tensorflow as tf
import numpy as np

def model(x, hidden_dims, activation, output_activation):
    for h in hidden_dims:
        x = tf.layers.dense(inputs=x, units=h, activation=activation)
    return tf.layers.dense(inputs=x, units=1, activation=output_activation)

class Discriminator:
    def __init__(self, sess, state_size, action_size):
        self.sess = sess
        self.learning_rate = 0.001
        self.hidden_dims = [128, 128]

        self.expert_state = tf.placeholder(tf.float32, shape=[None, state_size])
        self.expert_action = tf.placeholder(tf.float32, shape=[None, action_size])
        self.noise_expert_action = self.expert_action# + tf.random_normal(tf.shape(self.expert_action), mean=0.2, stddev=0.1) / 1.2
        self.expert_concat = tf.concat([self.expert_state, self.noise_expert_action], axis=1)

        self.agent_state = tf.placeholder(tf.float32, shape=[None, state_size])
        self.agent_action = tf.placeholder(tf.float32, shape=[None, action_size])
        self.noise_agent_action = self.agent_action# + tf.random_normal(tf.shape(self.agent_action), mean=0.2, stddev=0.1) / 1.2
        self.agent_concat = tf.concat([self.agent_state, self.noise_agent_action], axis=1)

        with tf.variable_scope('model'):
            self.prob_1 = model(self.expert_concat, self.hidden_dims, tf.nn.relu, tf.sigmoid)
        with tf.variable_scope('model', reuse=True):
            self.prob_2 = model(self.agent_concat, self.hidden_dims, tf.nn.relu, tf.sigmoid)

        loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(1 - self.prob_1, 1e-8, 1.0)))
        loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(self.prob_2, 1e-8, 1.0)))
        self.loss = - loss_expert - loss_agent

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        self.reward = - tf.squeeze(tf.log(tf.clip_by_value(self.prob_2, 1e-10, 1.0)), axis=1)

        self.sess.run(tf.global_variables_initializer())

    def get_rewards(self, agent_state, agent_action):
        return self.sess.run(self.reward, feed_dict={self.agent_state: agent_state, self.agent_action: agent_action})

    def update(self, expert_state, expert_action, agent_state, agent_action):
        return self.sess.run([self.loss, self.train_op], feed_dict={self.agent_state: agent_state, self.agent_action: agent_action, self.expert_state: expert_state, self.expert_action: expert_action})