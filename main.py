import tensorflow as tf
import numpy as np

import sys
import gym
import ppo
import utils
import random
import collections
import discriminator

from tensorboardX import SummaryWriter

data = np.load('expert_data.npy')
expert_data = collections.deque()
for x in data:
    expert_data.append(x)

sess = tf.Session()

state_size = 4
action_size = 2
n_step = 128
agent = ppo.PPO(sess, state_size, action_size)
dis = discriminator.Discriminator(sess, state_size, action_size)

env = gym.make('CartPole-v0')
score = 0
episode = 0
p = 0
gail = True

writer = SummaryWriter()

state = env.reset()

while True:
    values_list, states_list, actions_list, dones_list, logp_ts_list, rewards_list = \
                [], [], [], [], [], []
    
    for _ in range(n_step):

        a, v, logp_t = agent.get_action(state)
        next_state, reward, done, _ = env.step(a)

        score += reward

        r = 0.
        if done:
            if score == 200:
                r = 1.
            else:
                r = -1.
        
        states_list.append(state)
        actions_list.append(a)
        dones_list.append(done)
        values_list.append(v)
        logp_ts_list.append(logp_t)
        rewards_list.append(r)

        state = next_state

        if done:
            episode += 1
            print(episode, score)
            writer.add_scalar('score', score, episode)
            state = env.reset()
            score = 0
            if episode == 250:
                sys.exit()

    _, v, _ = agent.get_action(state)
    values_list.append(v)

    values_list = np.stack(values_list)
    current_value_list = values_list[:-1]
    next_value_list = values_list[1:]

    states_list = np.stack(states_list)
    actions_list = np.stack(actions_list)
    dones_list = np.stack(dones_list)
    logp_ts_list = np.stack(logp_ts_list)
    current_value_list = np.stack(current_value_list)
    next_value_list = np.stack(next_value_list)
    rewards_list = np.stack(rewards_list)

    onehot_action_list = utils.list_to_onehot(actions_list, action_size)

    expert_sample = np.stack(random.sample(expert_data, n_step))

    agent_state = states_list
    agent_action = onehot_action_list
    expert_state = expert_sample[:, :state_size]
    expert_action = expert_sample[:, state_size:]

    dis.update(expert_state, expert_action, agent_state, agent_action)

    if gail:
        rewards_list += dis.get_rewards(agent_state, agent_action) * 0.1
    
    adv, target = ppo.get_gaes(
        rewards_list, dones_list, current_value_list, next_value_list,
        0.99, 0.95, True)

    value_loss, kl, ent = agent.update(states_list, actions_list, target, adv, logp_ts_list)

    writer.add_scalar('value loss', value_loss, p)
    writer.add_scalar('kl', kl, p)
    writer.add_scalar('ent', ent, p)

    p += 1