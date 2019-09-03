import tensorflow as tf
import numpy as np

import sys
import gym
import ppo
import collections

from tensorboardX import SummaryWriter

writer = SummaryWriter('from_base')

score = 0
episode = 0
p = 0
train_flag = 0

sess = tf.Session()

agent = ppo.PPO(sess, 4, 2)

env = gym.make('CartPole-v0')

state = env.reset()

score_board = collections.deque(maxlen=10)
expert_state_action = []

while True:

    values_list, states_list, actions_list, dones_list, logp_ts_list, rewards_list = \
                [], [], [], [], [], []

    for _ in range(128):

        a, v, logp_t = agent.get_action(state)
        next_state, reward, done, _ = env.step(a)

        if train_flag:
            if len(expert_state_action) < 3000:
                onehot_action = np.zeros([2])
                onehot_action[a] = 1
                expert_data = [state, onehot_action]
                expert_data = [y for x in expert_data for y in x]
                expert_state_action.append(expert_data)
            else:
                np.save('expert_data.npy', np.stack(expert_state_action))
                print('finish')
                sys.exit()
        
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
            score_board.append(score)
            print(episode, score, train_flag)
            if sum(score_board) == 10 * 200:
                train_flag = 1
            writer.add_scalar('score', score, episode)
            score = 0
            state = env.reset()

    if not train_flag:

        _, v, _ = agent.get_action(state)
        values_list.append(v)

        values_list = np.stack(values_list)
        current_value_list = values_list[:-1]
        next_value_list = values_list[1:]

        states_list = np.stack(states_list)
        actions_list = np.stack(actions_list)
        dones_list = np.stack(dones_list)
        logp_ts_list = np.stack(logp_ts_list)
        rewards_list = np.stack(rewards_list)
        current_value_list = np.stack(current_value_list)
        next_value_list = np.stack(next_value_list)

        adv, target = ppo.get_gaes(
            rewards_list, dones_list, current_value_list, next_value_list,
            0.99, 0.95, True)

        value_loss, kl, ent = agent.update(states_list, actions_list, target, adv, logp_ts_list)

        writer.add_scalar('value loss', value_loss, p)
        writer.add_scalar('kl', kl, p)
        writer.add_scalar('ent', ent, p)

        p += 1