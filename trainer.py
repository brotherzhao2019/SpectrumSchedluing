from collections import namedtuple
import numpy as np
from torch import optim
import torch
import torch.nn as nn
from utils import *
Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask',
                                        'next_state', 'reward', 'misc'))

class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
        self.params = [p for p in self.policy_net.parameters()]

    def get_episode(self, epoch=None):
        episode = []
        state = self.env.reset()            # may need to use parameter (epoch) in reset. todo: need to append an axis in state
                                            # state: np array: nagents * inputdim
        stat = dict()
        info = dict()

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        for t in range(self.args.max_steps):                      # args.max_step 设置大一些，停止由环境触发
            misc = dict()
            # no communication action in this setting
            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])
                x = [state, prev_hid]       # prev_hid :
                action_out, value, prev_hid = self.policy_net(x, info)      # action_out: [tensor(batch, nagents, naction)]
                                                                            # value: [tensor(batch, nagents, 1)]
                                                                            # prev_hid: (tensor(batch * nagents, hid_dim),
                                                                            #               tensor(batch * nagents, hid_dim)
                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                action_out, value = self.policy_net(x, info)
            action = select_action(action_out)                              # action: tensor(batch, nagents)
            action, actual = translate_action(action)                       # action, actual: np.array(nagents) (batch = 1)
            next_state, reward, done, info = self.env.step(actual)          # check format of actual here..

            # attention: no communication action in this setting

            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            #env should handle this make sure that rewrd of dead agents is not counted
            stat['reward'] = stat.get('reward', 0) + reward[: self.args.nagents]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            trans = Transition(state, action, action_out[0], value, episode_mask,
                               episode_mini_mask, next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        return (episode, stat)

    def compute_grad(self, batch):
        stat = dict()
        num_actions = self.args.num_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        rewards = torch.Tensor(batch.reward)                            # tensor(batch , nagents)
        episode_masks = torch.Tensor(batch.episode_mask)                # tensor(batch , nagents)
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)      # tensor(batch , nagents)
        actions = torch.Tensor(batch.action)                            # tensor(batch , nagents)
        #actions = actions.transpose(1, 2).view(-1, n, dim_actions)     # todo: check dimension here, why transpose

        values = torch.cat(batch.value, dim=0)                          # batch * nagents * 1
        #action_out = list(zip(*batch.action_out))
        #action_out = list(batch.action_out)
        action_out = torch.cat(batch.action_out, dim=0)                 # (batch, nagents, nactions)
                                                                        # alive_masks: np.array(batch*nagents)
        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)

        coop_returns = torch.Tensor(batch_size, n)
        ncoop_returns = torch.Tensor(batch_size, n)
        returns = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(batch_size)):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma *\
                               prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = self.args.mean_ratio * coop_returns[i].mean() + (1 - self.args.mean_ratio) * ncoop_returns[i]

        for i in reversed(range(batch_size)):
            advantages[i] = returns[i] - values.data[i]

        # add total reward log:
        tmp_return = (rewards.view(-1) * alive_masks).view(batch_size, n)
        num_cars = alive_masks.view(batch_size, n).sum(-1)
        avg_rewards = (tmp_return / (num_cars.unsqueeze(1) + 0.001)).squeeze().mean().data
        stat['avg_rewards'] = avg_rewards
        if self.args.normalized_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        log_p_a = action_out.view(-1, num_actions)
        actions = actions.contiguous().view(-1)


        log_prob = multinomials_log_density(actions, log_p_a)               #log_prob: tensor: (batch * nagents, 1)
        action_loss = -advantages.view(-1) * log_prob.squeeze()
        action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

        entropy = -(log_p_a * log_p_a.exp()).sum(-1).mean()
        stat['entropy'] = entropy.item()
        if self.args.entr > 0:
            loss -= self.args.entr * entropy

        loss.backward()
        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            #batch.append(episode)
            batch += episode
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        #merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return s

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    def save(self, dir):
        self.policy_net.save(dir)














