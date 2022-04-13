import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
import hydra

from drq import DRQAgent

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)
    
class APTAgent(DRQAgent):
    def __init__(self, reward_free, knn_clip, knn_k, knn_avg, knn_rms, lr, **kwargs):
        super().__init__(lr = lr, **kwargs)
        self.reward_free = reward_free
        self.projection = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
        ).to(self.device)
        
        self.projection.train()
        
        self.projection_optimizer = torch.optim.Adam(self.projection.parameters(), lr=lr)
        #contrastive learning
        self.cross_entropy_loss = nn.CrossEntropyLoss().to(self.device)
        #rms = utils.RMS(self.device)
        rms = utils.RunningMeanStd()
        self.pbe = utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms,
                             self.device)

    def compute_reward(self, next_obs):
        representation = self.critic.encoder(next_obs)
        reward = self.pbe(representation).reshape(-1,1)
        return reward
    
    def update_representation(self, obs, logger, step):
        anchor = self.aug(obs)
        anchor_rep = self.critic.get_representation(anchor)
        anchor_proj = self.projection(torch.relu(anchor_rep))
        
        positive = self.aug(obs)
        positive_rep = self.critic.get_representation(positive)
        positive_proj = self.projection(torch.relu(positive_rep)).detach() #curl        
        
        logits = torch.matmul(anchor_proj, positive_proj.T)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)
        
        self.critic_optimizer.zero_grad()
        self.projection_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        self.projection_optimizer.step()
        logger.log('train/nce_loss', loss, step)
    def update_critic(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done, logger, step):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            dist_aug = self.actor(next_obs_aug)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1,
                                                                  keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_aug,
                                                      next_action_aug)
            target_V = torch.min(
                target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (not_done * self.discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder = True) #detach_encoder = True
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action, detach_encoder = True) #####

        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
            Q2_aug, target_Q)

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update(self, replay_buffer, logger, step):
        obs, action, extr_reward, next_obs, not_done = replay_buffer.sample(
            self.batch_size)
        logger.log('train/batch_reward', extr_reward.mean(), step)
        
        obs, obs_aug = self.augmentation(obs)
        next_obs, next_obs_aug = self.augmentation(next_obs)
        if self.reward_free : 
            self.update_representation(obs, logger, step)
            intr_reward = self.compute_reward(next_obs).detach()
            logger.log('train/intr_reward', intr_reward.mean(), step)
            reward = intr_reward
        else :
            reward = extr_reward
        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

