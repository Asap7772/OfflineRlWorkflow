from collections import OrderedDict
import torch
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
import torch.optim as optim
import numpy as np
import wandb
import os

class TD3BCTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,
            beta = 1.0, 

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            log_dir=None,
            wand_b=True,
            variant_dict=None,
            real_data=False,
            log_pickle=True,
            pickle_log_rate=5,
            *args,
            **kwargs
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.beta = beta

        self.wand_b = wand_b
        self.log_dir = log_dir
        self.real_data = real_data
        self.log_pickle = True
        self.pickle_log_rate = 5
        self._log_epoch = 0
        if self.wand_b:
            if self.real_data:
                wandb.init(project='real_drawer_cql', reinit=True)
            else:
                wandb.init(project='cog_td3bc', reinit=True)
            wandb.run.name=log_dir.split('/')[-1]
            if variant_dict is not None:
                wandb.config.update(variant_dict)

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        # TODO This should be removed
        self.discrete = False

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        new_obs_actions = self.policy(obs)
        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )

        crit = torch.nn.MSELoss()
        policy_loss = (-q_new_actions + self.beta*crit(actions, new_obs_actions)).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        new_next_actions = self.policy(next_obs)
        target_q_values = torch.min(self.target_qf1(next_obs, new_next_actions), self.target_qf2(next_obs, new_next_actions))

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            if self.log_pickle and self._log_epoch % self.pickle_log_rate == 0:
                new_path = os.path.join(self.log_dir,'model_pkl')
                if not os.path.isdir(new_path):
                    os.mkdir(new_path)
                torch.save({
                    'qf1_state_dict': self.qf1.state_dict(),
                    'qf2_state_dict': self.qf2.state_dict(),
                    'targetqf1_state_dict': self.target_qf1.state_dict(),
                    'targetqf2_state_dict': self.target_qf2.state_dict(),
                    'policy_state_dict': self.policy.state_dict(),
                }, os.path.join(new_path, str(self._log_epoch)+'.pt'))

            policy_loss = ((actions - q_new_actions)**2 + self.beta * q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            if self.wand_b:
                wandb.log({'trainer/'+k:v for k,v in self.eval_statistics.items()}, step=self._log_epoch)
            self._log_epoch += 1
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )

