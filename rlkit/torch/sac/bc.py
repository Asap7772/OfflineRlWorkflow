from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
import wandb
import os

class BCTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            policy_lr=1e-3,
            optimizer_class=optim.Adam,
            log_dir=None,
            wand_b=True,
            variant_dict=None,
            real_data=False,
            log_pickle=True,
            pickle_log_rate=5,
            imgstate = False,
            validation=False,
            validation_buffer=None,
            bc_cql_comp=False,
            *args, **kwargs
    ):
        super().__init__()
        self.env = env
        self.imgstate = imgstate
        self.policy = policy
        self.log_dir = log_dir

        self.log_pickle=log_pickle
        self.pickle_log_rate=pickle_log_rate

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self._optimizer_class = optimizer_class #for loading

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self._current_epoch = 0
        self._log_epoch = 0

        self._num_policy_update_steps = 0
        self.discrete = False

        self.validation = validation
        self.validation_buffer = validation_buffer

        self.real_data = real_data
        self.log_dir = log_dir
        self.wand_b = wand_b

        if self.wand_b:
            wandb.init(project='bc_cql_comp' if bc_cql_comp else 'cog_cql', reinit=True)
            wandb.run.name=log_dir.split('/')[-1]
            if variant_dict is not None:
                wandb.config.update(variant_dict)

    def train_from_torch(self, batch, online=False):
        self._current_epoch += 1

        if self.imgstate:
            obs = batch['observations']
            state = batch['state']
        else:
            obs = batch['observations']
        actions = batch['actions']
        """
        Policy and Alpha Loss
        """
        """Start with BC"""
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True, extra_fc_input = state if self.imgstate else None,
        )
        alpha = 0.0
        policy_log_prob = self.policy.log_prob(obs, actions, extra_fc_input = state if self.imgstate else None)
        policy_loss = (alpha * log_pi - policy_log_prob).mean()
        """
        Update networks
        """
        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        self.policy_optimizer.step()

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            if self.log_pickle and self._log_epoch % self.pickle_log_rate == 0:
                new_path = os.path.join(self.log_dir,'model_pkl')
                if not os.path.isdir(new_path):
                    os.mkdir(new_path)
                torch.save({
                    'policy_state_dict': self.policy.state_dict(),
                }, os.path.join(new_path, str(self._log_epoch)+'.pt'))

            if self.validation:
                num_val = 32
                batch_val = self.validation_buffer.random_batch(num_val)
                if self.imgstate:
                    state_val = ptu.from_numpy(batch_val['state'])
                rewards_val = ptu.from_numpy(batch_val['rewards'])
                terminals_val = ptu.from_numpy(batch_val['terminals'])
                obs_val = ptu.from_numpy(batch_val['observations'])
                actions_val = ptu.from_numpy(batch_val['actions'])
                next_obs_val = ptu.from_numpy(batch_val['next_observations'])
                val_new_obs_actions, val_policy_mean, val_policy_log_std, val_log_pi, *_ = self.policy(
                    obs_val, reparameterize=True, return_log_prob=True, extra_fc_input=state if self.imgstate else None,
                )
                val_policy_log_prob = self.policy.log_prob(obs_val, actions_val, extra_fc_input=state if self.imgstate else None)
                val_policy_loss = (alpha * val_log_pi - val_policy_log_prob).mean()
                self.eval_statistics['Val Policy Loss'] = np.mean(ptu.get_numpy(
                    val_policy_loss
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Val Log Pis',
                    ptu.get_numpy(val_policy_log_prob),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Val Policy mu',
                    ptu.get_numpy(val_policy_mean),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Val Policy log std',
                    ptu.get_numpy(val_policy_log_std),
                ))

            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics[
                'Num Policy Updates'] = self._num_policy_update_steps
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
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
        base_list = [self.policy]
        return base_list
