from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
import torch.nn.functional as F


class RegressTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            network,
            network_lr=1e-3,
            optimizer_class=optim.Adam,
            alt_buffer=None,
            regress_key='object_positions'
    ):
        super().__init__()
        self.env = env
        self.network = network
        self.policy = network

        self.network_optimizer = optimizer_class(
            self.network.parameters(),
            lr=network_lr,
        )

        self._optimizer_class = optimizer_class #for loading

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self._current_epoch = 0
        self._num_network_update_steps = 0
        self.discrete = False
        self.alt_buffer = alt_buffer
        self.regress_key=regress_key

    def train_from_torch(self, batch, online=False):
        self._current_epoch += 1

        obs = batch['observations']
        orient = batch[self.regress_key]

        """Start with Regression"""
        pred = self.network(obs)
        network_loss = F.mse_loss(pred, orient)

        """
        Update networks
        """
        self._num_network_update_steps += 1
        self.network_optimizer.zero_grad()
        network_loss.backward(retain_graph=False)
        self.network_optimizer.step()

        if self.alt_buffer is not None:
            batch_alt = self.alt_buffer.random_batch(obs.shape[0])
            obs_new = ptu.from_numpy(batch_alt['observations'])
            orient_new = ptu.from_numpy(batch_alt[self.regress_key])
            pred = self.network(obs_new)
            network_val_loss = F.mse_loss(pred, orient_new)
        else:
            network_val_loss = network_loss

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics[
                'Num network Updates'] = self._num_network_update_steps
            self.eval_statistics['Network Train Loss'] = np.mean(ptu.get_numpy(
                network_loss
            ))
            self.eval_statistics['Network Val Loss'] = np.mean(ptu.get_numpy(
                network_val_loss
            ))
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [self.network]
        return base_list
