from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import log2_, nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
import wandb
import os


class BRACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            behavior_policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            beta = 1.0,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            
            log_dir=None,
            wand_b=True,
            variant_dict=None,
            real_data=False,
            log_pickle=True,
            pickle_log_rate=5,
            continual=False,
            bottleneck=False,
            bottleneck_type='policy',
            bottleneck_const=1.0,
            start_bottleneck=0,
            *args,
            **kwargs
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.behavior_policy = behavior_policy
        self.beta = beta
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.bottleneck_type=bottleneck_type
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.real_data = real_data
        self.continual = continual
        self.bottleneck = bottleneck
        self.bottleneck_const = bottleneck_const

        self.log_pickle = log_pickle
        self.pickle_log_rate = pickle_log_rate
        self._log_epoch = 0
        self.log_dir = log_dir
        self.wand_b = wand_b
        self.start_bottleneck =start_bottleneck
        
        if self.wand_b:
            wandb.init(project='cog_brac', reinit=True)
            wandb.run.name=log_dir.split('/')[-1]
            if variant_dict is not None:
                wandb.config.update(variant_dict)

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self.behavior_policy_optimizer = optimizer_class(
            self.behavior_policy.parameters(),
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
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )

        log_pi_behavior = self.behavior_policy.log_prob(obs,new_obs_actions)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - self.beta * log_pi_behavior - q_new_actions).mean()


        if self.continual:
            log_pi_behaviordata = self.behavior_policy.log_prob(obs,actions)
            policy_loss_behavioral = -log_pi_behaviordata.mean()
            
            self.behavior_policy_optimizer.zero_grad()
            policy_loss_behavioral.backward()
            self.behavior_policy_optimizer.step()


        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )

        new_log_pi_behavior = self.behavior_policy.log_prob(next_obs, new_next_actions)

        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi + self.beta * new_log_pi_behavior[None].T

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values

        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
        
        """
        Bottleneck Loss
        """
        if self.bottleneck:
            cond = self._log_epoch < self.start_bottleneck 
            w = (0 if cond else self.bottleneck_const)

            if self.bottleneck_type=='policy':
                reg_loss = self.policy.obs_processor.detailed_forward(obs,return_conv_outputs=False)[2]
                policy_loss = policy_loss + w * reg_loss.mean()
            elif self.bottleneck_type=='qf':
                _, _, qf1_bottleneck_loss, _, _, _ = self.qf1.detailed_forward(obs,actions)
                _, _, qf2_bottleneck_loss, _, _, _ = self.qf2.detailed_forward(obs,actions)
                qf1_loss = qf1_loss + w * qf1_bottleneck_loss.mean()
                qf2_loss = qf2_loss + w * qf2_bottleneck_loss.mean() 
                reg_loss = qf1_bottleneck_loss.mean() + qf2_bottleneck_loss.mean()
            else:
                raise NotImplementedError
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

            policy_loss = (log_pi - q_new_actions).mean()

            log1 = (q_new_actions + self.beta * log_pi_behavior)
            log2 = (q_new_actions - self.beta * log_pi_behavior)

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
            if self.bottleneck:
                self.eval_statistics['Bottleneck Reg Loss'] = np.mean(ptu.get_numpy(reg_loss))

            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'Conservative Q Value Plus',
                ptu.get_numpy(log1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Conservative Q Value Minus',
                ptu.get_numpy(log2),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis Behavior',
                ptu.get_numpy(log_pi_behavior),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
            
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