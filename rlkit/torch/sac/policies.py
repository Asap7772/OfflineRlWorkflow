import numpy as np
import torch
from torch import nn as nn

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.core import eval_np
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.distributions import Normal
from rlkit.torch.networks import Mlp

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0

def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5*torch.log(one_plus_x/ one_minus_x)


class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            obs_processor=None,
            shared_encoder=False,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        self.obs_processor = obs_processor
        self.shared_encoder = shared_encoder # If shared encoder, don't backprop gradients through obs_processor

        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, state=None, deterministic=False, context_key='camera_orientation'):
        if type(obs_np) == dict:
            obs_np, context = obs_np['image'], np.array(list(obs_np[context_key].values()))[None]
        else: 
            context = None
        actions = self.get_actions(obs_np[None], deterministic=deterministic, fc_input=state[None] if state is not None else context)
        return actions[0, :], {}

    def get_actions(self, obs_np, state=None, deterministic=False, fc_input=None):
        return eval_np(self, obs_np, deterministic=deterministic, extra_fc_input=fc_input)[0]

    def log_prob(self, obs, actions, extra_fc_input=None,):
        raw_actions = atanh(actions)

        if self.obs_processor is None:
            h = obs
        else:
            h = obs
            if extra_fc_input is not None:
                h = torch.cat((h, extra_fc_input), dim=1)
            h = self.obs_processor(h)
            if self.shared_encoder:
                h = h.detach()

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        return log_prob.sum(-1)

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            extra_fc_input=None,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        # import ipdb; ipdb.set_trace()
        if self.obs_processor is None:
            h = obs
        else:
            h = obs
            if extra_fc_input is not None:
                h = torch.cat((h, extra_fc_input), dim=1)
            h = self.obs_processor(h)
            if self.shared_encoder:
                h = h.detach()
                        
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )


class GaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = GaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            obs_processor=None,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        self.obs_processor = obs_processor

        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False, context_key='camera_orientation'):
        if type(obs_np) == dict:
            obs_np, context = obs_np['image'], np.array(list(obs_np[context_key].values()))[None]
        else: 
            context = None
        actions = self.get_actions(obs_np[None], deterministic=deterministic, fc_input=context)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False, fc_input=None):
        return eval_np(self, obs_np, deterministic=deterministic, extra_fc_input=fc_input)[0]

    def log_prob(self, obs, actions, extra_fc_input=None,):
        raw_actions = atanh(actions)

        if self.obs_processor is None:
            h = obs
        else:
            h = obs
            if extra_fc_input is not None:
                h = torch.cat((h, extra_fc_input), dim=1)
            h = self.obs_processor(h)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        normal = Normal(mean, std)
        log_prob = normal.log_prob(actions)
        return log_prob.sum(-1)

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            extra_fc_input=None,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        # import ipdb; ipdb.set_trace()
        if self.obs_processor is None:
            h = obs
        else:
            h = obs
            if extra_fc_input is not None:
                h = torch.cat((h, extra_fc_input), dim=1)
            import ipdb; ipdb.set_trace()
            h = self.obs_processor(h)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = mean
        else:
            normal = Normal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action = normal.rsample()
                else:
                    action = normal.sample()
                log_prob = normal.log_prob(action)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = normal.rsample()
                else:
                    action = normal.sample()
        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

class MakeDeterministic(nn.Module, Policy):
    def __init__(self, stochastic_policy):
        super().__init__()
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation, state = None):
        if state is not None:
            return self.stochastic_policy.get_action(observation,state,
                                                 deterministic=True)
        else:
            return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)
