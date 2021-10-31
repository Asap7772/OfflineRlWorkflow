from rlkit.torch.networks import Mlp
import torch
from torch import nn as nn
from torch.distributions import Normal
from rlkit.pythonplusplus import identity
from rlkit.torch.vae.vq_vae import Encoder

import numpy as np


class CNN(nn.Module):
    # TODO: remove the FC parts of this code
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes=None,
            added_fc_input_size=0,
            conv_normalization_type='none',
            fc_normalization_type='none',
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            output_conv_channels=False,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,
            image_augmentation=False,
            image_augmentation_padding=4,
            spectral_norm_conv=False,
            spectral_norm_fc=False,
            normalize_conv_activation=False,
            dropout=False,
            dropout_prob=0.2,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert conv_normalization_type in {'none', 'batch', 'layer'}
        assert fc_normalization_type in {'none', 'batch', 'layer'}
        assert pool_type in {'none', 'max2d'}
        if pool_type == 'max2d':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.conv_normalization_type = conv_normalization_type
        self.fc_normalization_type = fc_normalization_type
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.output_conv_channels = output_conv_channels
        self.pool_type = pool_type
        self.image_augmentation = image_augmentation
        self.image_augmentation_padding = image_augmentation_padding

        self.dropout = dropout
        self.dropout_prob = dropout_prob

        self.dropout_layer = nn.Dropout(p=self.dropout_prob)
        self.dropout2d_layer = nn.Dropout2d(p=self.dropout_prob)

        self.spectral_norm_conv = spectral_norm_conv
        self.spectral_norm_fc = spectral_norm_fc

        self.normalize_conv_activation = normalize_conv_activation
        
        if normalize_conv_activation:
            print('normalizing conv activation')

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(n_channels, kernel_sizes, strides, paddings)
        ):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            if self.spectral_norm_conv and 0 < i < len(n_channels)-1:
                conv = nn.utils.spectral_norm(conv)

            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

            if pool_type == 'max2d':
                if pool_sizes[i] > 1:
                    self.pool_layers.append(
                        nn.MaxPool2d(
                            kernel_size=pool_sizes[i],
                            stride=pool_strides[i],
                            padding=pool_paddings[i],
                        )
                    )

        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )
        #find output dim of conv_layers by trial and add norm conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            test_mat = conv_layer(test_mat)
            if self.conv_normalization_type == 'batch':
                self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
            if self.conv_normalization_type == 'layer':
                self.conv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))
            if self.pool_type != 'none' and len(self.pool_layers) > i:
                test_mat = self.pool_layers[i](test_mat)

        self.conv_output_flat_size = self.get_conv_output_size()
        
        if self.spectral_norm_fc:
            print('Building FC layers with spectral norm')

        if self.output_conv_channels:
            self.last_fc = None
        else:
            fc_input_size = self.conv_output_flat_size
            # used only for injecting input directly into fc layers
            fc_input_size += added_fc_input_size
            for idx, hidden_size in enumerate(hidden_sizes):
                fc_layer = nn.Linear(fc_input_size, hidden_size)
                if self.spectral_norm_fc and 0 < idx < len(hidden_sizes)-1:
                    fc_layer = nn.utils.spectral_norm(fc_layer)
                fc_input_size = hidden_size

                fc_layer.weight.data.uniform_(-init_w, init_w)
                fc_layer.bias.data.uniform_(-init_w, init_w)

                self.fc_layers.append(fc_layer)

                if self.fc_normalization_type == 'batch':
                    self.fc_norm_layers.append(nn.BatchNorm1d(hidden_size))
                if self.fc_normalization_type == 'layer':
                    self.fc_norm_layers.append(nn.LayerNorm(hidden_size))

            self.last_fc = nn.Linear(fc_input_size, output_size)
            self.last_fc.weight.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.uniform_(-init_w, init_w)

        if self.image_augmentation:
            self.augmentation_transform = RandomCrop(
                input_height, self.image_augmentation_padding, device='cuda')

    def get_conv_output_size(self):
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )

        test_mat = self.apply_forward_conv(test_mat)
        return int(np.prod(test_mat.shape))

    def forward(self, input, return_last_activations=False, return_conv_outputs=False):
        conv_input = input.narrow(start=0,
                                  length=self.conv_input_length,
                                  dim=1).contiguous()
        # reshape from batch of flattened images into (channels, w, h)
        h = conv_input.view(conv_input.shape[0],
                            self.input_channels,
                            self.input_height,
                            self.input_width)

        if h.shape[0] > 1 and self.image_augmentation:
            # h.shape[0] > 1 ensures we apply this only during training
            h = self.augmentation_transform(h)

        h = self.apply_forward_conv(h)
        if self.output_conv_channels:
            return h

        # flatten channels for fc layers
        h = h.view(h.size(0), -1)
        
        if self.normalize_conv_activation:
            h = h/(torch.norm(h)+1e-9)
        conv_outputs_flat = h

        if self.added_fc_input_size != 0:
            extra_fc_input = input.narrow(
                start=self.conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )
            h = torch.cat((h, extra_fc_input), dim=1)

        h = self.apply_forward_fc(h)

        if return_last_activations:
            return h

        if return_conv_outputs:
            return self.output_activation(self.last_fc(h)), conv_outputs_flat
            # return self.output_activation(self.last_fc(h)), h # for dr3 last layer
        else:
            return self.output_activation(self.last_fc(h))

    def apply_forward_conv(self, h):
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if self.conv_normalization_type != 'none':
                h = self.conv_norm_layers[i](h)
            if self.pool_type != 'none' and len(self.pool_layers) > i:
                h = self.pool_layers[i](h)
            if self.dropout:
                h = self.dropout2d_layer(h)
            h = self.hidden_activation(h)
        return h

    def apply_forward_fc(self, h):
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.fc_normalization_type != 'none':
                h = self.fc_norm_layers[i](h)
            if self.dropout:
                h = self.dropout_layer(h)
            h = self.hidden_activation(h)
        return h



class RegressCNN(CNN):
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        
        init_w=1e-4
        self.regress_layer = nn.Linear(conv_output_flat_size, dim)
        self.regress_layer.weight.data.uniform_(-init_w, init_w)
        self.regress_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_last_activations=False):
        conv_input = input.narrow(start=0,
                                  length=self.conv_input_length,
                                  dim=1).contiguous()
        # reshape from batch of flattened images into (channels, w, h)
        h = conv_input.view(conv_input.shape[0],
                            self.input_channels,
                            self.input_height,
                            self.input_width)

        if h.shape[0] > 1 and self.image_augmentation:
            # h.shape[0] > 1 ensures we apply this only during training
            h = self.augmentation_transform(h)

        h = self.apply_forward_conv(h)
        if self.output_conv_channels:
            return h

        # flatten channels for fc layers
        h = h.view(h.size(0), -1)

        regress_out = self.regress_layer(h)

        if self.added_fc_input_size != 0:
            extra_fc_input = input.narrow(
                start=self.conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )
            h = torch.cat((h, extra_fc_input), dim=1)
        h = self.apply_forward_fc(h)

        if return_last_activations:
            return h
        return self.output_activation(self.last_fc(h)), regress_out

class ConcatMlp(Mlp):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)
        
class ConcatCNN(CNN):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)

class ConcatCNNWrapperRegress(nn.Module):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, model, output_size, action_dim):
        super().__init__()
        self.output_size = output_size
        self.model = model
        self.model.output_conv_channels = True
        for param in self.model.parameters():
            param.requires_grad = False
        self.mlp = Mlp([512,512,512],output_size,self.model.fc_layers[0].in_features-action_dim)

    def forward(self, *inputs, **kwargs):
        tensor = self.model.forward(*inputs,**kwargs).flatten(1)
        return self.mlp(tensor)

class ConcatBottleneckCNN(CNN):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, action_dim, bottleneck_dim=16, output_size=1, dim=1, deterministic=False, width=48, height=48,
                 spectral_norm_conv=False,spectral_norm_fc=False):
        cnn_params=dict(
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[1, 1, 1],
            hidden_sizes=[1024], #mean/std
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
            image_augmentation=True,
            image_augmentation_padding=4,
        )
        
        cnn_params.update(
            input_width=width,
            input_height=height,
            input_channels=3,
            output_size=bottleneck_dim*2,
            added_fc_input_size=action_dim,
            spectral_norm_fc=spectral_norm_fc,
            spectral_norm_conv=spectral_norm_conv
        )

        self.cnn_params = cnn_params
        self.action_dim = action_dim

        super().__init__(**cnn_params)
        self.bottleneck_dim = bottleneck_dim
        self.deterministic=deterministic
        self.mlp = Mlp([512,512,512],output_size,self.cnn_params['output_size']//2, spectral_norm=spectral_norm_fc)
        self.dim = dim
        self.output_conv_channels = False

    def forward(self, *inputs, **kwargs):
        forward_outs = self.detailed_forward(*inputs, **kwargs)
        if 'return_conv_outputs' in kwargs and kwargs['return_conv_outputs']:
            outs = forward_outs[0]
            conv_outs = forward_outs[1]
        else:
            outs = forward_outs

        if self.output_conv_channels:
            ret_val = outs[-1]
        else:
            ret_val = outs[0]
        if 'return_conv_outputs' in kwargs and kwargs['return_conv_outputs']:
            return ret_val, conv_outs
        return ret_val

    def detailed_forward(self, *inputs, **kwargs): # used to calculate loss
        flat_inputs = torch.cat(inputs, dim=self.dim)
        ret_conv = False
        if 'return_conv_outputs' in kwargs and kwargs['return_conv_outputs']:
            ret_conv = True
            cnn_out, conv_outputs = super().forward(flat_inputs, **kwargs)
        else:
            cnn_out = super().forward(flat_inputs, **kwargs)
        size = self.cnn_params['output_size']//2
        mean, log_std = cnn_out[:,:size], cnn_out[:,size:]
        assert mean.shape == log_std.shape
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        sample = dist.rsample()
        log_prob = dist.log_prob(sample)
        # equation is $$\frac{1}{2}[\mu^T\mu + tr(\Sigma) -k -log|\Sigma|]$$
        K = self.bottleneck_dim
        reg_loss = 1/2*(mean.norm(dim=-1, keepdim=True)**2 + (std**2).sum(axis=-1, keepdim=True)- K - torch.log(std**2+1E-9).sum(axis=-1, keepdim=True))

        if self.deterministic:
            sample=mean
            log_prob= -torch.ones_like(log_prob).cuda()
            reg_loss = torch.zeros_like(reg_loss).cuda()

        if ret_conv:
            return (self.mlp(sample), log_prob, reg_loss, mean, log_std, sample), conv_outputs
        else:
            return self.mlp(sample), log_prob, reg_loss, mean, log_std, sample


class ConcatBottleneckActionCNN(CNN):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, action_dim, bottleneck_dim=16, output_size=1, dim=1, deterministic=False):
        cnn_params=dict(
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[1, 1, 1],
            hidden_sizes=[1024], #mean/std
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
            image_augmentation=True,
            image_augmentation_padding=4,
        )
        
        cnn_params.update(
            input_width=48,
            input_height=48,
            input_channels=3,
            output_size=bottleneck_dim*2,
            added_fc_input_size=action_dim,
        )

        self.cnn_params = cnn_params
        self.action_dim = action_dim

        super().__init__(**cnn_params)
        self.bottleneck_dim = bottleneck_dim
        self.deterministic=deterministic
        self.mlp = Mlp([512,512,512],output_size,self.cnn_params['output_size']//2)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        return self.detailed_forward(*inputs, **kwargs)[0]

    def detailed_forward(self, *inputs, **kwargs): # used to calculate loss
        flat_inputs = torch.cat(inputs, dim=self.dim)
        cnn_out = super().forward(flat_inputs, **kwargs)
        size = self.cnn_params['output_size']//2
        mean, log_std = cnn_out[:,:size], cnn_out[:,size:]
        assert mean.shape == log_std.shape
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        sample = dist.rsample()
        log_prob = dist.log_prob(sample)
        # equation is $$\frac{1}{2}[\mu^T\mu + tr(\Sigma) -k -log|\Sigma|]$$
        K = self.bottleneck_dim
        reg_loss = 1/2*(mean.norm(dim=-1, keepdim=True)**2 + (std**2).sum(axis=-1, keepdim=True)- K - torch.log(std**2+1E-9).sum(axis=-1, keepdim=True))

        if self.deterministic:
            sample=mean
            log_prob= -torch.ones_like(log_prob).cuda()
            reg_loss = torch.zeros_like(reg_loss).cuda()
        return self.mlp(sample), log_prob, reg_loss, mean, log_std, sample

class TwoHeadCNN(CNN):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, action_dim, concat_size=256, output_size=1, dim=1, deterministic=False, bottleneck_dim=16, width=48,height=48): #TODO implement bottleneck
        cnn_params=dict(
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[1, 1, 1],
            hidden_sizes=[1024], #mean/std
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
            image_augmentation=True,
            image_augmentation_padding=4,
        )
        
        cnn_params.update(
            input_width=width,
            input_height=height,
            input_channels=3,
            output_size=concat_size,
            added_fc_input_size=action_dim,
        )

        self.cnn_params = cnn_params
        self.action_dim = action_dim

        super().__init__(**cnn_params)
        self.concat_size = concat_size
        self.deterministic=deterministic
        self.head1 = Mlp([512,512,512],output_size,self.cnn_params['output_size'])
        self.head2 = Mlp([512,512,512],output_size,self.cnn_params['output_size'])
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        cnn_out = super().forward(flat_inputs, **kwargs)
        return self.head1(cnn_out), self.head2(cnn_out)

class ConcatRegressCNN(RegressCNN):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)


class TwoHeadDCNN(nn.Module):
    def __init__(
            self,
            fc_input_size,
            hidden_sizes,

            deconv_input_width,
            deconv_input_height,
            deconv_input_channels,

            deconv_output_kernel_size,
            deconv_output_strides,
            deconv_output_channels,

            kernel_sizes,
            n_channels,
            strides,
            paddings,

            batch_norm_deconv=False,
            batch_norm_fc=False,
            init_w=1e-3,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
    ):
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        self.deconv_input_width = deconv_input_width
        self.deconv_input_height = deconv_input_height
        self.deconv_input_channels = deconv_input_channels
        deconv_input_size = self.deconv_input_channels * self.deconv_input_height * self.deconv_input_width
        self.batch_norm_deconv = batch_norm_deconv
        self.batch_norm_fc = batch_norm_fc

        self.deconv_layers = nn.ModuleList()
        self.deconv_norm_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)

            norm_layer = nn.BatchNorm1d(hidden_size)
            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)

            self.fc_layers.append(fc_layer)
            self.fc_norm_layers.append(norm_layer)
            fc_input_size = hidden_size

        self.last_fc = nn.Linear(fc_input_size, deconv_input_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        for out_channels, kernel_size, stride, padding in \
                zip(n_channels, kernel_sizes, strides, paddings):
            deconv = nn.ConvTranspose2d(deconv_input_channels,
                                        out_channels,
                                        kernel_size,
                                        stride=stride,
                                        padding=padding)
            hidden_init(deconv.weight)
            deconv.bias.data.fill_(0)

            deconv_layer = deconv
            self.deconv_layers.append(deconv_layer)
            deconv_input_channels = out_channels

        test_mat = torch.zeros(1, self.deconv_input_channels,
                               self.deconv_input_width,
                               self.deconv_input_height)  # initially the model is on CPU (caller should then move it to GPU if
        for deconv_layer in self.deconv_layers:
            test_mat = deconv_layer(test_mat)
            self.deconv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))

        self.first_deconv_output = nn.ConvTranspose2d(
            deconv_input_channels,
            deconv_output_channels,
            deconv_output_kernel_size,
            stride=deconv_output_strides,
        )
        hidden_init(self.first_deconv_output.weight)
        self.first_deconv_output.bias.data.fill_(0)

        self.second_deconv_output = nn.ConvTranspose2d(
            deconv_input_channels,
            deconv_output_channels,
            deconv_output_kernel_size,
            stride=deconv_output_strides,
        )
        hidden_init(self.second_deconv_output.weight)
        self.second_deconv_output.bias.data.fill_(0)

    def forward(self, input):
        h = self.apply_forward(input, self.fc_layers, self.fc_norm_layers,
                               use_batch_norm=self.batch_norm_fc)
        h = self.hidden_activation(self.last_fc(h))
        h = h.view(-1, self.deconv_input_channels, self.deconv_input_width,
                   self.deconv_input_height)
        h = self.apply_forward(h, self.deconv_layers, self.deconv_norm_layers,
                               use_batch_norm=self.batch_norm_deconv)
        first_output = self.output_activation(self.first_deconv_output(h))
        second_output = self.output_activation(self.second_deconv_output(h))
        return first_output, second_output

    def apply_forward(self, input, hidden_layers, norm_layers,
                      use_batch_norm=False):
        h = input
        for layer, norm_layer in zip(hidden_layers, norm_layers):
            h = layer(h)
            if use_batch_norm:
                h = norm_layer(h)
            h = self.hidden_activation(h)
        return h


class DCNN(TwoHeadDCNN):
    def forward(self, x):
        return super().forward(x)[0]


class RandomCrop:
    """
    Source: # https://github.com/pratogab/batch-transforms
    Applies the :class:`~torchvision.transforms.RandomCrop` transform to
    a batch of images.
    Args:
        size (int): Desired output size of the crop.
        padding (int, optional): Optional padding on each border of the image.
            Default is None, i.e no padding.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, size, padding=None, dtype=torch.float, device='cpu'):
        self.size = size
        self.padding = padding
        self.dtype = dtype
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        if self.padding is not None:
            padded = torch.zeros((tensor.size(0), tensor.size(1),
                                  tensor.size(2) + self.padding * 2,
                                  tensor.size(3) + self.padding * 2),
                                 dtype=self.dtype, device=self.device)
            padded[:, :, self.padding:-self.padding,
            self.padding:-self.padding] = tensor
        else:
            padded = tensor

        w, h = padded.size(2), padded.size(3)
        th, tw = self.size, self.size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = torch.randint(0, h - th + 1, (tensor.size(0),),
                              device=self.device)
            j = torch.randint(0, w - tw + 1, (tensor.size(0),),
                              device=self.device)

        rows = torch.arange(th, dtype=torch.long, device=self.device) + i[:,
                                                                        None]
        columns = torch.arange(tw, dtype=torch.long, device=self.device) + j[:,
                                                                           None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[:, torch.arange(tensor.size(0))[:, None, None],
                 rows[:, torch.arange(th)[:, None]], columns[:, None]]
        return padded.permute(1, 0, 2, 3)


class VQVAEEncoderConcatCNN(ConcatCNN):
    def __init__(self, *args, num_res=3, **kwargs):
        kwargs['kernel_sizes'] = []
        kwargs['n_channels'] = []
        kwargs['strides'] = []
        kwargs['paddings'] = []
        super().__init__(*args, **kwargs)
        
        self.encoder = Encoder(self.input_channels, 128, num_res, 64, spectral_norm=kwargs['spectral_norm_conv'] if 'spectral_norm_conv' in kwargs else False, input_dim=kwargs['input_width'])

    def apply_forward_conv(self, h):
        out = self.encoder(h)
        return out

    def get_conv_output_size(self):
        if self.input_width == self.input_height == 48:
            return 128 * 12 * 12
        elif self.input_width == self.input_height == 64:
            return 128 * 16 * 16
        else:
            raise ValueError

    def forward(self, *inputs, **kwargs):
        return super().forward(*inputs, **kwargs)


class VQVAEEncoderCNN(CNN):
    def __init__(self, *args, num_res=3, **kwargs):
        kwargs['kernel_sizes'] = []
        kwargs['n_channels'] = []
        kwargs['strides'] = []
        kwargs['paddings'] = []
        super().__init__(*args, **kwargs)

        self.encoder = Encoder(self.input_channels, 128, num_res, 64, spectral_norm=kwargs['spectral_norm_conv'] if 'spectral_norm_conv' in kwargs else False, input_dim=kwargs['input_width'])

    def apply_forward_conv(self, h):
        out = self.encoder(h)
        return out

    def get_conv_output_size(self):
        if self.input_width == self.input_height == 48:
            return 128 * 12 * 12
        elif self.input_width == self.input_height == 64:
            return 128 * 16 * 16
        else:
            raise ValueError

    def forward(self, *inputs, **kwargs):
        return super().forward(*inputs, **kwargs)

class ConcatBottleneckVQVAECNN(VQVAEEncoderConcatCNN):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """

    def __init__(self, action_dim, bottleneck_dim=16, output_size=1, dim=1, deterministic=False, width=48, height=48,
                 spectral_norm_conv=False, spectral_norm_fc=False):
        cnn_params = dict(
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[1, 1, 1],
            hidden_sizes=[1024],  # mean/std
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
            image_augmentation=True,
            image_augmentation_padding=4,
        )

        cnn_params.update(
            input_width=width,
            input_height=height,
            input_channels=3,
            output_size=bottleneck_dim * 2,
            added_fc_input_size=action_dim,
            spectral_norm_fc=spectral_norm_fc,
            spectral_norm_conv=spectral_norm_conv
        )

        self.cnn_params = cnn_params
        self.action_dim = action_dim

        super().__init__(**cnn_params)
        self.bottleneck_dim = bottleneck_dim
        self.deterministic = deterministic
        self.mlp = Mlp([512, 512, 512], output_size, self.cnn_params['output_size'] // 2, spectral_norm=cnn_params['spectral_norm_fc'])
        self.dim = dim
        self.output_conv_channels = False

    def forward(self, *inputs, **kwargs):
        forward_outs = self.detailed_forward(*inputs, **kwargs)
        if 'return_conv_outputs' in kwargs and kwargs['return_conv_outputs']:
            outs = forward_outs[0]
            conv_outs = forward_outs[1]
        else:
            outs = forward_outs
        if self.output_conv_channels:
            ret_val = outs[-1]
        else:
            ret_val = outs[0]
        if 'return_conv_outputs' in kwargs and kwargs['return_conv_outputs']:
            return ret_val, conv_outs
        return ret_val

    def detailed_forward(self, *inputs, **kwargs):  # used to calculate loss
        flat_inputs = torch.cat(inputs, dim=self.dim)
        ret_conv = False
        if 'return_conv_outputs' in kwargs and kwargs['return_conv_outputs']:
            ret_conv = True
            cnn_out, conv_outputs = super().forward(flat_inputs, **kwargs)
        else:
            cnn_out = super().forward(flat_inputs, **kwargs)
        size = self.cnn_params['output_size'] // 2
        mean, log_std = cnn_out[:, :size], cnn_out[:, size:]
        assert mean.shape == log_std.shape
        std = torch.exp(log_std)

        dist = Normal(mean, std)
        sample = dist.rsample()
        log_prob = dist.log_prob(sample)
        # equation is $$\frac{1}{2}[\mu^T\mu + tr(\Sigma) -k -log|\Sigma|]$$
        K = self.bottleneck_dim
        reg_loss = 1 / 2 * (
                    mean.norm(dim=-1, keepdim=True) ** 2 + (std ** 2).sum(axis=-1, keepdim=True) - K - torch.log(
                std ** 2 + 1E-9).sum(axis=-1, keepdim=True))

        if self.deterministic:
            sample = mean
            log_prob = -torch.ones_like(log_prob).cuda()
            reg_loss = torch.zeros_like(reg_loss).cuda()

        if ret_conv:
            return (self.mlp(sample), log_prob, reg_loss, mean, log_std, sample), conv_outputs
        else:
            return self.mlp(sample), log_prob, reg_loss, mean, log_std, sample


