import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
import hydra


class AdaptiveConvOut(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim,
                 adaptive_type="AdaptiveAvgPool2d", conv_out_size=35, keep_size=True):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = num_filters = 32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim
        self.keep_size = keep_size
        orig_dim = 35

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),
            nn.Conv2d(num_filters, num_filters, 3, stride=1),
            nn.Conv2d(num_filters, num_filters, 3, stride=1),
            nn.Conv2d(num_filters, num_filters, 3, stride=1)
        ])

        self.convs_out = getattr(nn, adaptive_type)((conv_out_size, conv_out_size))

        if keep_size:
            scale_factor = orig_dim / float(conv_out_size)
            self.upscale = nn.Upsample(scale_factor=scale_factor, mode='nearest')
            conv_out_size = orig_dim

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * conv_out_size * conv_out_size, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        conv = self.convs_out(conv)
        if self.keep_size:
            conv = self.upscale(conv)

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
            if len(v.shape) > 2:
                logger.log_image(f'train_encoder/{k}_img', v[0], step)

        for i in range(self.num_layers):
            logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)


class AdaptiveAvg(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim, pool_size=2):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = num_filters =  32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim
        self.pool_size = pool_size

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),
            nn.Conv2d(num_filters, num_filters, 3, stride=1),
            nn.Conv2d(num_filters, num_filters, 3, stride=1),
            nn.Conv2d(num_filters, num_filters, 3, stride=1)
        ])

        avg = nn.Conv2d(num_filters, num_filters, pool_size, stride=1, padding=pool_size//2, bias=False)
        avg.weight.data.fill_(1.)
        avg.weight.requires_grad = False

        self.avg = (avg, )

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * 35 * 35, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        if self.avg[0].weight.device != self.convs[0].weight.device:
            avg = self.avg[0].to(self.convs[0].weight.device)
            self.avg = (avg,)

        conv = self.avg[0](conv) / (self.pool_size ** 2)

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
            if len(v.shape) > 2:
                logger.log_image(f'train_encoder/{k}_img', v[0], step)

        for i in range(self.num_layers):
            logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)

