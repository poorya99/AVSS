#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

EPS = 1e-8
import copy
def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class ResBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(1, out_dims, eps=1e-8)
        self.norm2 = nn.GroupNorm(1, out_dims, eps=1e-8)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.mp = nn.AvgPool1d(3)
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        else:
            self.downsample = False

    def forward(self, x):
        # print("ResBlock input", x.shape)
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.downsample:
            residual = self.conv_downsample(residual)
        x = x + residual
        x = self.prelu2(x)
        x = self.mp(x)
        # print("ResBlock output", x.shape)
        return x

class SpeakerEncoder(nn.Module):
    def __init__(self, B=256, num_speaker=800):
        super(SpeakerEncoder, self).__init__()
        self.spk_encoder = speaker_encoder(B)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, s):
        # print("=== SpeakerEncoder Forward Pass ===")
        # print(f"Input shape: {s.shape}")  # [batch, B, K]

        x, x_avg = self.spk_encoder(s)
        # print(f"After speaker_encoder: x shape: {x.shape}, x_avg shape: {x_avg.shape}")  # [batch, H, K], [batch, H]

        return x, x_avg

class speaker_encoder(nn.Module):
    def __init__(self, B, R=3, H=256):
        super(speaker_encoder, self).__init__()
        self.layer_norm = ChannelWiseLayerNorm(B)
        self.bottleneck_conv1x1 = nn.Conv1d(B, B, 1, bias=False)

        self.mynet = nn.Sequential(
            ResBlock(B, B),
            ResBlock(B, H),
            ResBlock(H, H),
            nn.Dropout(0.9),
            nn.Conv1d(H, B, 1, bias=False)
        )
        self.avgPool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # print("=== speaker_encoder Forward Pass ===")
        # print(f"Input shape: {x.shape}")  # [batch, B, K]

        x = self.layer_norm(x)
        # print(f"After layer_norm: {x.shape}")  # [batch, B, K]

        x = self.bottleneck_conv1x1(x)
        # print(f"After bottleneck_conv1x1: {x.shape}")  # [batch, B, K]

        x = self.mynet(x)
        # print(f"After mynet: {x.shape}")  # [batch, B, K]

        x_avg = self.avgPool(x)
        # print(f"After avgPool: {x_avg.shape}")  # [batch, B, 1]

        x_avg = x_avg.squeeze(2)
        # print(f"After squeeze avgPool: {x_avg.shape}")  # [batch, B]

        return x, x_avg


class audioEncoder(nn.Module):
    def __init__(self, L, N):
        super(audioEncoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.conv1d_U(x))
        return x

class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


EPS = 1e-8

def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class reentry_net(nn.Module):
    def __init__(self, N, L, B, H, P, X, R, C, M, V, pretrained_v=1):
        super(reentry_net, self).__init__()
        self.pretrained_v = pretrained_v

        # Audio encoder and bottleneck layer
        self.a_encoder = audioEncoder(L, N)
        self.a_norm = nn.Sequential(ChannelWiseLayerNorm(N), nn.Conv1d(N, B, 1, bias=False))

        # Video encoder and adaptation layer for each TCN

        self.v_encoder = videoEncoder(input_dim=300, intermediate_dim=256, R=R)
        self.v_adapt_mask = _clones(nn.Conv1d(V, B, 1, bias=False), R)

        # Temporal CNN blocks
        self.projection_0 = nn.Conv1d(B * 2, B, 1, bias=False)
        self.projection = _clones(nn.Conv1d(B * 3, B, 1, bias=False), R - 1)
        self.tcn = _clones(TCN_block(X, P, B, H), R)

        # Speaker embedding extraction and classification
        self.spk_encoder = _clones(SpeakerEncoder(B), R - 1)
        self.spk_classifier = _clones(nn.Linear(B, M), R - 1)

        # Mask generation layer
        self.a_decoder = audioDecoder(B, N, L)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        # Pretrained models
        # self.v_front_end = slsyn_net()
        # Uncomment and modify the following lines if you have pretrained weights
        # pretrained_model = torch.load("../../pretrained_slsyn/slsyn_model_dict.pt")['model']
        # state = self.v_front_end.state_dict()
        # for key in state.keys():
        #     pretrain_key = 'module.' + key
        #     state[key] = pretrained_model.get(pretrain_key, state[key])
        # self.v_front_end.load_state_dict(state)
        # if self.pretrained_v:
        #     for param in self.v_front_end.parameters():
        #         param.requires_grad = False

    def forward(self, mixture, v):
        #print("=== reentry_net Forward Pass ===")
        if not self.pretrained_v:
            #print("Using v_front_end to process inputs.")
            v = self.v_front_end(mixture, v).transpose(1, 2)
            #print(f"After v_front_end and transpose: {v.shape}")

        else:
          pass
          # v = v.transpose(1, 2)
          #print(f"After and transpose: {v.shape}")
        T_origin = mixture.size(-1)
        #print(f"T_origin (Original Time): {T_origin}")

        # Audio encoder
        #print("\n-- Audio Encoder --")
        mixture_w = self.a_encoder(mixture)
        #print(f"mixture_w shape: {mixture_w.shape}")  # [batch, N, new_time]

        est_mask = self.a_norm(mixture_w)
        #print(f"est_mask shape after a_norm: {est_mask.shape}")  # [batch, B, new_time]

        # Video encoder
        #print("\n-- Video Encoder --")
        v = self.v_encoder(v).transpose(1, 2)
        #print(f"v after videoEncoder: {v.shape}")  # [batch, V, time]

        spks = []
        # TCN blocks
        for i in range(len(self.tcn)):
            #print(f"\n-- TCN Block {i + 1} --")
            v_mask = self.v_adapt_mask[i](v)
            #print(f"v_mask shape after v_adapt_mask[{i}]: {v_mask.shape}")  # [batch, B, time]

            # Interpolate to match est_mask's time dimension if necessary
            scale_factor = est_mask.size(2) / v_mask.size(2)
            if scale_factor != 1:
                v_mask = F.interpolate(v_mask, scale_factor=scale_factor, mode='linear', align_corners=False)
                #print(f"v_mask shape after interpolate with size {scale_factor}: {v_mask.shape}")  # [batch, B, est_time]

            # Pad if necessary
            pad_size = est_mask.size(2) - v_mask.size(2)
            if pad_size > 0:
                v_mask = F.pad(v_mask, (0, pad_size))
                #print(f"v_mask shape after padding: {v_mask.shape}")  # [batch, B, est_time]
            elif pad_size < 0:
                v_mask = v_mask[:, :, :est_mask.size(2)]
                #print(f"v_mask shape after cropping: {v_mask.shape}")  # [batch, B, est_time]

            if i == 0:
                est_mask = torch.cat((est_mask, v_mask), dim=1)
                #print(f"est_mask shape after concatenation with v_mask: {est_mask.shape}")  # [batch, 2B, est_time]
                est_mask = self.projection_0(est_mask)
                #print(f"est_mask shape after projection_0: {est_mask.shape}")  # [batch, B, est_time]
                est_mask = self.tcn[i](est_mask)
                #print(f"est_mask shape after TCN[{i}]: {est_mask.shape}")  # [batch, B, est_time]
            else:
                #print("Running Speaker Encoder and Classifier.")
                # Decode mixture_w and est_mask to obtain speaker embeddings
                est_source = self.a_decoder(mixture_w, est_mask, T_origin)
                #print(f"est_source shape: {est_source.shape}")  # [batch, C, T_origin]

                # Encode speaker embeddings
                spk_emb, spk_emb_avg = self.spk_encoder[i - 1](self.a_encoder(est_source))
                #print(f"spk_emb shape: {spk_emb.shape}, spk_emb_avg shape: {spk_emb_avg.shape}")  # [batch, B, K], [batch, B]

                if i <= 3:
                    spk_emb = torch.repeat_interleave(spk_emb_avg.unsqueeze(2), repeats=est_mask.size(2), dim=2)
                    #print(f"spk_emb after repeat_interleave: {spk_emb.shape}")  # [batch, B, est_time]

                spk_class = self.spk_classifier[i - 1](spk_emb_avg)
                #print(f"spk_class shape: {spk_class.shape}")  # [batch, M]
                spks.append(spk_class)

                est_mask = torch.cat((spk_emb, est_mask, v_mask), dim=1)
                #print(f"est_mask shape after concatenating spk_emb, est_mask, v_mask: {est_mask.shape}")  # [batch, 3B, est_time]
                est_mask = self.projection[i - 1](est_mask)
                #print(f"est_mask shape after projection[{i - 1}]: {est_mask.shape}")  # [batch, B, est_time]
                est_mask = self.tcn[i](est_mask)
                #print(f"est_mask shape after TCN[{i}]: {est_mask.shape}")  # [batch, B, est_time]

        # Decoder
        #print("\n-- Audio Decoder --")
        est_source = self.a_decoder(mixture_w, est_mask, T_origin)
        #print(f"est_source shape: {est_source.shape}")  # [batch, C, T_origin]

        # Return speaker embeddings and estimated source
        return spks, est_source


class audioEncoder(nn.Module):
    def __init__(self, L, N):
        super(audioEncoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, x):
        #print("=== audioEncoder Forward Pass ===")
        #print(f"Input shape: {x.shape}")  # [batch, time]

        x = torch.unsqueeze(x, 1)
        #print(f"After unsqueeze(1): {x.shape}")  # [batch, 1, time]

        x = F.relu(self.conv1d_U(x))
        #print(f"After conv1d_U and ReLU: {x.shape}")  # [batch, N, new_time]

        return x


class audioDecoder(nn.Module):
    def __init__(self, B, N, L):
        super(audioDecoder, self).__init__()
        self.N, self.L = N, L
        self.mask_conv1x1 = nn.Conv1d(B, N, 1, bias=False)
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask, T_origin):
        #print("=== audioDecoder Forward Pass ===")
        #print(f"mixture_w shape: {mixture_w.shape}")  # [batch, N, K]
        #print(f"est_mask shape: {est_mask.shape}")    # [batch, B, K]
        #print(f"T_origin: {T_origin}")

        est_mask = self.mask_conv1x1(est_mask)
        #print(f"After mask_conv1x1: {est_mask.shape}")  # [batch, N, K]

        x = mixture_w * F.relu(est_mask)
        #print(f"After multiplication (mixture_w * ReLU(est_mask)): {x.shape}")  # [batch, N, K]

        x = torch.transpose(x, 2, 1)  # [batch, K, N]
        #print(f"After transpose(2,1): {x.shape}")  # [batch, K, N]

        x = self.basis_signals(x)  # [batch, K, L]
        #print(f"After basis_signals (Linear): {x.shape}")  # [batch, K, L]

        est_source = overlap_and_add(x, self.L // 2)  # [batch, C, T]
        #print(f"After overlap_and_add: {est_source.shape}")  # [batch, C, T]

        # Adjust the time dimension to match T_origin
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        #print(f"After padding to T_origin: {est_source.shape}")  # [batch, C, T_origin]

        return est_source


class videoEncoder(nn.Module):
    """
    Transforms input from (B, T, 300) -> (B, T, 256) via a linear layer,
    then applies R blocks of VisualConv1D, each keeping the channel dimension = 256.
    """
    def __init__(self, input_dim=300, intermediate_dim=256, R=5):
        super(videoEncoder, self).__init__()

        # 1) Project 300 -> 256
        self.pre_layer = nn.Linear(input_dim, intermediate_dim, bias=False)

        # 2) Stack R VisualConv1D blocks (all operating on 256 channels)
        ve_blocks = []
        for _ in range(R):
            ve_blocks.append(VisualConv1D(intermediate_dim))
        self.net = nn.Sequential(*ve_blocks)

    def forward(self, v):
        """
        Args:
            v: (batch, time, input_dim=300)
        Returns:
            out: (batch, time, 256)
        """
        # Project from 300 -> 256
        v = self.pre_layer(v)                 # (B, T, 256)

        # Transpose for Conv1d: (B, 256, T)
        v = v.transpose(1, 2)

        # Pass through R blocks
        v = self.net(v)                       # still (B, 256, T)

        # Transpose back: (B, T, 256)
        v = v.transpose(1, 2)
        return v

class VisualConv1D(nn.Module):
    def __init__(self, channels=256, hidden_dim=512):
        super(VisualConv1D, self).__init__()

        relu_0 = nn.ReLU()
        norm_0 = GlobalLayerNorm(channels)

        # Expand from channels to hidden_dim.
        conv1x1 = nn.Conv1d(channels, hidden_dim, kernel_size=1, bias=False)

        relu = nn.ReLU()
        norm_1 = GlobalLayerNorm(hidden_dim)

        # Depthwise 1D convolution
        dsconv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3,
            stride=1, padding=1, dilation=1, groups=hidden_dim, bias=False
        )

        prelu = nn.PReLU()
        norm_2 = GlobalLayerNorm(hidden_dim)

        # Project back down to channels (1×1)
        pw_conv = nn.Conv1d(hidden_dim, channels, kernel_size=1, bias=False)

        self.net = nn.Sequential(
            relu_0,
            norm_0,
            conv1x1,
            relu,
            norm_1,
            dsconv,
            prelu,
            norm_2,
            pw_conv
        )

    def forward(self, x):
        # "out" has the same shape as "x"
        out = self.net(x)
        # Add skip connection
        return x + out


class TCN_block(nn.Module):
    def __init__(self, X, P, B, H):
        super(TCN_block, self).__init__()
        tcn_blocks = []
        for x in range(X):
            dilation = 2**x
            padding = (P - 1) * dilation // 2
            tcn_blocks += [TemporalBlock(B, H, P, stride=1,
                                     padding=padding,
                                     dilation=dilation)]
        self.tcn = nn.Sequential(*tcn_blocks)

    def forward(self, x):
        x = self.tcn(x)
        return x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super(TemporalBlock, self).__init__()
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = GlobalLayerNorm(out_channels)
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):

        residual = x
        out = self.net(x)
        return out + residual  # look like w/o F.relu is better than w/ F.relu


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=False)

        prelu = nn.PReLU()
        norm = GlobalLayerNorm(in_channels)
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.net = nn.Sequential(depthwise_conv, prelu, norm,
                                 pointwise_conv)

    def forward(self, x):
        return self.net(x)

class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        x = torch.transpose(x, 1, 2)
        x = super().forward(x)
        x = torch.transpose(x, 1, 2)
        return x


class GlobalLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()#.cuda()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result