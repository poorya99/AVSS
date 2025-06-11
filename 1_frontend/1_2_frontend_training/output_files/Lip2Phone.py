import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from apex import amp

EPS = 1e-8

class avConv(nn.Module):
    def __init__(self, N=256, dim=512):
        super(avConv, self).__init__()
        relu_0 = nn.ReLU()
        norm_0 = GlobalLayerNorm(N)
        conv1x1 = nn.Conv1d(N, dim, 1, bias=False)
        relu = nn.ReLU()
        norm_1 = GlobalLayerNorm(dim)
        dsconv = nn.Conv1d(dim, dim, 3, stride=1, dilation=1, padding=1, groups=dim, bias=False)
        prelu = nn.PReLU()
        norm_2 = GlobalLayerNorm(dim)
        pw_conv = nn.Conv1d(dim, N, 1, bias=False)

        self.net = nn.Sequential(relu_0, norm_0, conv1x1, relu, norm_1, dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out


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
    # Assuming torch>=1.7 which supports LayerNorm over specific dimensions
    @amp.float_function
    def __init__(self, channel_size):
        super(ChannelWiseLayerNorm, self).__init__(channel_size)
        # No additional parameters
    @amp.float_function
    def forward(self, x):
        # x shape: [batch, channels, time]
        # LayerNorm expects last dimension to normalize
        x = x.transpose(1, 2)  # [batch, time, channels]
        x = super(ChannelWiseLayerNorm, self).forward(x)
        x = x.transpose(1, 2)  # [batch, channels, time]
        return x

class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    @amp.float_function
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.zeros(1, channel_size,1 ))  # [1, N, 1]
    @amp.float_function
    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        mean = y.mean(dim=[1, 2], keepdim=True)  # [M, 1, 1]
        var = (y - mean).pow(2).mean(dim=[1, 2], keepdim=True)  # [M, 1, 1]
        gLN_y = self.gamma * (y - mean) / torch.sqrt(var + EPS) + self.beta
        return gLN_y
# ----------------------------
# 1. Define ResNetLayer2D
# ----------------------------

class ResNetLayer1D(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(ResNetLayer1D, self).__init__()
        self.conv1a = nn.Conv1d(inplanes, outplanes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1a = nn.GroupNorm(1, outplanes, eps=1e-8)
        self.conv2a = nn.Conv1d(outplanes, outplanes, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2a = nn.GroupNorm(1, outplanes, eps=1e-8)

        self.conv1b = nn.Conv1d(outplanes, outplanes, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1b = nn.GroupNorm(1, outplanes, eps=1e-8)
        self.conv2b = nn.Conv1d(outplanes, outplanes, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2b = nn.GroupNorm(1, outplanes, eps=1e-8)

        # Downsample layer if needed
        self.stride = stride
        if stride != 1 or inplanes != outplanes:
            self.downsample = nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)
            self.downsample_bn = nn.GroupNorm(1, outplanes, eps=1e-8)
        else:
            self.downsample = None

    def forward(self, inputBatch):
        # print(f"    [ResNetLayer1D] Input: {inputBatch.shape}")

        # First Sub-block
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        # print(f"  ResNetLayer1D: After conv1a, bn1a, and ReLU: {batch.shape}")

        batch = self.conv2a(batch)
        # print(f"  ResNetLayer1D: After conv2a: {batch.shape}")

        # Residual Connection
        if self.downsample is not None:
            residualBatch = self.downsample_bn(self.downsample(inputBatch))
            # print(f"  ResNetLayer1D: After downsample: {residualBatch.shape}")
        else:
            residualBatch = inputBatch
            # print(f"  ResNetLayer1D: Using identity for residual: {residualBatch.shape}")

        batch += residualBatch
        intermediateBatch = batch
        batch = F.relu(self.bn2a(batch))
        # print(f"  ResNetLayer1D: After bn2a and ReLU: {batch.shape}")

        # Second Sub-block
        batch = F.relu(self.bn1b(self.conv1b(batch)))
        # print(f"  ResNetLayer1D: After conv1b, bn1b, and ReLU: {batch.shape}")

        batch = self.conv2b(batch)
        # print(f"  ResNetLayer1D: After conv2b: {batch.shape}")

        # Residual Connection
        batch += intermediateBatch
        outputBatch = F.relu(self.bn2b(batch))
        # print(f"  ResNetLayer1D: After bn2b and ReLU: {outputBatch.shape}")

        return outputBatch




class ResNet1D(nn.Module):
    def __init__(self, layers=[2, 2, 2], inplanes=64, base_width=64):
        super(ResNet1D, self).__init__()
        self.inplanes = inplanes
        self.layer1 = self._make_layer(layers[0], base_width, stride=2)
        self.layer2 = self._make_layer(layers[1], base_width * 2, stride=4)
        self.layer3 = self._make_layer(layers[2], base_width * 4, stride=4)

        # Modify pooling layers to 1D
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avgpool3 = nn.AvgPool1d(kernel_size=2, stride=1)

    def _make_layer(self, blocks, outplanes, stride=1):
        layers = []
        layers.append(ResNetLayer1D(self.inplanes, outplanes, stride=stride))
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(ResNetLayer1D(self.inplanes, outplanes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, inputBatch):
        # print("  ResNet1D Forward Pass:")
        batch = self.layer1(inputBatch)
        # print(f"  ResNet1D: After layer1: {batch.shape}")
        batch = self.avgpool1(batch)
        # print(f"  ResNet1D: After avgpool1: {batch.shape}")

        batch = self.layer2(batch)
        # print(f"  ResNet1D: After layer2: {batch.shape}")
        batch = self.avgpool2(batch)
        # print(f"  ResNet1D: After avgpool2: {batch.shape}")

        batch = self.layer3(batch)
        # print(f"  ResNet1D: After layer3: {batch.shape}")
        batch = self.avgpool3(batch)
        # print(f"  ResNet1D: After avgpool3: {batch.shape}")

        return batch



# ----------------------------
# 3. Define visualNet2D
# ----------------------------

class visualNet2D(nn.Module):
    def __init__(self):
        super(visualNet2D, self).__init__()

        self.frontend1D = nn.Sequential(
            # Input is [batch, 1, 25, 400]
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,7), stride=(1,2), padding=(2, 3), bias=False),
            nn.GroupNorm(1, 16, eps=1e-8),
            nn.ReLU()
            # No MaxPool2d to keep H=25 unchanged
        )

        self.resnet = ResNet1D(layers=[2, 2, 2], inplanes=16, base_width=64)
        # Output from ResNet2D: [batch, 256, 25]

    def forward(self, x):
        batchsize = x.shape[0]
        # print("\n=== visualNet2D Forward Pass ===")
        # print(f"  Input shape: {x.shape}")

        x = self.frontend1D(x)  # [batch,16,25,200]
        # print(f"  After frontend1D: {x.shape}")

        x = x.transpose(1, 2)
        # print(f"  After transpose(1,2): {x.shape}")  # [batch, frames', 64, H', W']

        B, T, C, H = x.shape
        x = x.reshape(B*T, C, H)
        # print(f"  After reshape(B*T, C, H, W): {x.shape}")  # [batch*frames', 64, H', W']


        x = self.resnet(x)       # [batch,256,25]
        # print(f"  After ResNet2D: {x.shape}")

        x = x.mean(dim=2, keepdim=True)
        x = x.squeeze(2)
        # print(f"  After squeeze(3) and squeeze(2): {outputBatch.shape}")  # [batch*frames', 256]

        x = x.reshape(batchsize, -1, 256)
        # print(f"  After reshape to [batch, frames', 256]: {x.shape}")  # [batch, frames', 256]

        x = x.transpose(1,2)
        # print(f"  After transpose(1,2): {x.shape}")  # [batch, 256, frames']


        return x  # [batch,256,25]




# Example definitions for missing classes
# These are simplified versions; replace them with your actual implementations

class tcn(nn.Module):
    def __init__(self, B=256, H=512, P=3, X=4):
        super(tcn, self).__init__()
        blocks = []
        for x in range(X):
            dilation = 2**x
            padding = (P - 1) * dilation // 2
            blocks += [TemporalBlock(B, H, P, stride=1,
                                     padding=padding,
                                     dilation=dilation)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.net(x)
        return out

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
        return out + residual  # Residual connection


# ----------------------------
# 4. Define avNet2D
# ----------------------------

# Modified avNet with print statements
class avNet(nn.Module):
    def __init__(self, N=256):
        super(avNet, self).__init__()
        self.avconv_1 = nn.Conv1d(N, N, kernel_size=1)
        self.tcn_1 = tcn(X=2)

    def forward(self, v):
        # print("avNet Forward Pass:")
        # print(f"  Audio features shape: {a.shape}")  # [batch, N, time_a]
        # print(f"  Visual features shape: {v.shape}")  # [batch, N, time_v]

        # # Ensure audio and visual have the same time dimension
        # if a.size(-1) != v.size(-1):
        #     print("  Padding audio features to match visual features...")
        #     a = F.pad(a, (0, v.size(-1) - a.size(-1)))
        #     print(f"  After padding audio: {a.shape}")

        # # Concatenate along the channel dimension
        # av = torch.cat((a, v), dim=1)
        # print(f"  After concatenation (audio + visual): {av.shape}")  # [batch, 2N, time]

        v = self.avconv_1(v)
        # print(f"  After avconv_1: {v.shape}")  # [batch, N, time]

        v = self.tcn_1(v)
        # print(f"  After tcn_1: {v.shape}")  # [batch, N, time']



        return v

# ----------------------------
# 5. Define slsyn_net2D
# ----------------------------

class Lip2Phone(nn.Module):
    def __init__(self, N=256, output_dim=231):
        super(Lip2Phone, self).__init__()

        # No audioNet since there's no audio input
        # self.audioNet = audioNet()  # Removed

        # visualNet2D processes the 2D feature sequence
        self.visualNet = visualNet2D()

        # avNet2D processes visual features to final output
        self.avNet = avNet()

        # Regression layer to map 256 channels to 231 embedding dimensions
        self.regression = nn.Conv1d(N, output_dim, kernel_size=1)

    def forward(self, x):
        """
        x shape: [batch, 1, 25, 400]
        """
        # print("\n=== slsyn_net2D Forward Pass ===")
        # print(f"Input shape to slsyn_net2D: {x.shape}")

        v_feats = self.visualNet(x)    # [batch,256,25]
        # print(f"Shape after visualNet2D: {v_feats.shape}")

        out = self.avNet(v_feats)      # [batch]
        # print(f"Final output shape: {out.shape}")

        out = self.regression(out)
        # print(f"Final output shape: {out.shape}")
        out = out.transpose(1,2)

        return out

if __name__=="__main__":
  # Instantiate the network
  model = Lip2Phone()

  # Create dummy audio input: [batch_size, time]
  dummy_audio = torch.randn(2, 80000)  # Example: batch_size=2, time=3200

  # Create dummy video input: [batch_size, frames, height, width]
  dummy_video = torch.randn(2, 125, 112, 112)  # Example: batch_size=2, frames=16, H=112, W=112

  # Run the model
  print("\n=== Running the model with dummy inputs ===\n")
  output = model(dummy_audio, dummy_video)
  print("\n=== Final Output ===")
  print(f"Output: {output}")


  # Example usage:
  # Assuming `my_model` is your PyTorch model
  model_size_mb = get_model_size_in_mb(model)
  print(f"Model size: {model_size_mb:.2f} MB")
