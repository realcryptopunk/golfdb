"""
CBAM-SwingNet: Attention-enhanced Golf Swing Event Detector
Achieves 80.5% PCE vs 76.1% for original SwingNet
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import math


# ============== CBAM Attention Blocks ==============

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# ============== MobileNetV2 Backbone ==============

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ============== CBAM-SwingNet Model ==============

class CBAMEventDetector(nn.Module):
    """
    CBAM-SwingNet: Golf Swing Event Detector with Attention

    Layer structure matches checkpoint naming:
    - cnn0: features[0] (conv_bn)
    - cnn1: features[1] (first inverted residual)
    - cnn2: features[2:4] -> cnn2.2, cnn2.3
    - cnn3: features[4:7] -> cnn3.4, cnn3.5, cnn3.6
    - cnn4: features[7:11] -> cnn4.7, cnn4.8, cnn4.9, cnn4.10
    - cnn5: features[11:14] -> cnn5.11, cnn5.12, cnn5.13
    - cnn6: features[14:17] -> cnn6.14, cnn6.15, cnn6.16
    - cnn7: features[17] (InvertedResidual)
    - cnn8: features[18] (conv_1x1_bn)
    """

    def __init__(self, pretrain_path=None, width_mult=1., lstm_layers=1, lstm_hidden=256,
                 bidirectional=True, dropout=True):
        super(CBAMEventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load MobileNetV2 backbone
        net = MobileNetV2(width_mult=width_mult)

        # Load pretrained weights if provided
        if pretrain_path:
            state_dict = torch.load(pretrain_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            net.load_state_dict(state_dict, strict=False)

        # Split CNN into segments with naming that matches checkpoint
        self.cnn0 = list(net.children())[0][0]      # features[0]
        self.cnn1 = list(net.children())[0][1]      # features[1]

        # Use nn.ModuleDict with string keys matching checkpoint naming
        self.cnn2 = nn.ModuleDict({
            '2': list(net.children())[0][2],
            '3': list(net.children())[0][3]
        })
        self.cnn3 = nn.ModuleDict({
            '4': list(net.children())[0][4],
            '5': list(net.children())[0][5],
            '6': list(net.children())[0][6]
        })
        self.cnn4 = nn.ModuleDict({
            '7': list(net.children())[0][7],
            '8': list(net.children())[0][8],
            '9': list(net.children())[0][9],
            '10': list(net.children())[0][10]
        })
        self.cnn5 = nn.ModuleDict({
            '11': list(net.children())[0][11],
            '12': list(net.children())[0][12],
            '13': list(net.children())[0][13]
        })
        self.cnn6 = nn.ModuleDict({
            '14': list(net.children())[0][14],
            '15': list(net.children())[0][15],
            '16': list(net.children())[0][16]
        })
        self.cnn7 = list(net.children())[0][17]     # features[17]
        self.cnn8 = list(net.children())[0][18]     # features[18]

        # CBAM attention modules at key points
        self.atn0 = CBAM(32)    # After first conv
        self.atn7 = CBAM(320)   # After layer 7
        self.atn8 = CBAM(1280)  # After final conv

        # LSTM for temporal modeling
        self.rnn = nn.LSTM(
            int(1280 * width_mult if width_mult > 1.0 else 1280),
            self.lstm_hidden,
            self.lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Output layer (9 classes: 8 events + no-event)
        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden, 9)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 9)

        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (
                Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden).to(self.device), requires_grad=True),
                Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden).to(self.device), requires_grad=True)
            )
        else:
            return (
                Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(self.device), requires_grad=True),
                Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(self.device), requires_grad=True)
            )

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        # CNN forward with CBAM attention
        c_in = x.view(batch_size * timesteps, C, H, W)

        c_out = self.cnn0(c_in)
        c_out = self.atn0(c_out)  # CBAM after first conv
        c_out = self.cnn1(c_out)

        # Forward through ModuleDict layers in order
        for key in ['2', '3']:
            c_out = self.cnn2[key](c_out)
        for key in ['4', '5', '6']:
            c_out = self.cnn3[key](c_out)
        for key in ['7', '8', '9', '10']:
            c_out = self.cnn4[key](c_out)
        for key in ['11', '12', '13']:
            c_out = self.cnn5[key](c_out)
        for key in ['14', '15', '16']:
            c_out = self.cnn6[key](c_out)

        c_out = self.cnn7(c_out)
        c_out = self.atn7(c_out)  # CBAM after layer 7
        c_out = self.cnn8(c_out)
        c_out = self.atn8(c_out)  # CBAM after final conv

        c_out = c_out.mean(3).mean(2)
        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in, self.hidden)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, 9)

        return out


def load_cbam_swingnet(weights_path, mobilenet_path=None):
    """
    Load CBAM-SwingNet model with pretrained weights.

    Args:
        weights_path: Path to CBAM-SwingNet weights (cbam_swingnet.pth.tar)
        mobilenet_path: Path to MobileNetV2 weights (optional, for backbone init)

    Returns:
        Loaded model in eval mode
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = CBAMEventDetector(
        pretrain_path=mobilenet_path,
        width_mult=1.,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False
    )

    # Load trained weights
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model
