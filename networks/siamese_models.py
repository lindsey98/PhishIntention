"""OCR-aided Siamese logo-matching network.

A BiT-style ResNet v2 whose pooled features are concatenated with an OCR
embedding before the final fully-connected layer. Exposed through
``KNOWN_MODELS`` (the active model is ``BiT-M-R50x1``).
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from networks.bit_backbone import StdConv2d, PreActBottleneck, tf2th


class ResNetV2(nn.Module):
    """Pre-activation (v2) ResNet with an OCR-embedding fusion head."""

    def __init__(self, block_units, width_factor, head_size=21843, zero_head=False, ocr_emb_size=512):
        super().__init__()
        wf = width_factor
        self.wf = wf
        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.Sequential(OrderedDict([
                ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
                ('pad', nn.ConstantPad2d(1, 0)),
                ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
        ]))

        self.body = nn.Sequential(OrderedDict([
                ('block1', nn.Sequential(OrderedDict(
                        [('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf))] +
                        [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf)) for i in range(2, block_units[0] + 1)],
                ))),
                ('block2', nn.Sequential(OrderedDict(
                        [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2))] +
                        [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf)) for i in range(2, block_units[1] + 1)],
                ))),
                ('block3', nn.Sequential(OrderedDict(
                        [('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2))] +
                        [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf)) for i in range(2, block_units[2] + 1)],
                ))),
                ('block4', nn.Sequential(OrderedDict(
                        [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2))] +
                        [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf)) for i in range(2, block_units[3] + 1)],
                ))),
        ]))
        # pylint: enable=line-too-long

        self.zero_head = zero_head
        self.head = nn.Sequential(OrderedDict([
                ('gn', nn.GroupNorm(32, 2048*wf)),
                ('relu', nn.ReLU(inplace=True)),
                ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
        ]))

        self.additionalfc = nn.Sequential(OrderedDict([
            ('conv_add', nn.Linear(2048*wf+ocr_emb_size, head_size)),
        ]))

    def features(self, x, ocr_emb):
        x = self.head(self.body(self.root(x)))
        x = x.view(-1, 2048*self.wf)
        x = torch.cat((x, ocr_emb), dim=1)
        return x.squeeze(-1).squeeze(-1)

    def forward(self, x, ocr_emb):
        x = self.head(self.body(self.root(x)))
        x = x.view(-1, 2048*self.wf)
        x = torch.cat((x, ocr_emb), dim=1)
        x = self.additionalfc(x)
        return x

    def load_from(self, weights, prefix='resnet/'):
        with torch.no_grad():
            self.root.conv.weight.copy_(tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel']))    # pylint: disable=line-too-long
            self.head.gn.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
            self.head.gn.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
            for bname, block in self.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')


KNOWN_MODELS = OrderedDict([
        ('BiT-M-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
        ('BiT-M-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
        ('BiT-M-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
        ('BiT-M-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
        ('BiT-M-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
        ('BiT-M-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
        ('BiT-S-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
        ('BiT-S-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
        ('BiT-S-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
        ('BiT-S-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
        ('BiT-S-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
        ('BiT-S-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
])
