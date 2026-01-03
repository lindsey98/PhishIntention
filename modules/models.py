# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Bottleneck ResNet v2 with GroupNorm and Weight Standardization."""

from collections import OrderedDict  # pylint: disable=g-importing-member

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import StdConv2d, PreActBottleneck, tf2th


class ResNetV2Screenshot(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode.
    Screenshot-only CRP classifier
    """

    def __init__(self, block_units, width_factor, head_size=21843, zero_head=False):
        super().__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        StdConv2d(
                            3, 64 * wf, kernel_size=7, stride=2, padding=3, bias=False
                        ),
                    ),
                    ("pad", nn.ConstantPad2d(1, 0)),
                    ("pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
                    # The following is subtly not the same!
                    # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=64 * wf, cout=256 * wf, cmid=64 * wf
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=256 * wf, cout=256 * wf, cmid=64 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[0] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=256 * wf,
                                            cout=512 * wf,
                                            cmid=128 * wf,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=512 * wf, cout=512 * wf, cmid=128 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[1] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=512 * wf,
                                            cout=1024 * wf,
                                            cmid=256 * wf,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=1024 * wf, cout=1024 * wf, cmid=256 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[2] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block4",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=1024 * wf,
                                            cout=2048 * wf,
                                            cmid=512 * wf,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=2048 * wf, cout=2048 * wf, cmid=512 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[3] + 1)
                                ],
                            )
                        ),
                    ),
                ]
            )
        )
        # pylint: enable=line-too-long

        self.zero_head = zero_head
        self.head = nn.Sequential(
            OrderedDict(
                [
                    ("gn", nn.GroupNorm(32, 2048 * wf)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("avg", nn.AdaptiveAvgPool2d(output_size=1)),
                    ("conv", nn.Conv2d(2048 * wf, head_size, kernel_size=1, bias=True)),
                ]
            )
        )

    def features(self, x):
        x = self.head[:-1](self.body(self.root(x)))

        return x.squeeze()

    def forward(self, x):
        x = self.head(self.body(self.root(x)))
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0]

    def load_from(self, weights, prefix="resnet/"):
        with torch.no_grad():
            self.root.conv.weight.copy_(
                tf2th(weights[f"{prefix}root_block/standardized_conv2d/kernel"])
            )  # pylint: disable=line-too-long
            self.head.gn.weight.copy_(tf2th(weights[f"{prefix}group_norm/gamma"]))
            self.head.gn.bias.copy_(tf2th(weights[f"{prefix}group_norm/beta"]))
            if self.zero_head:
                nn.init.zeros_(self.head.conv.weight)
                nn.init.zeros_(self.head.conv.bias)
            else:
                self.head.conv.weight.copy_(
                    tf2th(weights[f"{prefix}head/conv2d/kernel"])
                )  # pylint: disable=line-too-long
                self.head.conv.bias.copy_(tf2th(weights[f"{prefix}head/conv2d/bias"]))

            for bname, block in self.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, prefix=f"{prefix}{bname}/{uname}/")


class ResNetV2Hybrid(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode.
    Mixed CRP classifier
    """

    def __init__(self, block_units, width_factor, head_size=21843, zero_head=False):
        super().__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        StdConv2d(
                            8, 64 * wf, kernel_size=7, stride=2, padding=3, bias=False
                        ),
                    ),
                    ("pad", nn.ConstantPad2d(1, 0)),
                    ("pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
                    # The following is subtly not the same!
                    # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=64 * wf, cout=256 * wf, cmid=64 * wf
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=256 * wf, cout=256 * wf, cmid=64 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[0] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=256 * wf,
                                            cout=512 * wf,
                                            cmid=128 * wf,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=512 * wf, cout=512 * wf, cmid=128 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[1] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=512 * wf,
                                            cout=1024 * wf,
                                            cmid=256 * wf,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=1024 * wf, cout=1024 * wf, cmid=256 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[2] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block4",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=1024 * wf,
                                            cout=2048 * wf,
                                            cmid=512 * wf,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=2048 * wf, cout=2048 * wf, cmid=512 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[3] + 1)
                                ],
                            )
                        ),
                    ),
                ]
            )
        )
        # pylint: enable=line-too-long

        self.zero_head = zero_head
        self.head = nn.Sequential(
            OrderedDict(
                [
                    ("gn", nn.GroupNorm(32, 2048 * wf)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("avg", nn.AdaptiveAvgPool2d(output_size=1)),
                    ("conv", nn.Conv2d(2048 * wf, head_size, kernel_size=1, bias=True)),
                ]
            )
        )

    def features(self, x):
        x = self.head[:-1](self.body(self.root(x)))

        return x.squeeze()

    def forward(self, x):
        x = self.head(self.body(self.root(x)))
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0]

    def load_from(self, weights, prefix="resnet/"):
        with torch.no_grad():
            self.root.conv.weight.copy_(
                tf2th(weights[f"{prefix}root_block/standardized_conv2d/kernel"])
            )  # pylint: disable=line-too-long
            self.head.gn.weight.copy_(tf2th(weights[f"{prefix}group_norm/gamma"]))
            self.head.gn.bias.copy_(tf2th(weights[f"{prefix}group_norm/beta"]))
            if self.zero_head:
                nn.init.zeros_(self.head.conv.weight)
                nn.init.zeros_(self.head.conv.bias)
            else:
                self.head.conv.weight.copy_(
                    tf2th(weights[f"{prefix}head/conv2d/kernel"])
                )  # pylint: disable=line-too-long
                self.head.conv.bias.copy_(tf2th(weights[f"{prefix}head/conv2d/bias"]))

            for bname, block in self.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, prefix=f"{prefix}{bname}/{uname}/")


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class LayoutClassifier(nn.Module):
    def __init__(self, input_ch_size=9, grid_num=10, head_size=2):
        super(LayoutClassifier, self).__init__()
        self.fc1 = nn.Linear(input_ch_size * grid_num * grid_num, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, head_size)
        self.grid_num = grid_num
        self.input_ch_size = input_ch_size

    def features(self, x):
        x = x.view(-1, self.input_ch_size * self.grid_num * self.grid_num)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def forward(self, x):
        x = x.view(-1, self.input_ch_size * self.grid_num * self.grid_num)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


KNOWN_MODELS = OrderedDict(
    [
        ("BiT-M-R50x1", lambda *a, **kw: ResNetV2Screenshot([3, 4, 6, 3], 1, *a, **kw)),
        ("BiT-M-R50x3", lambda *a, **kw: ResNetV2Screenshot([3, 4, 6, 3], 3, *a, **kw)),
        (
            "BiT-M-R101x1",
            lambda *a, **kw: ResNetV2Screenshot([3, 4, 23, 3], 1, *a, **kw),
        ),
        (
            "BiT-M-R101x3",
            lambda *a, **kw: ResNetV2Screenshot([3, 4, 23, 3], 3, *a, **kw),
        ),
        (
            "BiT-M-R152x2",
            lambda *a, **kw: ResNetV2Screenshot([3, 8, 36, 3], 2, *a, **kw),
        ),
        (
            "BiT-M-R152x4",
            lambda *a, **kw: ResNetV2Screenshot([3, 8, 36, 3], 4, *a, **kw),
        ),
        ("BiT-S-R50x1", lambda *a, **kw: ResNetV2Screenshot([3, 4, 6, 3], 1, *a, **kw)),
        ("BiT-S-R50x3", lambda *a, **kw: ResNetV2Screenshot([3, 4, 6, 3], 3, *a, **kw)),
        (
            "BiT-S-R101x1",
            lambda *a, **kw: ResNetV2Screenshot([3, 4, 23, 3], 1, *a, **kw),
        ),
        (
            "BiT-S-R101x3",
            lambda *a, **kw: ResNetV2Screenshot([3, 4, 23, 3], 3, *a, **kw),
        ),
        (
            "BiT-S-R152x2",
            lambda *a, **kw: ResNetV2Screenshot([3, 8, 36, 3], 2, *a, **kw),
        ),
        (
            "BiT-S-R152x4",
            lambda *a, **kw: ResNetV2Screenshot([3, 8, 36, 3], 4, *a, **kw),
        ),
        ("BiT-M-R50x1V2", lambda *a, **kw: ResNetV2Hybrid([3, 4, 6, 3], 1, *a, **kw)),
        ("FCMax", lambda *a, **kw: LayoutClassifier(*a, **kw)),
    ]
)

if __name__ == "__main__":
    from torchsummary import summary

    model = KNOWN_MODELS["BiT-M-R50x1V3"](head_size=2)
    model.to("cuda:0")
    summary(model, (20, 256, 256))
