import megengine.functional as F
import megengine.module as M
import megengine.hub as hub
import megengine
import numpy as np

class FrozenBatchNorm2d(M.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.weight = megengine.Parameter(megengine.tensor(F.ones(n), dtype=np.float32))
        self.bias = megengine.Parameter(megengine.tensor(F.zeros(n), dtype=np.float32))
        self.running_mean = megengine.Parameter(megengine.tensor(F.zeros(n), dtype=np.float32))
        self.running_var = megengine.Parameter(megengine.tensor(F.zeros(n), dtype=np.float32))
    def forward(self, x):
        scale = self.weight * 1/F.sqrt(self.running_var)
        1/F.sqrt(self.running_var)
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

class BaseStem(M.Module):
    def __init__(self):
        super(BaseStem, self).__init__()
        out_channels = 64
        self.conv1 = M.Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = FrozenBatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
    dilation=1,
    dcn_config=None
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
        )
        stride = 1
        in_channels = out_channels
    return M.Sequential(*blocks)

class Bottleneck(M.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func,
        dcn_config
    ):
        super(Bottleneck, self).__init__()
        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = M.Sequential(
                M.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
        if dilation > 1:
            stride = 1 # reset to be 1
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = M.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        self.conv2 = M.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation
        )
        self.bn2 = norm_func(bottleneck_channels)
        self.conv3 = M.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm_func(out_channels)
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out0 = self.conv3(out)
        out = self.bn3(out0)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out

class BottleneckWithFixedBatchNorm(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config=None
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d,
            dcn_config=dcn_config
        )

class ResNet(M.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        stem_module = BaseStem
        stage_specs = (
            dict(index=1, block_count=3, return_features=True),
            dict(index=2, block_count=4, return_features=True),
            dict(index=3, block_count=6, return_features=True),
            dict(index=4, block_count=3, return_features=True),
        )
        transformation_module = BottleneckWithFixedBatchNorm

        # Construct the stem module
        self.stem = stem_module()

        # Constuct the specified ResNet stages
        num_groups = 1
        width_per_group = 64
        in_channels = 64
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = 256
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec['index'])
            stage2_relative_factor = 2 ** (stage_spec['index'] - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec['block_count'],
                num_groups,
                True,
                first_stride=int(stage_spec['index'] > 1) + 1
            )
            in_channels = out_channels
            setattr(self, name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec['return_features']

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs



