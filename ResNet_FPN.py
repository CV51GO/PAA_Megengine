
from collections import OrderedDict
import megengine.functional as F
import megengine.module as M
from backbone import ResNet

class FPN(M.Module):
    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            setattr(self, inner_block, inner_block_module)
            setattr(self, layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_lateral = getattr(self, inner_block)(feature)
            inner_top_down = F.vision.interpolate(last_inner, 
                                                  size=(int(inner_lateral.shape[-2]), int(inner_lateral.shape[-1])), 
                                                  mode='nearest')


            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))
        last_results = self.top_blocks(x[-1], results[-1])
        results.extend(last_results)
        return tuple(results)


class LastLevelP6P7(M.Module):
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = M.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = M.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]

def conv_with_kaiming_uniform():
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = M.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=dilation * (kernel_size - 1) // 2, 
            dilation=dilation, 
            bias=True
        )
        module = [conv,]
        if len(module) > 1:
            return M.Sequential(*module)
        return conv
    return make_conv

def build_resnet_fpn_p3p7_backbone():
    body = ResNet()
    in_channels_stage2 = 256 
    out_channels = 256
    in_channels_p6p7 = 256
    fpn = FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(),
        top_blocks=LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = M.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model