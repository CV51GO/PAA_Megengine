from collections import OrderedDict
import numpy as np
import math

import megengine
import megengine.functional as F
import megengine.module as M
import megengine.hub as hub

from ResNet_FPN import build_resnet_fpn_p3p7_backbone
from rpn import PAAModule

class PAA(M.Module):
    def __init__(self):
        super(PAA, self).__init__()

        self.backbone = build_resnet_fpn_p3p7_backbone()
        self.rpn = PAAModule(self.backbone.out_channels)
    def forward(self, images, targets=None):
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        result = proposals
        return result

@hub.pretrained(
"https://studio.brainpp.com/api/v1/activities/3/missions/55/files/044e5766-6db9-4d7c-8f80-b3f30ff8b211"
)
def get_megengine_hardnet_model():
    model_megengine = PAA()
    return model_megengine

