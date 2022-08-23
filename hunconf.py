
from paa import PAA as modelPAA
import megengine.hub as hub


@hub.pretrained(
"https://studio.brainpp.com/api/v1/activities/3/missions/55/files/044e5766-6db9-4d7c-8f80-b3f30ff8b211"
)
def PAA(pretrained=False):
    model_megengine = modelPAA()
    return model_megengine
