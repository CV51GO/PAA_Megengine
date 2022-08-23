
from paa import PAA
import megengine.hub as hub


@hub.pretrained(
"https://studio.brainpp.com/api/v1/activities/3/missions/55/files/044e5766-6db9-4d7c-8f80-b3f30ff8b211"
)
def get_megengine_hardnet_model():
    model_megengine = PAA()
    return model_megengine
