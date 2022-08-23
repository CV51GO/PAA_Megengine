## 介绍

本项目是论文《Probabilistic Anchor Assignment with IoU Prediction for Object Detection》的Megengine实现。该论文的官方实现地址：https://github.com/kkhoot/PAA


## 环境安装

依赖于CUDA10

```
conda create -n PAA python=3.7
pip install -r requirements.txt
```

下载官方的权重：https://drive.google.com/file/d/1i8i38lCkItS7H2gYN20Om_OyNJeAupoC/view?usp=sharing
，将下载后的文件置于./official_PAA路径下。 

## 使用方法

安装完环境后，直接运行`python compare.py`。

`compare.py`文件对官方实现和Megengine实现的推理结果进行了对比。

运行`compare.py`时，会读取`./data`中存放的图片进行推理。`compare.py`中实现了Megengine框架和官方使用的Pytorch框架的推理，并判断两者推理结果的一致性。


## 模型加载示例

在paa.py中，定义了```get_megengine_hardnet_model```方法，该方法能够利用hub加载模型。
```python
@hub.pretrained(
"https://studio.brainpp.com/api/v1/activities/3/missions/55/files/044e5766-6db9-4d7c-8f80-b3f30ff8b211"
)
def get_megengine_hardnet_model():
    model_megengine = PAA()
    return model_megengine
```

在使用模型时，使用如下代码即可加载权重：
```python
from paa import get_megengine_hardnet_model
megengine_model = get_megengine_hardnet_model(pretrained=True)
```
