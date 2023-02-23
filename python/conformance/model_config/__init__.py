# 12 models for classfication
from .resnet50_config import resnet50_config
from .resnet101_config import resnet101_config
from .mobilenet_v2_config import mobilenet_v2_config
from .efficientnet_config import efficientnet_config
from .seresnet50_config import seresnet50_config
from .densenet_config import densenet_config # redo
from .vgg16_config import vgg16_config
from .repvgg_config import repvgg_config
from .shufflenet_v2_config import shufflenet_v2_config
from .swin_transformer_config import swin_transformer_config
from .vit_config import vit_config
from .inceptionv3_config import inceptionv3_config

# 6 models for segmentation
from .unet_config import unet_config
from .upernet_config import upernet_config
from .fcn_config import fcn_config
from .pspnet_config import pspnet_config
from .deeplabv3_config import deeplabv3_config
from .deeplabv3plus_config import deeplabv3plus_config

# 4 models for detetcion
from .faster_rcnn_r50_config import faster_rcnn_r50_config
from .retinanet_config import retinanet_config
from .ssd300_config import ssd300_config # 8.2G
from .yolov3_config import yolov3_config


__all__ = ['resnet50_config', 'resnet101_config', 'mobilenet_v2_config',
           'efficientnet_config', 'seresnet50_config', 'densenet_config',
           'vgg16_config', 'vgg16_config', 'shufflenet_v2_config',
           'swin_transformer_config', 'vit_config', 'inceptionv3_config',
           'unet_config', 'upernet_config', 'fcn_config',
           'pspnet_config', 'deeplabv3_config', 'deeplabv3plus_config',
           'faster_rcnn_r50_config', 'retinanet_config',
           'ssd300_config', 'yolov3_config']
