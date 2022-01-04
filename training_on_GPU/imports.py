import math
import numpy as np
import sys

from PIL import Image
#import requests
import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'

#import ipywidgets as widgets
#from IPython.display import display, clear_output

import torch
#print("torch.__version__", torch.__version__)
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);

import os
#import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO


#%matplotlib inline
#import pycocotools.coco as coco

import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
import torchvision

import json
from collections import OrderedDict

from transformers import DetrFeatureExtractor, DetrConfig, DetrForObjectDetection, DetrModel, DetrPreTrainedModel

import pytorch_lightning as pl
from pytorch_lightning import Trainer


sys.path.insert(0,"/zhome/91/a/164752/02456/venv/content/detr/")
print(sys.path)
from datasets import get_coco_api_from_dataset
from datasets.coco_eval import CocoEvaluator
