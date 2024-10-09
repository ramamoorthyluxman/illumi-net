import importlib
from utils import params
from utils import dataset
import os
import train
import numpy as np

data = dataset(params.ACQ_PATHS)

train = train.train(data.distance_matrices, data.cosine_matrices, data.surface_albedos, data.surface_normals, data.images)