import importlib
from utils import params
from utils import dataset
import os
import train
import numpy as np

data_ = dataset(params.ACQ_PATHS)

print("Distance matrices shape: ", data_.distance_matrices.shape)
print("Cosine matrices shape: ", data_.cosine_matrices.shape)
print("Surface albedos shape: ", data_.surface_albedos.shape)
print("Surface normals shape: ", data_.surface_normals.shape)
print("Images shape: ", data_.images.shape)

train = train.train(data_.distance_matrices, data_.cosine_matrices, data_.surface_albedos, data_.surface_normals, data_.images)