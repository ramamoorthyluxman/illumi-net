import importlib
from utils import params
from utils import dataset
import os
import numpy as np
import train
# import relight

data_ = dataset(params.ACQ_PATHS)

print("Distance matrices shape: ", data_.distance_matrices.shape)
print("Cosine matrices shape: ", data_.cosine_matrices.shape)
print("Surface albedos shape: ", data_.surface_albedos.shape)
print("Surface normals shape: ", data_.surface_normals.shape)
if params.TRAINING:
    print("Images shape: ", data_.images.shape)

if params.TRAINING:
    train = train.train(distances=data_.distance_matrices, 
                        cosines=data_.cosine_matrices, 
                        albedo=data_.surface_albedos, 
                        normals=data_.surface_normals, 
                        targets=data_.images)

else:
    relight = relight.relight(model_path=params.RTI_MODEL_PATH, 
                            distances=data_.distance_matrices, 
                            cosines=data_.cosine_matrices, 
                            albedo=data_.surface_albedos, 
                            normals=data_.surface_normals, 
                            output_dir='/work/imvia/ra7916lu/illumi-net/data/subset/buddhaPNG/reconstructed/')