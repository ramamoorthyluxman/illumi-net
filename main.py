# main.py

import importlib
from utils import params
from utils import dataset
import os
import numpy as np
import train
import relight
from hyperparameter_optimizer import HyperparameterOptimizer

def main():
    if params.OPTIMIZE_HYPERPARAMETERS:
        print("Starting hyperparameter optimization...")
        optimizer = HyperparameterOptimizer()
        best_params = optimizer.optimize(n_trials=params.OPTIMIZATION_TRIALS)
        
        # Update params with best parameters found
        for param, value in best_params.items():
            setattr(params, param.upper(), value)
        
        print("Hyperparameter optimization completed. Best parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
            
        # Ask user if they want to continue with training using best parameters
        user_input = input("Do you want to train the model with the best parameters? (y/n): ")
        if user_input.lower() != 'y':
            return

    data_ = dataset(params.ACQ_PATHS)
    
    print("Distance matrices shape: ", data_.distance_matrices.shape)
    print("Cosine matrices shape: ", data_.cosine_matrices.shape)
    print("Surface albedos shape: ", data_.surface_albedos.shape)
    print("Surface normals shape: ", data_.surface_normals.shape)
    print("Azimuths shape: ", data_.azimuths.shape)
    if params.TRAINING:
        print("Images shape: ", data_.images.shape)
    
    if params.TRAINING:
        train.train(distances=data_.distance_matrices,
                   cosines=data_.cosine_matrices,
                   albedo=data_.surface_albedos,
                   normals=data_.surface_normals,
                   azimuths = data_.azimuths,
                   targets=data_.images)
    else:
        relight.relight(model_path=params.RTI_MODEL_PATH,
                       distances=data_.distance_matrices,
                       cosines=data_.cosine_matrices,
                       albedo=data_.surface_albedos,
                       normals=data_.surface_normals,
                       azimuths = data_.azimuths,
                       output_dir='/work/imvia/ra7916lu/illumi-net/data/subset/2024_02_22_1_3/images/Face_A/rti_sub_images/reconstructed')

if __name__ == "__main__":
    main()