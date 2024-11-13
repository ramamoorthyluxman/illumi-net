import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import json
from utils import params
import train
import importlib
from utils import dataset
import seaborn as sns
import pandas as pd
import GPyOpt
from GPyOpt.methods import BayesianOptimization

class HyperparameterOptimizer:
    def __init__(self, base_save_path='hyperparameter_optimization'):
        self.base_save_path = base_save_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.study_path = os.path.join(base_save_path, f'study_{self.timestamp}')
        os.makedirs(self.study_path, exist_ok=True)
        
        self.best_params = None
        self.param_history = []
        self.mse_history = []
        self.iteration = 0
        
        # Define parameter bounds for Bayesian optimization
        self.bounds = [
            {'name': 'batch_size', 'type': 'discrete', 'domain': (8, 16, 32)},
            {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
            {'name': 'lambda_mse', 'type': 'continuous', 'domain': (0.2, 2.0)},
            {'name': 'lambda_l1', 'type': 'continuous', 'domain': (0, 2.0)},
            {'name': 'lambda_highlight', 'type': 'continuous', 'domain': (0, 2.0)},
            {'name': 'lambda_gradient', 'type': 'continuous', 'domain': (0, 1.0)},
            {'name': 'lambda_specular', 'type': 'continuous', 'domain': (0.2, 2.0)},
            {'name': 'lambda_contrast', 'type': 'continuous', 'domain': (0, 1.0)},
            {'name': 'lambda_perceptual', 'type': 'continuous', 'domain': (0, 1.0)},
            {'name': 'lambda_dark', 'type': 'continuous', 'domain': (0.0, 1.0)},
            {'name': 'lambda_mid', 'type': 'continuous', 'domain': (0.0, 1.0)},
            {'name': 'lambda_bright', 'type': 'continuous', 'domain': (0.0, 1.0)},
            {'name': 'lambda_large_dev', 'type': 'continuous', 'domain': (0.0, 1.0)},
            {'name': 'weight_decay', 'type': 'continuous', 'domain': (1e-4, 1e-2)}
        ]
        print(f"Initialized optimizer with {len(self.bounds)} parameters to optimize")

    def update_params(self, x):
        """Update params with new hyperparameters"""
        param_dict = {
            'batch_size': int(x[0]),
            'learning_rate': float(x[1]),
            'lambda_mse': float(x[2]),
            'lambda_l1': float(x[3]),
            'lambda_highlight': float(x[4]),
            'lambda_gradient': float(x[5]),
            'lambda_specular': float(x[6]),
            'lambda_contrast': float(x[7]),
            'lambda_perceptual': float(x[8]),
            'lambda_dark': float(x[9]),
            'lambda_mid': float(x[10]),
            'lambda_bright': float(x[11]),
            'lambda_large_dev': float(x[12]),
            'weight_decay': float(x[13])
        }
        
        print("\nTrying parameters:", param_dict)
        
        # Update params module
        for param, value in param_dict.items():
            setattr(params, param.upper(), value)
        
        return param_dict

    def compute_mse(self, model, data_loader, device):
        """Compute MSE between target and output images"""
        model.eval()
        total_mse = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                distances = batch['distances'].to(device)
                cosines = batch['cosines'].to(device)
                albedo = batch['albedo'].to(device)
                normals = batch['normals'].to(device)
                azimuth = batch['azimuth'].to(device)
                targets = batch['target'].to(device)

                outputs, _ = model(distances, cosines, albedo, normals, azimuth)
                
                # Compute MSE for this batch
                mse = torch.nn.functional.mse_loss(outputs, targets)
                total_mse += mse.item() * targets.size(0)
                num_samples += targets.size(0)

        return total_mse / num_samples

    def objective_function(self, x):
        """Objective function for Bayesian optimization"""
        self.iteration += 1
        print(f"\nStarting iteration {self.iteration}")
        
        try:
            # Update parameters
            param_dict = self.update_params(x[0])
            print("\nTrying parameters:", param_dict)
            
            # Load dataset
            print("Loading dataset...")
            data_ = dataset(params.ACQ_PATHS)

            print("distances shape: ", data_.distance_matrices.shape)
            print("cosine shape: ", data_.cosine_matrices.shape)
            print("surface albedos shape: ", data_.surface_albedos.shape)
            print("surface_normals shape: ", data_.surface_normals.shape)
            print("azimuths shape: ", data_.azimuths.shape)
            print("images shape: ", data_.images.shape)

            # Create data loaders
            train_loader, val_loader, _, _ = train.prepare_data(
                distances=data_.distance_matrices,
                cosines=data_.cosine_matrices,
                albedo=data_.surface_albedos,
                normals=data_.surface_normals,
                azimuths=data_.azimuths,  
                targets=data_.images
            ) 

            
            # Initialize model
            print("Initializing model...")
            model = train.RelightingModel(albedo_channels=data_.surface_albedos.shape[-1])
            
            

            # Train model
            print("Training model...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # Reduce epochs for optimization
            original_epochs = params.RTI_NET_EPOCHS
            params.RTI_NET_EPOCHS = 20  # Use fewer epochs during optimization

            # Train model
            model = train.train_model(model, train_loader, val_loader, num_epochs=params.RTI_NET_EPOCHS)
            
            # Compute MSE on validation set
            print("Computing MSE...")
            mse = self.compute_mse(model, val_loader, device)
            print(f"MSE: {mse}")
            
            # Store results
            self.param_history.append(param_dict)
            self.mse_history.append(mse)
            
            # Save intermediate results
            self.save_results(intermediate=True)
            
            # Create visualization
            self.visualize_optimization()
            
            # Restore original epochs
            params.RTI_NET_EPOCHS = original_epochs
            
            return mse
            
        except Exception as e:
            print(f"Error in iteration {self.iteration}: {str(e)}")
            return float('inf')

    def visualize_optimization(self):
        """Create visualizations of the optimization process"""
        print("\nCreating visualizations...")
        viz_path = os.path.join(self.study_path, 'visualizations')
        os.makedirs(viz_path, exist_ok=True)
        
        # 1. MSE Evolution Plot
        plt.figure(figsize=(10, 6))
        iterations = range(1, len(self.mse_history) + 1)
        plt.plot(iterations, self.mse_history, 'bo-')
        plt.title('MSE Evolution')
        plt.xlabel('Trial')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.savefig(os.path.join(viz_path, f'mse_evolution_{self.iteration}.png'))
        plt.close()
        
        if len(self.param_history) > 1:
            # Convert history to DataFrame
            history_df = pd.DataFrame(self.param_history)
            history_df['mse'] = self.mse_history
            
            # 2. Parameter Correlation Heatmap
            plt.figure(figsize=(12, 10))
            correlation = history_df.corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Parameter Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_path, f'parameter_correlation_{self.iteration}.png'))
            plt.close()
            
            # 3. Parameter Importance Plot
            plt.figure(figsize=(12, 6))
            importance = correlation['mse']
            importance.plot(kind='bar')
            plt.title('Parameter Importance (Based on Correlation with MSE)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_path, f'parameter_importance_{self.iteration}.png'))
            plt.close()

    def save_results(self, intermediate=False):
        """Save optimization results"""
        suffix = 'intermediate' if intermediate else 'final'
        results = {
            'iteration': self.iteration,
            'best_params': self.best_params,
            'param_history': self.param_history,
            'mse_history': [float(mse) for mse in self.mse_history]
        }
        
        save_path = os.path.join(self.study_path, f'optimization_results_{suffix}.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Saved results to {save_path}")

    def optimize(self, n_trials=100):
        """Run the optimization process"""
        print(f"Starting optimization with {n_trials} trials...")
        
        optimizer = BayesianOptimization(
            f=self.objective_function,
            domain=self.bounds,
            model_type='GP',
            acquisition_type='EI',
            maximize=False,
            initial_design_numdata=45
        )
        
        optimizer.run_optimization(max_iter=n_trials)
        
        # Get best parameters
        best_x = optimizer.x_opt
        self.best_params = self.update_params(best_x)
        
        # Final save and visualization
        self.save_results(intermediate=False)
        self.visualize_optimization()
        
        print("\nOptimization completed!")
        print("Best parameters found:", self.best_params)
        print(f"Best MSE: {min(self.mse_history)}")
        
        return self.best_params
