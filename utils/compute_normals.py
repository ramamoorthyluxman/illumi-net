import cv2
import numpy as np
import time
from .RobustPhotometricStereo import rps
import psutil
from typing import List, Tuple
import os
from utils import params

class PhotometricStereo_traditional:
    def __init__(self, image_paths: List[str], light_dirs: List[Tuple[float, float, float]]):
        if len(image_paths) != len(light_dirs):
            raise ValueError("Number of images must match number of light directions")

        self.light_dirs = np.array(light_dirs)
        self.images = self._load_images(image_paths)

        self.METHOD = None
        self.rps = None

        if params.PS_METHOD == "L2_SOLVER":
            self.METHOD = rps.RPS.L2_SOLVER    
        elif params.PS_METHOD == "L1_SOLVER_MULTICORE":
            self.METHOD = rps.RPS.L1_SOLVER_MULTICORE   
        elif params.PS_METHOD == "SBL_SOLVER_MULTICORE":
            self.METHOD = rps.RPS.SBL_SOLVER_MULTICORE
        elif params.PS_METHOD == "RPCA_SOLVER":
            self.METHOD = rps.RPS.RPCA_SOLVER

        if len(self.images) < 3:
            raise ValueError("At least 3 images are required for photometric stereo")

        self.height, self.width, self.channels = self.images[0].shape

    def _load_images(self, image_paths: List[str]) -> List[np.ndarray]:
        images = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Failed to load image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            images.append(img.astype(np.float32) / 255.0)  # Normalize to [0, 1]
        return images

    def estimate_normals_and_albedo(self):
        normal_maps = []
        albedo_maps = []

        for channel in range(3):  # Process each RGB channel separately
            self.rps = rps.RPS()
            self.rps.L = np.array(self.light_dirs).T
            img_vectors = [im[:,:,channel].reshape((-1, 1)).flatten() for im in self.images]
            self.rps.M = np.array(img_vectors).T
            self.rps.height = self.height
            self.rps.width = self.width
            
            start = time.time()
            self.rps.solve(self.METHOD)
            elapsed_time = time.time() - start
            print(f"Photometric stereo for channel {channel}: elapsed_time:{elapsed_time:.2f}[sec]")

            normal_maps.append(self.rps.N)
            albedo_maps.append(self.rps.albedo)

        # Combine the results
        self.normal_map = np.mean(normal_maps, axis=0)
        # Reshape normal_map to (height, width, 3)
        self.normal_map = self.normal_map.reshape(self.height, self.width, 3)
        # Normalize each normal vector
        norm = np.linalg.norm(self.normal_map, axis=2, keepdims=True)
        self.normal_map = np.where(norm != 0, self.normal_map / norm, 0)

        self.albedo_map = np.stack(albedo_maps, axis=-1)
        # Reshape albedo_map to (height, width, 3)
        self.albedo_map = self.albedo_map.reshape(self.height, self.width, 3)

        return self.normal_map

    def save_results(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # Save normal map
        normal_map = (self.normal_map + 1) / 2  # Convert from [-1, 1] to [0, 1]
        normal_map = (normal_map * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'normal_map.png'), cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR))
        print("Saved normal map to:", os.path.join(output_dir, 'normal_map.png'))

        # Save albedo map
        albedo_normalized = (self.albedo_map - self.albedo_map.min()) / (self.albedo_map.max() - self.albedo_map.min())
        albedo_image = (albedo_normalized * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'albedo.png'), cv2.cvtColor(albedo_image, cv2.COLOR_RGB2BGR))
        print("Saved albedo to:", os.path.join(output_dir, 'albedo.png'))