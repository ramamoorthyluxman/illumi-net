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
        """
        Initialize with paths to images and light directions in cartesian coordinates.

        :param image_paths: A list of strings, paths to the images.
        :param light_dirs: A list of tuples (x, y, z) representing light directions in cartesian coordinates.
        """
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

        self.height, self.width = self.images[0].shape[:2]

    def _load_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Load images from paths, converting to grayscale if necessary.

        :param image_paths: List of image file paths.
        :return: List of loaded images as numpy arrays.
        """
        images = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Failed to load image: {path}")
            images.append(img.astype(np.float32) / 255.0)  # Normalize to [0, 1]
        return images

    def estimate_normals_and_albedo(self):
        """
        Estimate surface normals and albedo using the photometric stereo approach.

        
        """
        # Photometric Stereo
        self.rps = rps.RPS()
        self.rps.L = np.array(self.light_dirs).T
        img_vectors = [im.reshape((-1, 1)).flatten() for im in self.images]
        self.rps.M = np.array(img_vectors).T
        self.rps.height = self.height
        self.rps.width = self.width
        start = time.time()
        self.rps.solve(self.METHOD)    # Compute
        elapsed_time = time.time() - start
        print("Photometric stereo: elapsed_time:{0}".format(elapsed_time) + "[sec]")

        return self.rps.N

    def save_results(self, output_dir):
        """
        Save the normal map and albedo map as images.

        """
        os.makedirs(output_dir, exist_ok=True)
        self.rps.save_normalmap(filename=os.path.join(output_dir, 'normal_map'))    # Save the estimated normal map
        normal_img = np.reshape(self.rps.N, (self.rps.height * self.rps.width, 3))
        # Save the estimated normal map as image``
        normal_map = self.rps.N
        normal_map = (normal_map - normal_map.min()) / (normal_map.max() - normal_map.min())  # Normalize to [0, 1]
        normal_map = (normal_map * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
        normal_map = np.reshape(normal_map, (self.rps.height, self.rps.width, 3))  # Reshape to (height, width, 3)
        normal_map = cv2.cvtColor(normal_map, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_dir, 'normal_map.png'), normal_map)
        print("Saved normal map to: ", os.path.join(output_dir, 'normal_map.png'))        
        # Reshape albedo to 2D
        albedo_map = self.rps.albedo.reshape((self.height, self.width))
        
        # Normalize albedo to [0, 1] range
        albedo_normalized = (albedo_map - albedo_map.min()) / (albedo_map.max() - albedo_map.min())
        
        # Convert to 8-bit unsigned integer [0, 255]
        albedo_image = (albedo_normalized * 255).astype(np.uint8)
        
        # Save as PNG
        cv2.imwrite(os.path.join(output_dir, 'albedo.png'), albedo_image)
        print("Saved albedo to: ", os.path.join(output_dir, 'albedo.png'))        
        

