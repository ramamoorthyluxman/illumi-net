import numpy as np
import os
import glob
import math
from .compute_normals import PhotometricStereo_traditional
from . import params
import cv2
import random
from tqdm import tqdm

# Convert Cartesian coordinates to spherical coordinates    
def Cartesian2spherical3D(x, y, z):
    """
    Takes X, Y, and Z coordinates as input and converts them to a spherical
    coordinate system

    Source: https://stackoverflow.com/questions/10868135/cartesian-to-spherical-3d-coordinates

    """

    r = math.sqrt(x*x + y*y + z*z)

    longitude = math.acos(x / math.sqrt(x*x + y*y)) * (-1 if y < 0 else 1)

    latitude = math.acos(z / r)

    return r, longitude, latitude

# Convert spherical coordinates to Cartesian coordinates
def spherical2Cartesian3D(r, longitude, latitude):
    """
    Takes, r, longitude, and latitude coordinates in a spherical coordinate
    system and converts them to a 3D cartesian coordinate system

    Source: https://stackoverflow.com/questions/10868135/cartesian-to-spherical-3d-coordinates
    """

    x = r * math.sin(latitude) * math.cos(longitude)
    y = r * math.sin(latitude) * math.sin(longitude)
    z = r * math.cos(latitude)

    return x, y, z

class dataset:
    def __init__(self, acq_paths):
        self.acqs = acq_paths
        self.lp_files = []
        self.lps_cartesian = []
        self.lps_spherical = []
        self.image_paths = []
        self.surface_normals = []
        self.surface_albedos = []
        self.distance_matrices = []
        self.cosine_matrices = []
        self.images = []

        self.rand_indices = []
        for i in range(len(self.acqs)):
            self.rand_indices.append(random.sample(range(len(self.acqs[i])), min(len(self.acqs[i]), params.MAX_NB_IMAGES_PER_ACQ)))

        self.lp_files, errors = ([path for folder in self.acqs for path in glob.glob(os.path.join(folder, "*.lp"))], [f"Error: No .lp file found in {folder}" for folder in self.acqs if not glob.glob(os.path.join(folder, "*.lp"))])
        print(errors)

        self.load_lps_and_img_paths()

        if params.COMPUTE_NORMALS_AND_ALBEDO:
            self.computed_normals_and_albedos()

        if params.COMPUTE_DISTANCES_AND_COSINES:
            self.compute_distances_and_cosines()

        print("Loading distance matrices and cosine matrices...")
        self.load_distance_matrices()
        self.load_cosine_matrices()
        print("loading surface normals and albedos...")
        self.load_surface_normals()
        self.load_surface_albedos()
        print("Loading images...")
        self.load_images()
    
    def load_images(self):
        for i in tqdm(range(len(self.acqs)), desc="Loading acquisitions", unit="acq"):
            acq_images = []
            for idx in tqdm(self.rand_indices[i],
                            desc=f"Loading images for acq {i+1}/{len(self.acqs)}",
                            leave=False, 
                            unit="img",
                            total=len(self.rand_indices)):
                img_path = self.image_paths[i][idx]
                img = cv2.imread(img_path)
                acq_images.append(img)
            self.images.append(acq_images)

        self.images = np.array(self.images)
    
    def load_lps_and_img_paths(self):
        print("Loading LP files and image paths...")
        for lp_file in self.lp_files:
            with open(lp_file, 'r') as f:
                lines = f.readlines()
                lps_cartesian = []
                lps_spherical = []
                img_paths = []
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) != 4:
                        print(f"Line in {lp_file} does not contain exactly four parts: {line}")
                        return

                    img_file, x, y, z = parts
                    x = float(x)
                    y = float(y)
                    z = float(z)
                    lps_cartesian.append((x, y, z))
                    lps_spherical.append(Cartesian2spherical3D(x, y, z))
                    img_paths.append(os.path.join(os.path.dirname(lp_file), img_file))
            self.lps_cartesian.append(lps_cartesian)
            self.lps_spherical.append(lps_spherical) 
            self.image_paths.append(img_paths)                      


    def computed_normals_and_albedos(self):
        print("Computing normals and albedos...")
        for i in range(len(self.acqs)):
            # Create PhotometricStereo instance
            ps = PhotometricStereo_traditional(self.image_paths[i], self.lps_cartesian[i])
            # Compute normals
            self.normals = ps.estimate_normals_and_albedo()
            print("Normals computed successfully.")
            # Get the directory of the first image
            img_dir = os.path.dirname(self.image_paths[i][0])
            # Save normal map image in the image directory
            print("Saving normal map to " + img_dir)
            ps.save_results(img_dir)
            print("Finished computing and saving normal maps.")

    def compute_distances_and_cosines(self):
        print("Computing distances and cosines...")
        for i in range(len(self.acqs)):
            img_shape = cv2.imread(self.image_paths[i][0]).shape
            distance_matrices = []
            angles_matrices = []
            for j in range(len(self.image_paths[i])):
                center_distance = np.linalg.norm(np.array(self.lps_cartesian[i][j]))
                h, w, _ = img_shape
                surface_width, surface_height = params.SURFACE_PHYSCIAL_SIZE
                x = np.linspace(-surface_width / 2, surface_width / 2, w)
                y = np.linspace(-surface_height / 2, surface_height / 2, h)
                X, Y = np.meshgrid(y, x)
                Z = np.zeros_like(X)        
                distances = np.sqrt((X - self.lps_cartesian[i][j][0])**2 + (Y - self.lps_cartesian[i][j][1])**2 + (Z - self.lps_cartesian[i][j][2])**2)
                # Ensure that the angle calculation is safe for arccos
                cos_theta = np.clip(-self.lps_cartesian[i][j][2] / distances, -1.0, 1.0)
                angles = np.arccos(cos_theta)
                distance_matrices.append(distances)
                angles_matrices.append(angles)
        
            np.save(os.path.join(os.path.dirname(self.image_paths[i][0]), "distance_matrices.npy"), np.array(distance_matrices))
            np.save(os.path.join(os.path.dirname(self.image_paths[i][0]), "angles_matrices.npy"), np.array(angles_matrices))


    def load_surface_normals(self):
        for acq in self.acqs:
            self.surface_normals.append(np.load(os.path.join(acq, "normal_map.npy")))
        self.surface_normals = np.array(self.surface_normals)
                                                
    def load_surface_albedos(self):
        for acq in self.acqs:
            self.surface_albedos.append(cv2.imread(os.path.join(acq, "albedo.png"), cv2.IMREAD_GRAYSCALE))
        self.surface_albedos = np.array(self.surface_albedos)
                                                
    def load_distance_matrices(self):
        if not params.SAME_ILLUMINATION:
            for idx, acq in enumerate(self.acqs):
                distance_matrix = np.load(os.path.join(acq, "distance_matrices.npy"))
                self.distance_matrices.append(distance_matrix[self.rand_indices[idx], :, :])                
        else:
            self.distance_matrices = np.load(os.path.join(self.acqs[0], "distance_matrices.npy"))[self.rand_indices]

        self.distance_matrices = np.array(self.distance_matrices)
                                                    
    def load_cosine_matrices(self):
        if not params.SAME_ILLUMINATION:
            for idx, acq in enumerate(self.acqs):
                cosine_matrix = np.load(os.path.join(acq, "angles_matrices.npy"))
                self.cosine_matrices.append(cosine_matrix[self.rand_indices[idx], :, :])
        else:
            self.cosine_matrices = np.cos(np.load(os.path.join(self.acqs[0], "angles_matrices.npy")))[self.rand_indices]