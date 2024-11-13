import numpy as np
import os
import glob
import math
from .compute_normals import PhotometricStereo_traditional
from . import params
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


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
        self.azimuths = []
        self.elevations = []
        self.image_paths = []
        self.surface_normals = []
        self.surface_albedos = []
        self.distance_matrices = []
        self.cosine_matrices = []
        self.images = []

        self.rand_indices = []
        
        self.lp_files, errors = ([path for folder in self.acqs for path in glob.glob(os.path.join(folder, "*.lp"))], [f"Error: No .lp file found in {folder}" for folder in self.acqs if not glob.glob(os.path.join(folder, "*.lp"))])
        print(errors)

        # If training you pick randomly the desired number of samples. For relight, you consider whatever distances and cosines you pass. 
        if params.TRAINING:
            self.load_lps_and_img_paths()

        # If training you pick randomly the desired number of samples. For relight, you consider whatever distances and cosines you pass. 
        if params.TRAINING:
            for i in range(len(self.acqs)):
                self.rand_indices.append(random.sample(range(len(self.lps_cartesian[i])), min(len(self.lps_cartesian[i]), params.MAX_NB_IMAGES_PER_ACQ)))

        if params.COMPUTE_NORMALS_AND_ALBEDO and params.TRAINING:
            self.computed_normals_and_albedos()

        if params.COMPUTE_DISTANCES_AND_COSINES and params.TRAINING:
            self.compute_distances_and_cosines()
            

        print("Loading distance matrices and cosine matrices...")
        self.load_distance_matrices()
        self.load_cosine_matrices()
        print("loading surface normals and albedos...")
        self.load_surface_normals()
        self.load_surface_albedos()
        # You load the target images only for training. For relighting - you create new images
        if params.TRAINING:
            print("Loading images...")
            self.load_images()

        if params.CREATE_DIST_COSINES_HEATMAPS:
            print("Generating distance heatmaps..")
            self.create_and_save_heatmaps()
        
    
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
                azimuths = []
                elevations = []
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
                    _, azimuth, elevation = Cartesian2spherical3D(x, y, z)
                    azimuths.append(azimuth)
                    elevations.append(elevation)
                    img_paths.append(os.path.join(os.path.dirname(lp_file), img_file))
            self.azimuths.append(azimuths)
            self.elevations.append(elevations)
            self.lps_cartesian.append(lps_cartesian)
            self.lps_spherical.append(lps_spherical) 
            self.image_paths.append(img_paths) 
        self.azimuths = np.array(self.azimuths)


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
        print(f"Light type: {'Collimated' if params.COLLIMATED_LIGHT else 'Point source'}")
        
        for i in range(len(self.acqs)):
            img_shape = cv2.imread(self.image_paths[i][0]).shape
            distance_matrices = []
            angles_matrices = []
            
            for j in range(len(self.image_paths[i])):
                # Calculate center distance (used for both collimated and point source)
                center_distance = np.linalg.norm(np.array(self.lps_cartesian[i][j]))
                
                h, w, _ = img_shape
                surface_width, surface_height = params.SURFACE_PHYSCIAL_SIZE
                x = np.linspace(-surface_width / 2, surface_width / 2, w)
                y = np.linspace(surface_height / 2, -surface_height / 2, h)
                X, Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)
                
                if params.COLLIMATED_LIGHT:
                    # For collimated light, use constant distance for all points
                    distances = np.full_like(X, center_distance**2)
                else:
                    # For point source, calculate distance for each point
                    distances = np.sqrt((X - self.lps_cartesian[i][j][0])**2 + 
                                    (Y - self.lps_cartesian[i][j][1])**2 + 
                                    (Z - self.lps_cartesian[i][j][2])**2)
                    distances = distances**2
                
                # Angle calculation remains the same for both cases
                # For collimated light, this will give constant angles across the surface
                cos_theta = np.clip(-self.lps_cartesian[i][j][2] / np.sqrt(distances), -1.0, 1.0)
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
            albedo = cv2.imread(os.path.join(acq, "albedo.png"), cv2.IMREAD_UNCHANGED)
            if len(albedo.shape) == 2:
                # For grayscale, reshape to (H, W, 1)
                albedo = albedo[..., np.newaxis]
            self.surface_albedos.append(albedo)
        self.surface_albedos = np.array(self.surface_albedos)

    def load_distance_matrices(self):
        for i in tqdm(range(len(self.acqs)), desc="Loading acquisitions", unit="acq"):
            acq_distance_matrices = np.load(os.path.join(self.acqs[i], "distance_matrices.npy"))
            if len(self.rand_indices)==0:
                for i in range(len(self.acqs)):
                    self.rand_indices.append(random.sample(range(acq_distance_matrices.shape[0]), min(acq_distance_matrices.shape[0], params.MAX_NB_IMAGES_PER_ACQ)))
            self.distance_matrices.append(acq_distance_matrices[self.rand_indices[i]])
        self.distance_matrices = np.array(self.distance_matrices)
            
    def load_cosine_matrices(self):
        for i in tqdm(range(len(self.acqs)), desc="Loading acquisitions", unit="acq"):

            acq_cosine_matrices = np.load(os.path.join(self.acqs[i], "angles_matrices.npy"))
            self.cosine_matrices.append(acq_cosine_matrices[self.rand_indices[i]])
        self.cosine_matrices = np.array(self.cosine_matrices)

    def create_and_save_heatmaps(self):
        print("Creating and saving heatmaps with corresponding images and light positions...")

        for i in tqdm(range(len(self.acqs)), desc="Processing acquisitions", unit="acq"):
            # Create folder for the heatmaps if it doesn't exist
            heatmap_folder = os.path.join(self.acqs[i], "distances_cosines_heatmaps")
            os.makedirs(heatmap_folder, exist_ok=True)

            for j, idx in enumerate(tqdm(self.rand_indices[i],
                                        desc=f"Processing images for acq {i+1}/{len(self.acqs)}",
                                        leave=False, unit="img", total=len(self.rand_indices[i]))):
                # Load the distance matrix, corresponding image, and light position
                distance_matrix = self.distance_matrices[i][j]
                cosine_matrix = self.cosine_matrices[i][j]
                image = self.images[i][j]
                light_position = np.array(self.lps_cartesian[i][idx], dtype=float)

                # Normalize the light position to the given radius
                r = self.lps_spherical[i][idx][0]
                light_position = r * light_position / np.linalg.norm(light_position)

                # Create the plot with three subplots
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
                # Plot the distance heatmap
                im1 = ax1.imshow(distance_matrix, cmap='viridis_r', aspect='auto')
                ax1.set_title("Distance Heatmap")
                ax1.axis('off')
                fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

                # Plot the cosine heatmap
                im2 = ax2.imshow(cosine_matrix, cmap='viridis_r', aspect='auto')
                ax2.set_title("Cosine Heatmap")
                ax2.axis('off')
                fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

                # Plot the corresponding image
                ax3.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), aspect='auto')
                ax3.set_title("Corresponding Image")
                ax3.axis('off')

                # Plot the light position in 2D projection of Cartesian coordinates
                ax4.set_xlim(-r, r)
                ax4.set_ylim(-r, r)
                ax4.set_aspect('equal')
                ax4.set_title(f"Light Position: {[round(val, 2) for val in self.lps_cartesian[i][idx]], [round(val, 2) for val in self.lps_spherical[i][idx]]}")
                ax4.set_xlabel("X")
                ax4.set_ylabel("Y")

                # Draw the hemisphere boundary
                circle = plt.Circle((0, 0), r, fill=False, color='black')
                ax4.add_artist(circle)

                # Project and plot the light position
                x, y, z = light_position
                ax4.scatter(x, y, color='red', s=50)

                # Add gridlines
                ax4.grid(True)

                # Save the figure
                plt.tight_layout()
                save_path = os.path.join(heatmap_folder, f"heatmap_image_light_{idx}.png")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                plt.close()

        print("Heatmaps, corresponding images, and light positions saved successfully.")