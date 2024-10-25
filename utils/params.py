# ACQ_PATHS = [r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_1_1/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_1_2/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_1_3/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_1_4/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_1_5/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_2_1/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_2_2/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_2_3/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_2_4/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_2_5/images/Face_A/rti'
#             ]

ACQ_PATHS = [r'/work/imvia/ra7916lu/illumi-net/data/2024_02_22_1_1/rti_sub_images']
             
             
            
# Dataset params
MAX_NB_IMAGES_PER_ACQ = 105
COMPUTE_NORMALS_AND_ALBEDO = False # Ensure TRAINING is True
COMPUTE_DISTANCES_AND_COSINES = False # Ensure TRAINING is True
CREATE_DIST_COSINES_HEATMAPS = False # Ensure TRAINING is True

# SURFACE_PHYSCIAL_SIZE = [(128*0.250)/2704, (128*0.160)/1800] #default [0.250, 0.120]
SURFACE_PHYSCIAL_SIZE = [0.250, 0.160] #default [0.250, 0.160]


# Compute normals params
PS_METHOD = "L2_SOLVER"    # Least-squares
# PS_METHOD = "L1_SOLVER_MULTICORE"    # L1 residual minimization
# PS_METHOD = "SBL_SOLVER_MULTICORE"    # Sparse Bayesian Learning
# PS_METHOD = "RPCA_SOLVER"    # Robust PCA


# RTI training
RTI_NET_EPOCHS = 200000
RTI_NET_SAVE_MODEL_EVERY_N_EPOCHS = 10
# RTI_NET_PATCH_SIZE = [256,256] #[height, width]. Must be divisible by 32  or [192, 192], [256, 256], etc.
RTI_NET_PATCH_SIZE = [128,128] #[height, width]. Must be divisible by 32  or [192, 192], [256, 256], etc.
RTI_MAX_NUMBER_PATCHES = 20 # Randomly choose desired number of patches from the image. 
RTI_MAX_NUMBER_PATCHES = 1

# RTI Relighting
RTI_MODEL_PATH = r'/work/imvia/ra7916lu/illumi-net/saved_models/relighting_model_epoch_60.pth'

# Goal
TRAINING = True # Training: True, Relighting: False. If you choose relighting- ensure the path contains - 1. distances.npy, 2. cosines.npy, 3. albedo and 4. normals


# Model Params
BATCH_SIZE = 32
TRAIN_SHUFFLE = True
VAL_SHUFFLE = False
NUM_WORKERS = 4
LEARNING_RATE = 0.0001
LAMBDA_MSE = 1.0
LAMBDA_PERCEPTUAL = 0.1
LAMBDA_DETAIL = 0.5
SPECULAR_WEIGHT = 1.0  # Weight for specular detail loss
SHADOW_WEIGHT = 0.5    # Weight for shadow consistency
NORMAL_CONSISTENCY_WEIGHT = 0.3  # Weight for normal map consistency
ROUGHNESS_WEIGHT = 0.2  # Weight for surface roughness
PERSISTENT_WORKER =True,
PIN_MEMORY =True,
PREFETCH_FACTOR=3
CUDA_VISIBLE_DEVICES = "0,1"

# Visualizations
SAMPLE_NUMBER_OF_COMPARISONS = 8
