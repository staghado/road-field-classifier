# Device
DEVICE = 'cpu'

# Image size
IMG_SIZE = (256, 256)

# Training hyperparameters
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
MOMENTUM = 0.9
STEP_SIZE = 3
GAMMA = 0.1

# data loading
BATCH_SIZE = 4
NUM_WORKERS = 2

# Data paths
ROOT_DIR = './dataset/'
VALIDATION_RATIO = 0.1
RANDOM_SEED = 42

# data augmentation
MEANS, STDS = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
NUM_OPS = 3
MAGNITUDE = 9

# save directory
SAVE_DIR = './checkpoints'