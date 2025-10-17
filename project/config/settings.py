"""
Configuration settings for the Music Informatics project.

This module contains default values for various parameters used throughout the project.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# Data processing
ACOUSTICBRAINZ_DIR = DATA_DIR / "acousticbrainz_raw30s"
JAMENDO_DIR = PROJECT_ROOT.parent / "mtg-jamendo-dataset" / "data"

# Random seed for reproducibility
RANDOM_SEED = 42

# PCA settings
PCA_N_COMPONENTS = 36
PCA_EXPLAINED_VARIANCE_RATIO = 0.95

# K-means settings
KMEANS_N_TOKENS = 2
KMEANS_N_CLUSTERS = 128
KMEANS_MAX_ITER = 200
KMEANS_N_INIT = 16

# Dictionary Learning settings
DL_N_NONZERO_COEFS = 2
DL_N_DICT_COMPONENTS = 128
DL_MAX_ITER = 100
DL_BATCH_SIZE = 256

# T5 settings
T5_MODEL_NAME = "t5-base"
T5_MAX_LENGTH = 128
T5_UPSAMPLE_THRESHOLD = 16
T5_MAX_REPLICATION = 6

# Training settings
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 500
LOGGING_STEPS = 100
EVAL_STEPS = 200
SAVE_STEPS = 200

# Visualization settings
VISUALIZATION_SAMPLE_SIZE = 1000
VISUALIZATION_N_TOP_TAGS = 5
