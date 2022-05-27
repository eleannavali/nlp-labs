import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

EMB_PATH = os.path.join(BASE_PATH, "embeddings")

DATA_PATH = os.path.join(BASE_PATH, "datasets")

# Mean of lens: 21.5925
# Median of lens: 21.0
# Min-max of lens: 1-62

MAX_LENGTH=40
EMB_DIM = 50 
#NUM_EMB = 

