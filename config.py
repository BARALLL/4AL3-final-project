DATASET_NAME = "imdb"
PRETRAINED_MODEL = "bert-base-uncased"
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-3 #lr_scheduler?
L2_REG = 1e-3
MAX_LENGTH = 512
PICKLE_FILE = "tokenized_imdb.pkl"  # File to store the tokenized dataset
MODEL_FILE = 'cnn-model.pt'
SEED = 0
TRAIN = True
# Training Validation Test
SPLIT_PERCENTAGES = [0.7, 0.1, 0.2]


EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
DROPOUT = 0.5

#TODO
#- lr_scheduler?
#- early stopping?