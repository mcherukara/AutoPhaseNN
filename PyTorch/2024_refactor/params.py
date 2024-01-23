#Data parameters
scale_I = 1 # normalize diff or not
N_TRAIN = 50000
N_VALID = int(0.1*N_TRAIN)
TRAIN_ratio = 0.9
N_TEST = 100


#File parameters
MODEL_SAVE_PATH = 'models/'
data_path = '../../data/CDI_simulation_upsamp_noise/'


#Training parameters
EPOCHS = 60 #Full cycle is 12 epochs, good to end on a minimum

#Scale learning rate with batch size to avoid generalization gap
BATCH_SIZE_SCALER = 4
BATCH_SIZE_PER_GPU = 32 * BATCH_SIZE_SCALER #How many per GPU 
# Total Batch size is in turn scaled based on how many GPUs are used for training
BASE_LR = 1e-4 * BATCH_SIZE_SCALER #Scaled by batch size and number of GPUs

INIT_SW = 0.07 #Initial shrink wrap
FINAL_SW = 0.1  #Final shrink wrap
CONST_EPOCHS = 0 #How many epochs to not increase SW
