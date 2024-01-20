#Data parameters
scale_I = 1 # normalize diff or not
N_TRAIN = 5000
N_VALID = int(0.1*N_TRAIN)
TRAIN_ratio = 0.9


#File parameters
MODEL_SAVE_PATH = 'models/'
data_path = '../../data/CDI_simulation_upsamp_noise/'


#Training parameters
EPOCHS = 60 #Full cycle is 12 epochs, good to end on a minimum
BASE_LR = 1e-4 #Scaled by number of GPUs and triangle2 cyclicLR
INIT_SW = 0.07 #Initial shrink wrap
FINAL_SW = 0.1  #Final shrink wrap
CONST_EPOCHS = 0 #How many epochs to not increase SW
