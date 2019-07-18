import os
import sys

import json

import numpy as np
import torch

from crnn import CRNN
from data_utils import get_spectrogram_data_loaders
'''
1. Load trained model and data sets
2. Load the data to put through it
3. Collect Layer Activations
'''

#######################################
################## 1. #################
#######################################
hp = None
with open("training.json") as json_file:
    hp = json.load(json_file)

# training hyperparameters
epochs = hp["epochs"]
batch_size = hp["batch_size"]
lr = hp["learning_rate"]
clip = hp["gradient_clip"]

# convolution hyperparameters
conv_kernel = hp["kernel_size"]
stride = hp["stride"]
padding = hp["padding"]
channels_out = hp["channels_out"]

# RNN hyperparameters
hidden_neurons_1 = hp["hidden_layer_1"]
hidden_neurons_2 = hp["hidden_layer_2"]

# Fully connected layers
fc1 = hp["fc1"]

training_dataloader, testing_dataloader, label_list = get_spectrogram_data_loaders(0.8, 30)

print(training_dataloader.dataset.data[0].shape)
sys.exit(0)


inp_size = training_dataloader.dataset.data[0].shape[1]

#num_classes = len(label_list)
num_classes = 31
rnn_input_size = int((inp_size - conv_kernel + 2*padding)/stride + 1) * channels_out

print(f"INPUT SIZE: {inp_size}")
print(f"RNN INPUT SIZE: {rnn_input_size}")
sys.exit(0)

model = CRNN(conv_kernel, channels_out, rnn_input_size, hidden_neurons_1, hidden_neurons_2, fc1, num_classes)
print("MODEL ARCHITECTURE:")
print(model)
print("Load trained model weights...")

model_path = "trained_models/128_hidden/run2/best_model_1fc.pt"
model.load_state_dict(torch.load(model_path))
model.eval()
print("Loaded trained model weights successfully!")

#######################################
################## 2. #################
#######################################
os.makedirs("activations", exist_ok=True)
for i, (data, true_labels) in enumerate(testing_dataloader):

    data = data.type(torch.FloatTensor)
    true_labels = true_labels.type(torch.LongTensor)

    stg_activations, tp_activations, ifg_activations, word_predictions = model.out(data)

    torch.save(stg_activations, 'activations/stg_{}.pt'.format(i))
    torch.save(tp_activations, 'activations/tp_{}.pt'.format(i))
    torch.save(ifg_activations, 'activations/ifg_{}.pt'.format(i))

    '''
    print(stg_activations.shape) => torch.Size([2130, 3, 120])
    print(tp_activations.shape) => torch.Size([30, 71, 128])
    print(ifg_activations.shape) => torch.Size([30, 71, 128])
    sys.exit(0)
    '''

    print(stg_activations.shape)
    print(tp_activations.shape)
    print(ifg_activations.shape)

    sys.exit(0)



#######################################
################## 3. #################
#######################################
