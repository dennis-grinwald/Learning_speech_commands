import sys
import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from crnn import CRNN
from data_utils import get_spectrogram_data_loaders

'''
HYPERPARAMETERS
'''
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

'''
GET DATA LOADERS
'''
training_dataloader, testing_dataloader, label_list = get_spectrogram_data_loaders(0.8, batch_size)
inp_size = training_dataloader.dataset.data[0].shape[1]

num_classes = len(label_list)
rnn_input_size = int((inp_size - conv_kernel + 2*padding)/stride + 1) * channels_out

'''
INITIALIZE MODEL
'''
model = CRNN(conv_kernel, channels_out,rnn_input_size,hidden_neurons_1,hidden_neurons_2,fc1,num_classes)
print("MODEL ARCHITECTURE:")
print(model)

'''
INITIALIZE LOSS FUNCTION AND MODEL OPTIMIZER
'''
# Any optimizer can be chosen
optimizer = optim.SGD(params=model.parameters(), lr = lr, momentum=0.9)
lambda1 = lambda epoch: 0.95 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

loss_function = nn.NLLLoss()

'''
START TRAINING PROCEDURE
'''
best_acc = 0
training_loss = []
test_accuracy = []
for epoch in range(epochs):
    total_loss = 0
    '''
    START EVALUATION PROCEDURE (Start evaluation already before training for comparison)
    '''
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (data, true_labels) in enumerate(testing_dataloader):
            data = data.type(torch.FloatTensor)
            true_labels = true_labels.type(torch.LongTensor)

            output_conv, output_lstm1, output_lstm1, predictions = model.out(data)

            total += true_labels.size(0)
            correct += (predictions.max(1)[1] == true_labels).sum().item()
            accuracy = 100 * correct / total

        print('Accuracy of the network on the evaluation dataset: %d %%' % (100 * correct / total))

        if accuracy > best_acc:
            best_acc == accuracy
            # SAVE MODEL
            print("SAVING MODEL")
            torch.save(model.state_dict(), "trained_models/best_model.pt")

    test_accuracy.append(accuracy)

    running_loss = 0.0
    for i , (data, true_labels) in enumerate(training_dataloader):

        data = data.type(torch.FloatTensor)
        true_labels = true_labels.type(torch.LongTensor)

        # set all gradients to zero
        model.zero_grad()

        # Here we get the data from all layers, and the corresponding timesteps
        output_conv, output_lstm1, output_lstm2, predictions = model.out(data)
        loss = loss_function(predictions, true_labels)

        # Optimization part
        loss.backward()

        # Gradient Clipping to avoid exploding gradients
        #nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()
        if i % 20 == 0:  # print every 20 mini-batches
            print('[Epoch: %d, mini batch: %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

    training_loss.append(total_loss/i)

    scheduler.step()

print("Saving results...")
np.save("./results/train_loss", training_loss)
np.save("./results/test_accuracy", test_accuracy)
print('Finished Training')
