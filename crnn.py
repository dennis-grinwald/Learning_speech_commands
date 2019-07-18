import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class CRNN(nn.Module):

    def __init__(self, kernel_size, channels_out, rnn_input_size, hidden1, hidden2, fc1, output_size):
        super(CRNN, self).__init__()

        # Convolutional-layer
        self.conv = nn.Conv1d(1, channels_out, kernel_size)
        self.pool =  nn.MaxPool2d(2)

        self.output_size = output_size
        self.hidden_dim_1 = hidden1
        self.hidden_dim_2 = hidden2
        self.rnn_input_size = rnn_input_size

        # Recurrent Layers
        self.lstm_1 = nn.LSTM(rnn_input_size, hidden1, bidirectional=False, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden1, hidden2, bidirectional=False, batch_first=True)

        # Fully-connected layer for classification
        self.fc1 = nn.Linear(hidden2, output_size)
        #self.fc2 = nn.Linear(fc1, output_size)

    def out(self, input):

        '''
        CONVOLUTIONAL LAYER
        '''
        seq_len = input.size(3)
        batch_size = input.size(0)
        no_channels = input.size(1)

        input = input.view(batch_size*seq_len,no_channels,-1)

        output_conv = self.conv(input)
        # Apply non-linear activation function
        output_conv = F.relu(output_conv)
        stp_out = torch.mean(output_conv, dim=1).view(batch_size,seq_len,-1)

        # Prepare data batch for RNN-layer, INPUT EXPECTED: batch_size, sequence_length, features
        output_conv_reshaped = output_conv.view(batch_size,seq_len,-1)

        '''
        RNN LAYER 1
        '''
        h_0 = None
        output_lstm_1, (hidden_state_lstm_1, _) = self.lstm_1(output_conv_reshaped, h_0)

        '''
        RNN LAYER 2
        '''
        output_lstm_2, (hidden_state_lstm_2, _) = self.lstm_2(output_lstm_1, h_0)

        '''
        FC-/OUTPUT LAYER
        '''
        output = self.fc1(hidden_state_lstm_2.view(batch_size,self.hidden_dim_2))
        #output = F.relu(output)
        #output = self.fc2(output)
        output = F.log_softmax(output, dim=1)

        return stp_out, output_lstm_1, output_lstm_2, output
