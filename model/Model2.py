import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
import os

class BiLSTM2(nn.Module): 
    # This NLP part Will consist of two bidirectional lstm layers and it's output is 
    # determined by the LSTM's last hidden states or output vectors.

    # This will take as an input a sequence of words and output the last hidden layer
    # the last hidden states of 2-layer bidirectional LSTM will be the input of the last multimodel network 

    def __init__(self, embedding_dim, hidden_dim = 256, layer_dim =2, output_dim = 10):
        super(BiLSTM2, self).__init__()
        
        
        # Replace this when using GPU
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        
        self.embedding_dim = embedding_dim
        
        #Hidden dimensions
        self.hidden_dim = hidden_dim # maybe set this to 256

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building the LSTM 
        # batch_first = True causes the input/output to be of shape 3D (batch_dim, seq_dim, feature_dim) 
        # output will be the same dim as the hidden dim
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
                
        # Initialize hidden state with zeros
        # self.layer_dim * 2. because we have one going forwards and another going backwards
        #h0 = torch.randn(self.layer_dim * 2, x.size(0), self.hidden_dim, device=self.device)
        h0 = torch.randn(self.layer_dim * 2, x.size(0), self.hidden_dim)
        
        
        
        # Initialize cell state
        #c0 =  torch.randn(self.layer_dim * 2, x.size(0), self.hidden_dim, device=self.device)
        c0 =  torch.randn(self.layer_dim * 2, x.size(0), self.hidden_dim)
        
        # We suppose we are conducting a 28 time steps In case of using 
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm1(x, (h0.detach(), c0.detach()))
        # out = self.fc(out.view(out.size(0), -1))
          
        # Without the activation function, out will contain the last hidden layer.
        # This could be obtianed from hn[-1] as well.
        out = out[:, -1, :]
        
        out = self.dropout(out)
        
        out = self.fc(out)
        
        
        return out
        
        # Index hidden state of last time step
        # out.size() --> 256, 100, 256 if we have (input dim = 100 and hidden dim = 100)
        # out[:, -1, :] => 256, 256 --> because we just want the last time step hidden states
        #out = out[:, -1, :] # without an activation function

        # now our: out.size() --> 256, 10 (if output dimension is equal to 10)
        #return out