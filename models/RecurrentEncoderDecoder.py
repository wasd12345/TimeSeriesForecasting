# A typical Encoder-Decoder architecture
# Uses LSTM, assumes same encoder and deocder layer dims
# assuming univariate regression task for now.

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentEncoderDecoder(nn.Module):
    """
    Vanilla Encoder-Decoder network
    """
    def __init__(self, encoder, decoder):
        super(RecurrentEncoderDecoder, self).__init__()
        #Params
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X, Y):
        # Using input X is shape: (batch x length x features) to be in batch_first LSTM format
        # Y is similar, except feature dims will not include the series itself (or lagged inputs, etc.)
        # (but may includle known future timestamp related features)
        
        # keep the decoder predictions at each timestep
        outputs = []
        
        # Use last h from encoder as deocder initial hidden state
        h, c = self.encoder(X)
        
        # Decoder input at 1st decoding timestep
        inp = Y[:,0,:]
        inp = torch.unsqueeze(inp, 1) #Although slicing out 1st timestep only, keep in usual LSTM rank 3 tensor format
        
        # Dealing with randomized input/output lengths during training:
        horizon_size = Y.shape[1]
        
        # Predict as many timesteps into the future as needed
        # (will vary during training, but each batch will have same length to
        # save wasted computation from padding and predicting on padded values)
        for t in range(horizon_size):
            y, h, c = self.decoder(inp, h, c)
            # print('deocder forward -----------')
            # print(y.shape, h.shape, c.shape)
            outputs.append(y)
            inp = y
            
        #Make the outputs into a tensor:
        all_outputs = torch.cat([i for i in outputs], dim=1)
        return all_outputs





class Encoder(nn.Module):
    """
    The Encoder module. For now just uses a basic multilayer LSTM.
    **some assumptions on same dimensions for now, and univariate ts forecasting task
    """
    def __init__(self, d_input, d_hidden, n_layers):
        super().__init__()
        # Params
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        # Layers
        #batch_first – If True, then the input and output tensors are provided as (batch, seq, feature).
        self.recurrence = nn.LSTM(d_input, d_hidden, n_layers, batch_first=True)
    def forward(self, X):
        out, (h, c) = self.recurrence(X)
        return h, c    
    
    
    
    
    
class Decoder(nn.Module):
    """
    Decoder module. For now, if multilayer, assume same dims as encoder...
    **can change d_output to >1 for multivariate regression but for now just test with =1
    """
    def __init__(self, d_output, d_input, d_hidden, n_layers):
        super().__init__()
        # Params
        self.d_output = d_output
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        # Layers
        self.recurrence = nn.LSTM(d_input, d_hidden, n_layers, batch_first=True)
        self.linear = nn.Linear(d_hidden, d_output)
    def forward(self, inp, h, c):
        # print(inp.shape)
        output, (h, c) = self.recurrence(inp, (h, c))
        y = self.linear(output.squeeze(0))
        return y, h, c