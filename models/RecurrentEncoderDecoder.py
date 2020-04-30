# Typical Encoder-Decoder architecture
# Stacked bidirectional LSTM; assumes same number of encoder and decoder layers.
# **Assuming univariate regression task for now.

import torch
import torch.nn as nn


class RecurrentEncoderDecoder(nn.Module):
    """
    Vanilla Encoder-Decoder network
    
    Bidirectional LSTM encoder
    
    Dropout between layers (vertically)
    
    **The encoder LSTM and decoder LSTM have the same number of layers.
    **The regression output is assumed univariate for now.
    """
    def __init__(self, encoder, decoder):
        super(RecurrentEncoderDecoder, self).__init__()
        #Params
        self.encoder = encoder
        self.decoder = decoder
        self.connector = nn.Linear(self.decoder.d_input, self.decoder.d_output)
        if self.encoder.bidirectional:
            #Independently project both directions for h, and both directions for c:
            N = self.encoder.n_layers * self.encoder.d_hidden
            print('N', N)
            self.bidi_project__h = nn.Linear(2*N, N)
            self.bidi_project__c = nn.Linear(2*N, N)
            self.nonlin__h = nn.ELU()
            self.nonlin__c = nn.ELU() #e.g.
    
    def forward(self, X, Y, horizon_span):
        # Using input X is shape: (batch x length x features) to be in batch_first LSTM format
        # Y is similar, except feature dims will not include the series itself (or lagged inputs, etc.)
        # (but may includle known future timestamp related features)
        
        # keep the decoder predictions at each timestep
        outputs = []
        
        # Use last h from encoder as deocder initial hidden state
        h, c = self.encoder(X)
        
        #If bidirectional encoder, concat the forward and backward directions,
        #and use small feedforward NN and project back to hidden size:
        if self.encoder.bidirectional:
            batchsize = X.shape[0]
            
            #Concat the forward and backward directions for hidden state vectors, h,
            #and pass through small feedforward network to project back to single hidden size:
            h = h.view(self.encoder.n_layers, 2, batchsize, self.encoder.d_hidden)
            h = torch.cat((h[:,0,:,:], h[:,1,:,:]), dim=2).transpose(0,1)
            h = h.reshape((batchsize, -1)) #flatten layers dimension with the (directions x hidden) dim
            h = self.bidi_project__h(h)
            h = self.nonlin__h(h)
            h = h.reshape((batchsize, self.encoder.n_layers, self.encoder.d_hidden))
            h = h.transpose(0,1)
            
            #And do exactly the same for the cell memory vector c:
            c = c.view(self.encoder.n_layers, 2, batchsize, self.encoder.d_hidden)
            c = torch.cat((c[:,0,:,:], c[:,1,:,:]), dim=2).transpose(0,1)
            c = c.reshape((batchsize, -1)) #flatten layers dimension with the (directions x hidden) dim
            c = self.bidi_project__h(c)
            c = self.nonlin__h(c)
            c = c.reshape((batchsize, self.encoder.n_layers, self.encoder.d_hidden))
            c = c.transpose(0,1)
            
            #h and c are now [n_layers x bacthsize x d_hidden],
            #which is exactly as needed for the (unidirectional) decoder

        # Decoder input at 1st decoding timestep
        #inp = Y[:,0,:]
        inp = X[:,-1,:]
        inp = torch.unsqueeze(inp, 1) #Although slicing out 1st timestep only, keep in usual LSTM rank 3 tensor format

        #If doing quantile forecasting on a univariate input,
        #the encoder will have d_input=1. 
        #But for the decodder, we want d_output = #quantiles
        #Also, we want the previous timestep's quantiles to be recursively fed
        #into the next timestep as features (same as how a typical deocder 
        #recursively uses the previous timesteps).
        #So we need an additional one-time step to convert the d_input dimension
        #to the d_output dimension:
        inp = self.connector(inp)
        
        # Predict as many timesteps into the future as needed
        # (will vary during training, but each batch will have same length to
        # save wasted computation from padding and predicting on padded values)
        for t in range(horizon_span):
            y, h, c = self.decoder(inp, h, c)
            # print('deocder forward -----------')
            # print(y.shape, h.shape, c.shape)
            outputs.append(y)
            inp = y
        
        # Stack the outputs along the timestep axis into a tensor with shape:
        # [batchsize x length x output]
        all_outputs = torch.cat([i for i in outputs], dim=1)
        return all_outputs





class Encoder(nn.Module):
    """
    The Encoder module. For now just uses a basic multilayer LSTM.
    **some assumptions on same dimensions for now, and univariate ts forecasting task
    """
    def __init__(self, d_input, d_hidden, n_layers, bidirectional, p_dropout_encoder):
        super().__init__()
        # Params
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        # Layers
        #batch_first – If True, then the input and output tensors are provided as (batch, seq, feature).
        self.recurrence = nn.LSTM(d_input, d_hidden, n_layers, batch_first=True, bidirectional=self.bidirectional, dropout=p_dropout_encoder)
    def forward(self, X):
        out, (h, c) = self.recurrence(X)
        return h, c    
    
    
    
    
    
class Decoder(nn.Module):
    """
    Decoder module. For now, if multilayer, assume same dims as encoder...
    **can change d_output to >1 for multivariate regression but for now just test with =1
    """
    def __init__(self, d_output, d_input, d_hidden, n_layers, p_dropout_decoder):
        super().__init__()
        # Params
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        # Layers
        self.recurrence = nn.LSTM(d_output, d_hidden, n_layers, batch_first=True, dropout=p_dropout_decoder)
        self.linear = nn.Linear(d_hidden, d_output)
    def forward(self, inp, h, c):
        output, (h, c) = self.recurrence(inp, (h, c))
        y = self.linear(output.squeeze(0))
        return y, h, c