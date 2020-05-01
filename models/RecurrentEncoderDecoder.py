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
    def __init__(self, input_size, d_hidden, M, Q, d_future_features, n_layers,
                 bidirectional_encoder, p_dropout_encoder, p_dropout_decoder):
        super(RecurrentEncoderDecoder, self).__init__()
        #Params
        self.input_size = input_size
        self.d_hidden = d_hidden
        self.M = M
        self.Q = Q
        self.d_future_features = d_future_features
        self.n_layers = n_layers
        self.bidirectional_encoder = bidirectional_encoder
        self.p_dropout_encoder = p_dropout_encoder
        self.p_dropout_decoder = p_dropout_decoder
        
        self.encoder = Encoder(self.input_size, self.d_hidden, self.n_layers, self.bidirectional_encoder, self.p_dropout_encoder)
        self.decoder = Decoder(self.M, self.Q, self.d_future_features, self.d_hidden, self.n_layers, self.p_dropout_decoder)
        # self.connector = nn.Linear(self.decoder.ddddddddddddddddddddd, self.decoder.d_input)
        
        if self.bidirectional_encoder:
            #Independently project both directions for h, and both directions for c:
            N = self.encoder.n_layers * self.encoder.d_hidden
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
        if self.bidirectional_encoder:
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

        # =============================================================================
        # actually don't use the connector idea. 
        # instead, assume even mroe multivar, that you would have measured all those vars durign history input.
        # so won't need a conenctor since encoder and decoder will have dimensions for input #features
        # (could always use max(input features, future features) and then nan pad, or 2x dims per feature with [0,1] feature present absent approach...)
        # =============================================================================
            # #If doing quantile forecasting on a univariate input,
            # #the encoder will have d_input=1. 
            # #But for the decodder, we want d_output = #quantiles
            # #Also, we want the previous timestep's quantiles to be recursively fed
            # #into the next timestep as features (same as how a typical deocder 
            # #recursively uses the previous timesteps).
            # #So we need an additional one-time step to convert the d_input dimension
            # #to the d_output dimension:
            # print(inp.shape)
            # # inp = self.connector(inp)
        
        
        print('inp.shape', inp.shape)
        # Predict as many timesteps into the future as needed
        # (will vary during training, but each batch will have same length to
        # save wasted computation from padding and predicting on padded values)
        for t in range(horizon_span):
            y_out, y_next, h, c = self.decoder(inp, h, c)
            # print('deocder forward -----------')
            # print(y_out.shape, y_next.shape, h.shape, c.shape)
            outputs.append(y_out)
            #y_next is same as y_out if NOT doing quantile estimates.
            #but if doing quantile estimates, y_out includes those, but y_next uses only the point estimates in a recursive fashion.
            #(to also use the quantile estimates as recursive input, uncomment the "self.linear_out_next_in" layer in Decoder)
            inp = y_next
        
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
    Decoder module. For now assumes same n_layer and d_hidden dims as encoder...
    
    
    d_future_features - int. Number of features at each timestep. E.g. if using
    some timestamp related features, you know those even for future time steps
    despite not knowing the value of the series itself.
    
    d_output - int. Number of output dimensions. Is M*Q, where M is number of
    variables of output (M=1 for univariate, M>=2 for multivariate) and Q is the
    number of quantiles being forecasted for each output variable dimension.
    **This assumes that each output variable has the SAME set of quantiles being
    forecasted. If you want to use only a susbet of quantiels for optimization
    then provide a mask tensor into the optimization function to exclude those
    quantiles. However, here every quantile will be predicted.
    
    d_hidden - int. Hidden size dimension of h and c of LSTM
    
    n_layers - int. Number of layers
    
    
    Derived params:
    d_input - int. Is the full input size to each step of the decoder, which
    for a given timestep t, concatenates the future features associated with 
    timestep t, with the predicted outputs from timestep t-1 (of size M*Q).
    
    **d_input should be same as encoder
    
    
    """
    def __init__(self, M, Q, d_future_features, d_hidden, n_layers, p_dropout_decoder):
        super().__init__()
        # Params
        #future_features is [batch x d_future_features], where
        self.d_future_features = d_future_features
        self.M = M #if univariate is 1, in general is M for multivariate output distribution
        self.Q = Q #is number of quantiles to predict (assumed same for each of the M variables [but subsets can be masked in loss function])
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        #Derived params:
        self.d_output = self.M * self.Q
        self.d_input = self.d_future_features + self.M
        # Layers
        self.recurrence = nn.LSTM(self.d_input, self.d_hidden, self.n_layers, batch_first=True, dropout=p_dropout_decoder)
        self.linear_out_predict = nn.Linear(self.d_hidden, self.d_output)
        #self.linear_out_next_in = nn.Linear(self.d_output, self.M) #to project all quantile information back into size of point estimates, for next timestep
    def forward(self, inp, h, c):
        print('self.d_output', self.d_output)
        print('self.d_input', self.d_input)
        print('self.d_future_features', self.d_future_features)
        print('self.M', self.M)
        print('inp.shape, h.shape, c.shape', inp.shape, h.shape, c.shape)
        #inp is the concatenated tensor of future features with the M recursively predicted outputs for previous timestep of the M-dim multivariate series
        output, (h, c) = self.recurrence(inp, (h, c))
        y_out = self.linear_out_predict(output.squeeze(0))
        point_inds = [i for i in range(self.M)] #!!!!!!! indices corresponding to point estimates of the M variables in multivariable forecasting case. If doing quantiles, could just assume 0:M indices is pt. estimates.
        y_next = y_out[:,:,point_inds] #slice the y_out to get only the elements corresponding to the point estimates
        print('y_out.shape, y_next.shape', y_out.shape, y_next.shape)
        return y_out, y_next, h, c