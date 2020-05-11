# Typical Encoder-Decoder architecture
# Stacked bidirectional LSTM; assumes same number of encoder and decoder layers.
# **Assuming univariate regression task for now.

import torch
import torch.nn as nn


class RecurrentEncoderDecoder(nn.Module):
    """
    Vanilla LSTM Encoder-Decoder network
    
    - Bidirectional LSTM encoder
    - Dropout between layers (vertically)
    - Teacher Forcing (probabilistic per time-step)
    
    **The encoder LSTM and decoder LSTM have the same number of layers.
    """
    def __init__(self, architecture, M, Q, encoder_params, decoder_params):
        super(RecurrentEncoderDecoder, self).__init__()
        #Params
        self.architecture = architecture
        self.M = M
        self.Q = Q
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.decoder_params['M'] = M
        self.decoder_params['Q'] = Q
        
        if encoder_params['architecture']=='recurrent':
            self.encoder = Encoder_recurrent(**self.encoder_params)
        elif encoder_params['architecture']=='conv':
            self.encoder = Encoder_recurrent(**self.encoder_params)            
            
        if decoder_params['architecture']=='recurrent':
            self.decoder = Decoder_recurrent(**self.decoder_params)
        # self.connector = nn.Linear(self.decoder.ddddddddddddddddddddd, self.decoder.d_input)
        
        if (self.encoder.architecture == 'recurrent') and (self.encoder.bidirectional):
            #Independently project both directions for h, and both directions for c:
            N = self.encoder.n_layers * self.encoder.d_hidden
            self.bidi_project__h = nn.Linear(2*N, N)
            self.bidi_project__c = nn.Linear(2*N, N)
            self.nonlin__h = nn.ELU()
            self.nonlin__c = nn.ELU() #e.g.
    
    
    def forward(self, X, future_features, Y_teacher, **kwargs):
        # M = dimensionality of the multiavriate time series
        # F = number of external features
        # Using input X is shape: (batch x length x [F + M]) to be in batch_first LSTM format
        # future_features is (batch x length x F)
        # Y_teacher is (batch x length x M), is only used when using teacher forcing
        
        horizon_span = kwargs['horizon_span']
        teacher_forcing = kwargs['teacher_forcing']
        teacher_forcing_prob = kwargs['teacher_forcing_prob']
        
        batchsize = X.shape[0]
        
        # Use last h from encoder as deocder initial hidden state
        h, c = self.encoder(X)
        
        #If bidirectional encoder, concat the forward and backward directions,
        #and use small feedforward NN and project back to hidden size:
        if self.encoder.bidirectional:
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
        
        #Now the decoder process:
        # Decoder input at 1st decoding timestep is previous value of the 
        # multivariate series (inp_y), along with the current timesrep's features (inp_f):
        inp_y = torch.unsqueeze(X[:,-1,:self.M], 1)            
        all_outputs = self.decoder(batchsize, Y_teacher, inp_y, h, c, future_features, horizon_span, teacher_forcing, teacher_forcing_prob)
        
        return all_outputs






class Encoder_recurrent(nn.Module):
    """
    The Encoder module. For now just uses a basic multilayer LSTM.
    **some assumptions on same dimensions for now, and univariate ts forecasting task
    """
    def __init__(self, architecture, d_input, d_hidden, n_layers, bidirectional, p_dropout_encoder):
        super().__init__()
        # Params
        self.architecture = architecture
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        # Layers
        #batch_first – If True, then the input and output tensors are provided as (batch, seq, feature).
        self.recurrence = nn.LSTM(d_input, d_hidden, n_layers, batch_first=True, bidirectional=self.bidirectional, dropout=p_dropout_encoder)
    def forward(self, X):
        out, (h, c) = self.recurrence(X)
        #for now without attention don't need to return out
        return h, c    
    
    
    
    
    
class Encoder__convolutional(nn.Module):
    """
    An encoder based on a convolutional arhcitecture.
    
    Use stacks of different size 1d convolution kernels, working in parallel
    as ~multihead encoders. / parallel read heads.
    
    Can work on arbitrary length input sequence.
    
    Designed such that each layers output is the same length as the input sequence. 
    This way, skip connections can be used in an elementwise additive way.
    So during decoding, outputs of different layers can be combined, effectively
    allowing the decoder to attend over outputs of different layers, which
    correspond to different levels of abstraction of the input representations.
    E.g. lower level conv have only seen small neighborhood context, but higher 
    levels can aggregate to higher level of abstraction.
    
    To ensure this proper matchup of dimensions, 0-padding is used such that:
        a) kernel_size = odd integer, stride=1, dilation=1
        b) padding = kernel_size - 1
        c) then left and right trim the output by (kernel_size - 1)/2 before passing to next layer
    """
    def __init__(self, input_size):
        super(Encoder__convolutional, self).__init__()
        #Params
        
    def forward(self, x):
        pass    
    
    
    
    
    
    
    
    
class Decoder_recurrent(nn.Module):
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
    def __init__(self, architecture, attention_type, M, Q, d_future_features, d_hidden, n_layers, p_dropout_decoder):
        super().__init__()
        # Params
        self.architecture = architecture
        self.attention_type = attention_type
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
    
    def forward(self, batchsize, Y_teacher, inp_y, h, c, future_features, horizon_span, teacher_forcing, teacher_forcing_prob):
        
        # keep the decoder predictions at each timestep
        outputs = []
        
        # Predict as many timesteps into the future as needed
        # (will vary during training, but each batch will have same length to
        # save wasted computation from padding and predicting on padded values)
        for mm, t in enumerate(range(horizon_span)):
            inp_f = torch.unsqueeze(future_features[:,mm,:], 1)
            #Although slicing out 1st timestep only, keep in usual LSTM rank 3 tensor format
            inp = torch.cat([inp_y, inp_f], dim=2)
            
            #inp is the concatenated tensor of future features with the M recursively predicted outputs for previous timestep of the M-dim multivariate series
            output, (h, c) = self.recurrence(inp, (h, c))
            y_out = self.linear_out_predict(output.squeeze(0))
            outputs.append(y_out)
            
            #Get the part of the output that will actually be used for next timestep's (recursive) input:
            point_inds = torch.arange(self.M) #!!!!!!! indices corresponding to point estimates of the M variables in multivariable forecasting case. If doing quantiles, could just assume 0:M indices is pt. estimates.
            inp_y = y_out[:,:,point_inds] #slice the y_out to get only the elements corresponding to the point estimates

            #However, if using teaher forcing (during training only), then use the ground truth:
            #Draw random number to decide if to use teach forcing on this timestep:
            if teacher_forcing and (torch.rand(1).item() < teacher_forcing_prob):
                print(f'Teacher forcing on step {mm}')
                #If end up doing Teacher Forcing, use groun truth Y_teacher as input to next step:
                inp_y = Y_teacher[:,mm,:]
                inp_y = torch.unsqueeze(inp_y, 1)

        # Stack the outputs along the timestep axis into a tensor with shape:
        # [batchsize x length x output]
        all_outputs = torch.cat([i for i in outputs], dim=1)

        # Multivariate and quantiles:
        # Standardize the shape to be:
        # [batchsize x T x M x Q]
        # e.g. univariate point estimates are then just [batch x T x 1 x 1]
        # multivariate point estimates are [batch x T x M x 1]
        all_outputs = all_outputs.reshape((batchsize, horizon_span, self.M, self.Q))
        
        return all_outputs