# Typical Encoder-Decoder architecture
# Stacked bidirectional LSTM; assumes same number of encoder and decoder layers.
# **Assuming univariate regression task for now.

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf

class RecurrentEncoderDecoder(nn.Module):
    """
    Vanilla LSTM Encoder-Decoder network
    
    - Bidirectional LSTM encoder
    - Dropout between layers (vertically)
    - Teacher Forcing (probabilistic per time-step)
    
    **The encoder LSTM and decoder LSTM have the same number of layers.
    """
    def __init__(self, M, Q, encoder_params, decoder_params):
        super(RecurrentEncoderDecoder, self).__init__()
        #Params
        self.M = M
        self.Q = Q
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.architecture = self.encoder_params['architecture'] + '->' + self.decoder_params['architecture']
        self.decoder_params['M'] = M
        self.decoder_params['Q'] = Q
        
        if encoder_params['architecture']=='LSTM':
            self.encoder = Encoder_recurrent(**self.encoder_params)
        elif encoder_params['architecture']=='conv':
            self.encoder_params['n_layers_decoder'] = self.decoder_params['n_layers_decoder']
            self.encoder = Encoder_convolutional(**self.encoder_params)            
            
        if decoder_params['architecture']=='LSTM':
            self.decoder = Decoder_recurrent(**self.decoder_params)
        # self.connector = nn.Linear(self.decoder.ddddddddddddddddddddd, self.decoder.d_input)
        
        if (self.encoder.architecture == 'LSTM') and (self.encoder.bidirectional):
            #Independently project both directions for h, and both directions for c:
            N = self.encoder.n_layers_encoder_encoder * self.encoder.d_hidden
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
        print(kwargs)
        horizon_span = kwargs['horizon_span']
        teacher_forcing = kwargs['teacher_forcing']
        teacher_forcing_prob = kwargs['teacher_forcing_prob']
        
        batchsize = X.shape[0]
        
        # Use last h from encoder as deocder initial hidden state
        h, c = self.encoder(X)


        #If bidirectional encoder, concat the forward and backward directions,
        #and use small feedforward NN and project back to hidden size:
        if (self.encoder.architecture=='LSTM') and (self.encoder.bidirectional):
            #Concat the forward and backward directions for hidden state vectors, h,
            #and pass through small feedforward network to project back to single hidden size:
            h = h.view(self.encoder.n_layers_encoder, 2, batchsize, self.encoder.d_hidden)
            h = torch.cat((h[:,0,:,:], h[:,1,:,:]), dim=2).transpose(0,1)
            h = h.reshape((batchsize, -1)) #flatten layers dimension with the (directions x hidden) dim
            h = self.bidi_project__h(h)
            h = self.nonlin__h(h)
            h = h.reshape((batchsize, self.encoder.n_layers_encoder, self.encoder.d_hidden))
            h = h.transpose(0,1)
            
            #And do exactly the same for the cell memory vector c:
            c = c.view(self.encoder.n_layers_encoder, 2, batchsize, self.encoder.d_hidden)
            c = torch.cat((c[:,0,:,:], c[:,1,:,:]), dim=2).transpose(0,1)
            c = c.reshape((batchsize, -1)) #flatten layers dimension with the (directions x hidden) dim
            c = self.bidi_project__h(c)
            c = self.nonlin__h(c)
            c = c.reshape((batchsize, self.encoder.n_layers_encoder, self.encoder.d_hidden))
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
    def __init__(self, architecture, d_input, d_hidden, n_layers_encoder, bidirectional, p_dropout_encoder):
        super().__init__()
        # Params
        self.architecture = architecture
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers_encoder = n_layers_encoder
        self.bidirectional = bidirectional
        # Layers
        #batch_first – If True, then the input and output tensors are provided as (batch, seq, feature).
        self.recurrence = nn.LSTM(d_input, d_hidden, n_layers_encoder, batch_first=True, bidirectional=self.bidirectional, dropout=p_dropout_encoder)
    def forward(self, X):
        out, (h, c) = self.recurrence(X)
        #for now without attention don't need to return out
        return h, c
    
    
    
    
    
class Encoder_convolutional(nn.Module):
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
    def __init__(self, architecture, d_input, d_hidden, n_layers_encoder, n_layers_decoder, kernel_sizes, n_filters_each_kernel, reduce_ops_list):
        super().__init__()
        #Params
        self.architecture = architecture
        # self.T_history = T_history #For this encoder, the history size is fixed
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.kernel_sizes = kernel_sizes        
        self.n_filters_each_kernel = n_filters_each_kernel
        self.N_heads = len(self.kernel_sizes)
        
        self.reduce_ops_list = reduce_ops_list
        self.N_ops = len(self.reduce_ops_list)
        
        #Confirm kernel iszez and number of kernels matches; and kernel sizes are all positive odd ints
        assert(self.N_heads==len(self.n_filters_each_kernel))
        assert((i>0 and type(i)==int and i%2==1 for i in self.kernel_sizes))
        
        #Using 2D conv since [T x features]
        self.convs = nn.ModuleList([nn.Conv2d(1, self.n_filters_each_kernel[i], (self.kernel_sizes[i], self.d_input)) for i in range(len(self.kernel_sizes))])
        self.convs_reduce = nn.ModuleList([nn.Conv2d(self.n_filters_each_kernel[i], 1, (1, 1)) for i in range(len(self.kernel_sizes))])
        
        #FC to merge different branches of conv read heads:
        # N_in_features = self.N_heads*self.d_input*self.T_history
        # N_out_features = self.d_input*self.T_history
        # self.merge_heads_fc = nn.Linear(N_in_features, N_out_features)
        # print('N_heads', self.N_heads)
        # print('N_in_features', N_in_features)
        # print('N_out_features', N_out_features)
        
        self.merge_heads_conv = nn.Conv2d(4, 1, (1,1))
        
        # Final layers to make [batch x 1 x T_history x features] -> [batch x d_hidden]
        # to pass into recurrent decoder as initial hidden states:
        self.linear_out_h = nn.Linear(self.d_input*self.N_ops, self.d_hidden*self.n_layers_decoder)
        self.linear_out_c = nn.Linear(self.d_input*self.N_ops, self.d_hidden*self.n_layers_decoder)
        
    def merge_heads(self, tensor_list):
        """
        Merge the outputs of the different attention heads (sets of different 
        size kernel filters).
        
        They are merged into a single output to feed into the next layer
        
        **E.g. below, just doing basic fully connected layer with nonlinearity
        to reduce the [batch x N_heads x F x T] to [batch x 1 x F x T]:
        """
        x = torch.cat([i for i in tensor_list], dim=1)
        x = self.merge_heads_conv(x)
        x = F.elu(x)
        # self.merge_heads_fc
        return x
        

    def pad_conv_slice_res(self, index, x_0):
        """
        For a given kernel_size (attention head):
            - pad left and right side of input by (kernel_size - 1)
            - conv layer
            - trim by removing (kernel_size - 1)/2 from both sides to make
              same output dimension as input dimension
            - merge all filters into 1
            - non-linearities / normalization, etc.
            - residual connection
            
        x_0 - input tensor is [batch x T_history x features]
        """
        
        # print('padddd conv slice ----------')
        kernel_size = self.kernel_sizes[index]
        # n_filters = self.n_filters_each_kernel[index]
        # print('index', index)
        # print('x_0.shape', x_0.shape)
        # print('kernel_size', kernel_size)
        # print('n_filters', n_filters)
        
        #Pad input:
        padsize = kernel_size - 1
        pad_tuple = (0,0,padsize,padsize)
        x = F.pad(x_0, pad_tuple, "constant", 0)
        

        #Conv
        # print('x.shape after pad', x.shape)
        # print(self.convs[index])
        x = self.convs[index](x)
        # print('x.shape after conv', x.shape)
        # print(x[0,0,:,:])
        
        
        #Trim
        ind = int((kernel_size-1)/2)
        # print(ind)
        x = x[:,:,ind:-ind,:]
        # print(x)
        # print('x.shape after trim', x.shape)    
        # print(x[0,0,:,:])
        
        
        #Merge
        x = self.convs_reduce[index](x)
        # print('x.shape after reduce', x.shape) 
        #Repeat tensor along features dimension to be original shape ()
        x = x.repeat(1,1,1,self.d_input)
        # print('x.shape after repeat', x.shape) 
        
        #Residual connection:
        x += x_0
        return x


    def reduce_along_time(self, x, reduce_ops_list):
        """
        At the end of the conv encoder, need to reduce along T_history axis
        to be applicable to arbitrary sized inputs.
        
        E.g. take max, min, mean, etc. to reduce variable sized axis to fixed
        size representation
        
        x input is [batch x 1 x T_history x features]
        is reduced to
        [batch x (features*ops)]
        """
        v = []
        for op in reduce_ops_list:
            v.append(op(x,axis=2))
        v = torch.cat(v, dim=2)
        v = v.squeeze()
        return v
        
        

    def forward(self, X):
        
        #Put into format [batch x channels x T_history x features]
        X = X.unsqueeze(1)
        batchsize = X.shape[0]
        
        #Just doing glimpses with shared weights, i.e. for each layer, the 
        #parameters for the branch corresponding to a given kernel_size are reused
        #(shared across layers):
        for LL in range(self.n_layers_encoder):
            outputs_this_layer = []
            for HH in range(self.N_heads):
                outputs_this_layer.append(self.pad_conv_slice_res(HH, X))
            X = self.merge_heads(outputs_this_layer)

        X = self.reduce_along_time(X, self.reduce_ops_list)# is now [batch x (features*ops)]
        h = F.elu(self.linear_out_h(X))
        c = F.elu(self.linear_out_c(X))
        
        # Reshape to be decoder input format for h, c:
        # [n_layers_decoder x batch x ]
        h = h.reshape(batchsize, self.d_hidden, self.n_layers_decoder).permute([2,0,1])
        c = c.reshape(batchsize, self.d_hidden, self.n_layers_decoder).permute([2,0,1])
        
        return h, c
    
    
    
    
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
    
    n_layers_decoder - int. Number of layers
    
    
    Derived params:
    d_input - int. Is the full input size to each step of the decoder, which
    for a given timestep t, concatenates the future features associated with 
    timestep t, with the predicted outputs from timestep t-1 (of size M*Q).
    
    **d_input should be same as encoder
    
    
    """
    def __init__(self, architecture, attention_type, M, Q, d_future_features, d_hidden, n_layers_decoder, p_dropout_decoder):
        super().__init__()
        # Params
        self.architecture = architecture
        self.attention_type = attention_type
        #future_features is [batch x d_future_features], where
        self.d_future_features = d_future_features
        self.M = M #if univariate is 1, in general is M for multivariate output distribution
        self.Q = Q #is number of quantiles to predict (assumed same for each of the M variables [but subsets can be masked in loss function])
        self.d_hidden = d_hidden
        self.n_layers_decoder = n_layers_decoder
        #Derived params:
        self.d_output = self.M * self.Q
        self.d_input = self.d_future_features + self.M
        # Layers
        self.recurrence = nn.LSTM(self.d_input, self.d_hidden, self.n_layers_decoder, batch_first=True, dropout=p_dropout_decoder)
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