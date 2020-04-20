# A typical Encoder-Decoder architecture


import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    """
    Vanilla Encoder-Decoder network
    """
    def __init__(self, ..., encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        ....

        
    def forward(self, x):
        pass





class Encoder(nn.Module):
    """
    The Encoder module
    """
    def __init__(self, ...):
        super(Encoder, self).__init__()
        
    def forward(self, x):
        pass
    
    
    
class Decoder(nn.Module):
    """
    The Decoder module
    """
    def __init__(self, ...):
        super(Decoder, self).__init__()
        
    def forward(self, x):
        pass