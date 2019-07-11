from torch import nn
from convlstm2d import ConvLSTMCell
import torch
from seq2seq_parts import *
IMAGES_DEPTH = 1

class Encoder(nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self):
        
        super().__init__()
        self.inc = inconv(1, 16) #16
        self.down1 = down(16, 32)#16, 32
        self.down2 = down(32, 48)#32, 48
        self.down3 = down(48, 64)#48,64 
        self.down4 = down(64, 80)#64, 80
        self.cell = ConvLSTMCell(80, 80)
        

    def forward(self, x, states = None):
        ''' Forward pass through the network, returns the output logits '''
        x = x.cuda()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        output, hidden = self.cell(x5, states)
        states = [F.elu(output), F.elu(hidden)]
        return [x1, x2, x3, x4, x5], F.elu(output), states

class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.cell = ConvLSTMCell(80, 80)
        self.up0 = up(160, 80)
        self.up1 = up(144, 64) 
        self.up2 = up(112, 48)
        self.up3 = up(80, 32)
        self.up4 = up(48, 16)
        self.outc = outconv(16, 1)
        

    def forward(self, layers, output, states):
        ''' Forward pass through the network, returns the output logits '''
        output, hidden = self.cell(output, states)
        x = self.up0(F.elu(output), layers[4])
        x = self.up1(x, layers[3])
        x = self.up2(x, layers[2])
        x = self.up3(x, layers[1])
        x = self.up4(x, layers[0])
        x = self.outc(x)
        states = [F.elu(output), F.elu(hidden)]        
        return x, states

import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
      
    #initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, features, targets, teacher_forcing_ratio=0.5):
        
        batch_size = targets.shape[1]
        time = targets.shape[0]
        
        outputs = torch.FloatTensor(torch.zeros((time, batch_size, 1, 256, 256))).to(device)

        states = None
        
        for t in range(time):
            x1, x2, x3, x4, x5, output, states = self.encoder(features[t], states)
            
        decoder_hidden = states
        layers = [x1, x2, x3, x4, x5]
        output_encoder = output
        
        input = torch.FloatTensor(torch.zeros(output.shape)).to(device)
        
        for t in range(1, time):
            decoder_output, decoder_hidden = self.decoder(layers[0], layers[1], layers[2], layers[3], layers[4], input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            input = (output_encoder if teacher_force else decoder_hidden[0])
            
        return outputs



import torch.nn.functional as F

from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 16) #16
        self.down1 = down(16, 32)#16, 32
        self.down2 = down(32, 48)#32, 48
        self.down3 = down(48, 64)#48,64 
        self.down4 = down(64, 80)#64, 80
        self.up1 = up(144, 64) #160, 48
        self.up2 = up(112, 48)
        self.up3 = up(80, 32)
        self.up4 = up(48, 16)
        self.outc = outconv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # x = F.sigmoid(x)
        return x
      