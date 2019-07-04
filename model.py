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
    
    def set_encoder(self, x):
        x = x.cuda()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        output, hidden = self.cell(x5, None)
        return x1, x2, x3, x4, x5, output, [output, hidden]
        

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        x1, x2, x3, x4, x5, output, [output, hidden] = self.set_encoder(x)
        
        return x1, x2, x3, x4, x5, output, [output, hidden]

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
        
    def set_decoder(self, x1, x2, x3, x4, x5, encoder_output, states):
        hidden_h, hidden_c = self.cell(encoder_output, states)
        x = self.up0(hidden_c, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x, [hidden_h, hidden_c]

    def forward(self, x1, x2, x3, x4, x5, output, states):
        ''' Forward pass through the network, returns the output logits '''
        
        x, states = self.set_decoder(x1, x2, x3, x4, x5, output, states)
            
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
        
        for t in range(time):
            x1, x2, x3, x4, x5, output, states = self.encoder(features[t])
            
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
      