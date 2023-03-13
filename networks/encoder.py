

from utils.libraries import *
from utils.dataset import *
from utils.spectral_Normalization import *
from utils.attention import *
from utils.parameters import *


class Encoder(nn.Module):
    def __init__(self, channel =3, filters=[64, 128, 256, 512, 1024], z_dim=100, attention=True):
        super(Encoder, self).__init__()

        self.input_layer = nn.utils.spectral_norm(nn.Conv2d(channel, filters[0], 3,1,1))
        
        self.residual_conv_1 = E_ResidualConv(filters[0], filters[1], preactivation=True, downsample=True, bn=True, attention=attention)
                
        self.residual_conv_2 = E_ResidualConv(filters[1], filters[2], preactivation=True, downsample=True, bn=True, attention=attention)
        
        self.residual_conv_3 = E_ResidualConv(filters[2], filters[3],  preactivation=True, downsample=True, bn=True, attention=attention)
                
        self.residual_conv_4 = E_ResidualConv(filters[3], filters[4],  preactivation=True, downsample=True, bn=True, attention=attention)
        
        self.output_layer = E_ResidualConv(filters[4], filters[4],  preactivation=True, downsample=True, bn=True, attention=attention)
              
        self.linear =  nn.utils.spectral_norm(nn.Linear(filters[4]*4*4, z_dim))
   
        
    def forward(self, x):
        output = self.input_layer(x)
        output = self.residual_conv_1(output)
        
        output = self.residual_conv_2(output)
        
        output = self.residual_conv_3(output)

        output = self.residual_conv_4(output)

        output = self.output_layer(output)
        
        output = output.view(output.shape[0], -1)
        
        output = self.linear(output)

        return torch.tanh(output) 
