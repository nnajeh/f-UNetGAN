

from utils.libraries import *
from utils.dataset import *
from utils.spectral_Normalization import *
from utils.attention import *
from utils.parameters import *


## Generator 

class GResidualBlock(nn.Module):
    '''
    GResidualBlock Class
    Values:
    c_dim: the dimension of conditional vector [c, z], a scalar
    in_channels: the number of channels in the input, a scalar
    out_channels: the number of channels in the output, a scalar
    '''

    def __init__(self,in_channels, out_channels, upsample = True):
        super().__init__()

        self.conv1 = SNConv2d(in_channels, out_channels, kernel_size=3, padding= 1)
        self.conv2 = SNConv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.bn1 = nn.InstanceNorm2d(in_channels)
        self.bn2 = nn.InstanceNorm2d(out_channels)

        self.activation = nn.ReLU()
        self.upsample = upsample
        self.upsample_fn = nn.Upsample(scale_factor=2)     # upsample occurs in every gblock

        self.mixin = (in_channels != out_channels) or upsample
        if self.mixin:
            self.conv_mixin = SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)

            
    def forward(self, x):
        
        h = self.bn1(x)
        h = self.activation(h)
        h = self.upsample_fn(h)
        x = self.upsample_fn(x)
        h = self.conv1(h)

        h = self.bn2(h)
        h = self.activation(h)
        h = self.conv2(h)

        if self.mixin:
            x = self.conv_mixin(x)

        return h + x


# In[9]:


class Generator(nn.Module):
    '''
    Generator Class
    Values:
    z_dim: the dimension of random noise sampled, a scalar
    shared_dim: the dimension of shared class embeddings, a scalar
    base_channels: the number of base channels, a scalar
    bottom_width: the height/width of image before it gets upsampled, a scalar
    n_classes: the number of image classes, a scalar
    '''
 
    def __init__(self, base_channels=64, z_dim=100, channel=3, n_classes=2, upsample = True, G_init='ortho'):
        super().__init__()
        
        self.init = G_init

        self.z_dim = z_dim
 
        self.proj_z = SNLinear(z_dim, base_channels*16 * 4 ** 2)
 
        # Can't use one big nn.Sequential since we are adding class+noise at each block
        self.g_blocks = nn.ModuleList([
                nn.ModuleList([
                    GResidualBlock( 16 * base_channels, 16 * base_channels),
                ]),
                nn.ModuleList([
                    GResidualBlock( 16 * base_channels, 8 * base_channels),
                ]),
            
                nn.ModuleList([
                    GResidualBlock( 8 * base_channels, 4 * base_channels),
                ]),
                nn.ModuleList([
                    GResidualBlock( 4 * base_channels, 2 * base_channels),
                    CBAMBlock(2* base_channels),


                ]),
                nn.ModuleList([
                    GResidualBlock(2 * base_channels, 1* base_channels),
                    CBAMBlock(1* base_channels),

                ])
           
        ])
        
        # Using non-spectral Norm
        self.proj_o = nn.Sequential(
            nn.BatchNorm2d(base_channels),
            nn.Conv2d(base_channels, channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
            
        )
        

    
    
    def forward(self, z):
        '''
        z: random noise with size self.z_dim
        y: class embeddings with size self.shared_dim
            = NOTE =
            y should be class embeddings from self.shared_emb, not the raw class labels
        '''
        #First Linear Layer
        h = self.proj_z(z.contiguous())
        h = h.view(-1,1024, 4, 4)
        


        # Loop over blocks
        for index, blocklist in enumerate(self.g_blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h)

        # Project to 3 RGB channels with tanh to map values to [-1, 1]
        h = self.proj_o(h)
 
        return   torch.tanh(h)


## Discriminator


#### Discriminator Residual Block
# Residual block for the discriminator
class ResidualConv(nn.Module):
  def __init__(self, in_channels, out_channels,  preactivation=False,  downsample=None, bn=None, attention=None):
    
    super(ResidualConv, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    
    self.preactivation = preactivation
    self.activation = nn.LeakyReLU(0.2)
    
    self.downsample = downsample
    self.bn = bn
    
    self.attention=attention

    if self.bn==True:
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    elif self.bn==False:
        self.bn3 =  nn.InstanceNorm2d(out_channels)
        self.bn4 =  nn.InstanceNorm2d(out_channels)

    else:
        self.bn5=None
        
          
    if self.attention == True:
        self.attention1 = CBAMBlock(out_channels)

    elif self.attention == False:
        self.attention2 = AttentionBlock(out_channels)

    else:
        self.attention3 = None
        
    self.downsample_fn = nn.MaxPool2d(2,2)
    
        
    # Conv layers
    self.conv1 = SNConv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True)
    
    self.conv2 = SNConv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1,  bias=True)

    
    self.learnable_sc = True if (in_channels != out_channels) or self.downsample else False
    
    if self.learnable_sc:
        self.conv_sc = SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    
  def shortcut(self, x):
    if self.preactivation:
        if self.learnable_sc:
            x = self.conv_sc(x)
        if self.downsample:
            x = self.downsample_fn(x)
        
    else:  
        if self.downsample:
            x = self.downsample_fn(x)
        if self.learnable_sc:
            x = self.conv_sc(x)
    return x
    
    
    
  def forward(self, x):
    if self.bn ==True:
        x = self.bn1(x)
        
    elif self.bn ==False:
        x = self.bn3(x)
        
    else:
        x = x
        
    if self.preactivation:
        h = F.relu(x)
    else:
        h = x    
    

    h = self.conv1(h)
    
    if self.bn ==True:
        h = self.bn2(h)
    elif self.bn==False:
        h = self.bn4(h)
    else:
        h = h
        
        
    h = self.conv2(self.activation(h))
    
    if self.downsample:
        h = self.downsample_fn(h)     
        
            
    if self.attention ==True:
        h = self.attention1(h)
    elif self.attention == False:
        h = self.attention2(h)
    else :
        h = h
    
    h = h + self.shortcut (x)

    return h

##Discriminator Unet
class UNet(nn.Module):
    def __init__(self, channel= 3, filters=[64, 128, 256, 512, 1024], bn=None,  attention=None,  D_init='ortho', dim=64):
        super(UNet, self).__init__()
        
        self.init = D_init
            
        self.input_layer = ResidualConv(channel, filters[0], preactivation=False, downsample=False, 
                                        bn=False, attention=attention)
    
        self.residual_conv_1 = ResidualConv(filters[0], filters[1], preactivation=True, downsample=True,
                                          bn=False, attention=attention)
                
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], preactivation=True, downsample=True,
                                             bn=False, attention=attention)
        
         self.residual_conv_3 = ResidualConv(filters[2], filters[3], preactivation=True, downsample=True,
                                             bn=False, attention=attention)
        
        self.residual_conv_4 = ResidualConv(filters[3], filters[4], preactivation=True, downsample=True,
                                             bn=False, attention=attention)

      
        self.upsample_1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_residual_conv1 = ResidualConv(filters[4]+filters[3], filters[3],preactivation=False, downsample=False,
                                             bn=False, attention=None)
     
        self.upsample_2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_residual_conv2 = ResidualConv(filters[3]+filters[2], filters[2],preactivation=False, downsample=False,
                                             bn=False, attention=None)        

        self.upsample_3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_residual_conv3 = ResidualConv(filters[2]+filters[1], filters[1],preactivation=False, downsample=False,
                                            bn=False, attention=None)
     
        self.upsample_4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_residual_conv4 = ResidualConv(filters[1]+filters[0], filters[0],preactivation=False, downsample=False,
                                               bn=False, attention=None)
        
        
        
        
        # (64,128,128) --> (1,128,128)
        self.output_layer = nn.Conv2d(filters[0], 1,1,1)  #()
        
        self.linear = SNLinear(filters[4], 1)
        
        self.activation = nn.LeakyReLU(0.2)
    
                
                
                
    def extract_features_encoder(self, x0):
        x0 = x0.view(-1, channel, 128,128)

        conv1 = self.input_layer(x0)

        conv2 = self.residual_conv_1(conv1)

        conv3 = self.residual_conv_2(conv2)
        
        conv4 = self.residual_conv_3(conv3)

        conv5 = self.residual_conv_4(conv4)
        
        return  conv5, conv4, conv3, conv2, conv1


    def extract_features_decoder(self, conv5, conv4, conv3, conv2, conv1):
        
        x = self.upsample_1(conv5)   
        x = torch.cat([x, conv4], dim=1)
        x = self.up_residual_conv1(x)
        
        x = self.upsample_2(x) 
        x = torch.cat([x, conv3], dim=1)
        x = self.up_residual_conv2(x)
        
        x = self.upsample_3(x)     
        x = torch.cat([x, conv2], dim=1)
        x = self.up_residual_conv3(x)
        
        x = self.upsample_4(x) 
        x = torch.cat([x, conv1], dim=1)
        output_dec = self.up_residual_conv4(x)
        
        
        return output_dec
        
        
        
    def extract_all_features(self, x):
        features, conv3, conv2, conv1 , conv0 = self.extract_features_encoder(x)
        feats_encoder = torch.sum(self.activation(features), [2,3])

        feats = self.extract_features_decoder(features, conv3, conv2, conv1, conv0)

        return feats_encoder, feats
    
    
        
    def forward(self, x):
        feats_encoder, features = self.extract_all_features(x)

        feats_encoder = self.linear(feats_encoder)
        output1 = feats_encoder.view(-1)

        #output:(1,128,128)
        output2 = self.output_layer(features)
        output2 = output2.view(output2.size(0),1,128,128)

        return output1, output2  
