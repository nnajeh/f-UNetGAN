

from utils.libraries import *
from utils.dataset import *
from utils.spectral_Normalization import *
from utils.attention import *
from utils.parameters import *



#### Generator Residual Block
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


      
      
 
#### Discriminator Residual Block
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

















#### Encoder Residual Block
class E_ResidualConv(nn.Module):
  def __init__(self, in_channels, out_channels,  preactivation=False,  downsample=None, bn=None):
    
    super(E_ResidualConv, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
   # self.hidden_channels = self.out_channels if wide else self.in_channels

    self.preactivation = preactivation
    self.activation = nn.LeakyReLU(0.2)
    
    self.downsample = downsample
    self.bn = bn
        
    self.bn1 =  nn.InstanceNorm2d(out_channels)
    self.bn2 =  nn.InstanceNorm2d(out_channels)

    self.downsample_fn = nn.AvgPool2d(2,2)
    
        
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
    if self.bn:
        x = self.bn1(x)
        
    if self.preactivation:
      h = F.relu(x)
    else:
      h = x    
    

    h = self.conv1(h)
    
    if self.bn:
        h = self.bn2(h)
    h = self.conv2(self.activation(h))
    
    if self.downsample:
      h = self.downsample_fn(h)     
        
    return h + self.shortcut(x)



