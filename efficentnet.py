import torch
import torch.nn as nn
from math import ceil
from torchinfo import summary
# expand_ratio, repeats, channels, kernel_size, stride
base_line = [
   (1, 1, 16, 3, 1),
   (6, 2, 24, 3, 2),
   (6, 2, 40, 5, 2),
   (6, 3, 80, 3, 2),
   (6, 3, 112, 5, 1),
   (6, 4, 192, 5, 2),
   (6, 1, 320, 3, 1),
]

# 'b_n': (alpha**phi, beta**phi, res, drop_rate)
phi_value = {
   'b0':(1.0, 1.0, 224, 0.2),
   'b1':(1.1, 1.0, 240, 0.2),
   'b2':(1.2, 1.1, 288, 0.3),
   'b3':(1.4, 1.2, 320, 0.3),
   'b4':(1.8, 1.4, 384, 0.4),
   'b5':(2.2, 1.6, 456, 0.4),
   'b6':(2.6, 1.8, 528, 0.5),
   'b7':(3.1, 2.0, 600, 0.5),
}

class SqueezeExcitation(nn.Module):
   def __init__(self, in_channels, reduced_dim):
      super().__init__()
      self.se = nn.Sequential(
         nn.AdaptiveAvgPool2d(output_size=1),
         nn.Conv2d(in_channels=in_channels, out_channels=reduced_dim, kernel_size=1),
         nn.SiLU(inplace=True),
         nn.Conv2d(in_channels=reduced_dim, out_channels=in_channels, kernel_size=1),
         nn.Sigmoid(),
      )
   
   def forward(self, x):
      return x * self.se(x)

class MBConv(nn.Module):
   def __init__(
      self,
      expand_ratio,
      depth_wise_kernel_size,
      stride,
      padding,
      in_channel,
      out_channel,
      reduction = 4, # squeeze excitation
      survival_prob = 0.8, # for drop skip-connection (stochastic depth)
      ):
      
      super().__init__()
      
      self.survival_prob = survival_prob
      self.use_residual = in_channel == out_channel and stride == 1
      
      # expand_ration
      expanded_dim = in_channel * expand_ratio
      self.expand = in_channel != expanded_dim # Check expand_ratio =? 1
      
      # reduction
      reduced_dim = int(expanded_dim/reduction)
      
      if self.expand: # Expand_ration != 1 (6)
         self.expand_conv = nn.Sequential(
            nn.Conv2d(
            in_channels=in_channel,
            out_channels=expanded_dim,
            kernel_size=1,
            stride = 1,
            padding=0,),
            
            nn.BatchNorm2d(expanded_dim),
            nn.SiLU(True)
         )
         
      self.depth_wise = nn.Conv2d(
         in_channels=expanded_dim, out_channels=expanded_dim, kernel_size=depth_wise_kernel_size, groups=expanded_dim
         , stride = stride, padding=padding
      )
      
      self.se = SqueezeExcitation(in_channels=expanded_dim, reduced_dim=reduced_dim)
      
      self.last_conv = nn.Sequential(
         nn.Conv2d(in_channels=expanded_dim, out_channels=out_channel, kernel_size=1),
         nn.BatchNorm2d(num_features=out_channel)
      )
      
   def forward(self, x):
      original_x = x
      if self.expand: # Allow to expand
         x = self.expand_conv(x)
      x = self.depth_wise(x)
      x = self.se(x)
      x = self.last_conv(x)
      
      if self.use_residual:
         # Check for dropping skip-connection
         random_value = torch.rand(1).item()
         if random_value < self.survival_prob: # Allow to +
            return x + original_x
         
      return x
   
class EfficentNet(nn.Module):
   def __init__(
      self,
      version,
      num_classes
      ):
      
      super().__init__()
      depth_factor, width_factor, res, drop_rate = self.get_scale_rate(version)
      last_channels = ceil(1280*width_factor)
      
      self.features = self.create_features(depth_factor, width_factor, last_channels)
      self.pool = nn.AdaptiveAvgPool2d(output_size=1)
      self.classifier = nn.Sequential(
         nn.Dropout(p=drop_rate),
         nn.Linear(in_features=last_channels, out_features=num_classes),
         nn.Softmax(dim=1)
      )
      
   def get_scale_rate(sefl, version):
      return phi_value[version]
   
   def create_features(self, depth_factor, width_factor, last_channels):
      channels = int(32*width_factor)
      features = nn.ModuleList()
      features.append(
         nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, stride=2, padding=1)
      )
      in_channels = channels
      

      # Iter each stage      
      for expand_ratio, repeat, channel, kernel_size, stride in base_line:
         out_channel = 4*ceil((channel * width_factor)/4)
         block_repeats = ceil(repeat*depth_factor)
         # Iter each block in STAGE
         for block in range(block_repeats):
            features.append(
               MBConv(
                  expand_ratio=expand_ratio,
                  depth_wise_kernel_size=kernel_size,
                  stride=stride if block == 0 else 1,
                  padding=kernel_size//2,
                  in_channel=in_channels,
                  out_channel=out_channel
               )
            )      
            in_channels = out_channel
         
      features.append(
         nn.Conv2d(
            in_channels=in_channels, out_channels=last_channels, kernel_size=1, stride=1, padding=0
         )
      )
      
      return nn.Sequential(*features)
   
   def forward(self, x):
      x = self.features(x)
      x = self.pool(x)
      x = torch.flatten(x, 1)
      x = self.classifier(x)
      return x
   
if __name__ == "__main__":
   model = EfficentNet(version='b0', num_classes=10)
   x = torch.randn(1, 3, 224, 224)
   y = model(x)
   print(y.shape)
   summary(model, input_size=x.shape)
         

