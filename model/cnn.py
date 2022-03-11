import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvNet(nn.Module):
  """
  easy cnn implementation
  """
  def __init__(self):
    super(ConvNet,self).__init__()
    self.conv1=nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3)
    self.fc1=nn.Linear(in_features=8*111*111,out_features=32)
    self.out=nn.Linear(in_features=32,out_features=2)

  def forward(self,l):
    l=self.conv1(l)
    l=F.relu(l)
    l=F.max_pool2d(l,kernel_size=2)

    l=l.reshape(-1,8*111*111)
    l=self.fc1(l)
    l=self.out(l)

    return l