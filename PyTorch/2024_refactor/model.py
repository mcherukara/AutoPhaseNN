import torch
import torch.nn as nn
import params
import numpy as np

nconv = 32

class recon_model(nn.Module):

  H,W = 32,32

  def __init__(self):
    super(recon_model, self).__init__()
    
    self.sw_thresh = params.INIT_SW
    
    self.encoder = nn.Sequential( # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
      nn.Conv3d(in_channels=1, out_channels=nconv, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv),
      nn.Conv3d(in_channels=nconv, out_channels=nconv * 2, kernel_size=3, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv * 2),
    
      nn.Conv3d(nconv* 2, nconv*2, 3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv*2),
      nn.Conv3d(nconv*2, nconv*4, 3, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv* 4),
        
      nn.Conv3d(nconv*4, nconv*4, 3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv*4),
      nn.Conv3d(nconv*4, nconv*8, 3, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv*8),
        
      nn.Conv3d(nconv*8, nconv*8, 3, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv*8),
      )
    
    self.decoder1 = nn.Sequential(
      nn.Conv3d(nconv*8, nconv*4, 3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv*4),
      nn.Conv3d(nconv*4, nconv*4, 3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv*4),
      nn.Upsample(scale_factor=2, mode='trilinear'),
    
      nn.Conv3d(nconv*4, nconv*2, 3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv*2),
      nn.Conv3d(nconv*2, nconv*2, 3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv*2),
      nn.Upsample(scale_factor=2, mode='trilinear'), 
       
      nn.Conv3d(nconv*2, nconv, 3, stride=1, padding=1),  
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv),
      nn.Conv3d(nconv, nconv, 3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv),
      nn.Upsample(scale_factor=2, mode='trilinear'),
 
      nn.Conv3d(nconv, 1, 3, stride=1, padding=1),
      nn.Sigmoid() #Amplitude model
      )
    
    self.decoder2 = nn.Sequential(
      nn.Conv3d(nconv*8, nconv*4, 3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv*4),
      nn.Conv3d(nconv*4, nconv*4, 3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv*4),
      nn.Upsample(scale_factor=2, mode='trilinear'),
    
      nn.Conv3d(nconv*4, nconv*2, 3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv*2),
      nn.Conv3d(nconv*2, nconv*2, 3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv*2),
      nn.Upsample(scale_factor=2, mode='trilinear'), 
       
      nn.Conv3d(nconv*2, nconv, 3, stride=1, padding=1),  
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv),
      nn.Conv3d(nconv, nconv, 3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01),
      nn.BatchNorm3d(nconv),
      nn.Upsample(scale_factor=2, mode='trilinear'),
 
      nn.Conv3d(nconv, 1, 3, stride=1, padding=1),
      nn.Tanh() #Phase model
      )


  def forward(self,x):
    x1 = self.encoder(x)
    
    amp = self.decoder1(x1)
    ph = self.decoder2(x1)
    
    #Normalize amp to max 1 before applying support
    
    amp = torch.clip(amp, min=0, max=1.0)
    
    #Apply the support to amplitude
    mask = torch.tensor([0,1],dtype=amp.dtype, device=amp.device)
    amp = torch.where(amp<self.sw_thresh,mask[0],amp)
    
    #Restore -pi to pi range
    ph = ph*np.pi #Using tanh activation (-1 to 1) for phase so multiply by pi

    #Pad the predictions to 2X
    pad = nn.ConstantPad3d(int(self.H/2),0)
    amp = pad(amp)
    ph = pad(ph)

    #Create the complex number
    complex_x = torch.complex(amp*torch.cos(ph),amp*torch.sin(ph))

    #Compute FT, shift and take abs
    y = torch.fft.fftn(complex_x,dim=(-3,-2,-1))
    y = torch.fft.fftshift(y,dim=(-3,-2,-1)) #FFT shift will move the wrong dimensions if not specified
    y = torch.abs(y)
    
    #Normalize to scale_I
    if params.scale_I>0:
        max_I = torch.amax(y, dim=[-1, -2, -3], keepdim=True)
        y = params.scale_I*torch.div(y,max_I+1e-6) #Prevent zero div
    
    #get support for viz
    support = torch.zeros(amp.shape,device=amp.device)
    support = torch.where(amp<self.sw_thresh,mask[0],mask[1])

    return y, complex_x, amp, ph, support