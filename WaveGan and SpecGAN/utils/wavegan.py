import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
def get_number_parameters(model):
    
    """
    Prints the number of trainable parameters of the model
    
    """
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params,'trainable parameters')


def initialize_weights(m,debug=False):
    """
    Weights initializer: initialize weights with mean=0 and std= .02 like in DCGAN
    
    debug=True prints if the layer has been initialized
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('BatchNorm') != -1:
        nn.init.constant_(m.bias.data, 0)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if debug:
            print('init',classname)
    else:
        if debug:
            print('noinit',classname)
            
            
            
##################   Generator    ##########################   


       
class WaveGenerator(nn.Module):
    """
    Generator for WaveGAN
    
    d: model size (default 64)
    c: number of channels in the data (default 1)
    inplace: boolean (defaul True) #arg for the Relu 
    
    See page 15 of the Wavegan paper https://openreview.net/pdf?id=ByMVTsR5KQ 
    
    """
    def __init__(self,d=64, c=1 ,inplace=True):
        super(WaveGenerator, self).__init__()
        self.d=d # model size
        self.c=c # = 1 in the paper
        self.dense1= nn.Linear(100, 256*self.d)
        self.padding=11
        self.seq = nn.Sequential(
            # input is dense(Z), going into a convolution
            nn.ReLU(inplace), #out (n,16,16d)
            nn.ConvTranspose1d( 16*self.d, self.d * 8, 25, 4, self.padding,1, bias=True), # (25,16d,4d) | (n,64,8d)
            nn.ReLU(inplace), #no batch norm
         
            nn.ConvTranspose1d(self.d * 8, self.d * 4, 25, 4, self.padding,1, bias=True),#(25, 8d, 4d)| (n, 256, 4d)
            nn.ReLU(inplace),
        
            nn.ConvTranspose1d( self.d * 4,self.d * 2, 25, 4, self.padding,1, bias=True),#(25, 4d, 2d) | (n, 1024, 2d)
            nn.ReLU(inplace),
      
            nn.ConvTranspose1d( self.d * 2, self.d, 25, 4, self.padding,1, bias=True), #(25, 2d, d) | (n, 4096, d)
            nn.ReLU(inplace),
            nn.ConvTranspose1d( self.d, self.c, 25, 4, self.padding,1, bias=True),#(25, d, c) | (n, 16384, c)
            nn.Tanh() # as suggested       
        )

    def forward(self, x):
        #input (n,100)
        x=self.dense1(x) # output (n,256*d)
        x=torch.reshape(x, (-1,16*self.d,16)) # output (n,16,16d)
        
        return self.seq(x) #(n, 16384, c), c=1

    
################### phase suffling ##########################

class PhaseShuffling(nn.Module):
    """
    
    PhaseShuffling layer: shifts the features by a random int value between [-n,n]
    n: shift factor
    
    # paper code in tf https://github.com/chrisdonahue/wavegan/blob/master/wavegan.py
    """
    def __init__(self, n):
        super(PhaseShuffling, self).__init__()
        self.n = n

    def forward(self, x):
        #x:(n batch,channels,xlen)
        if self.n==0:
            return x
        shifts =  int(torch.Tensor(1).random_(0, 2*self.n + 1)) - self.n # same shuffle for data in batch
       
        if shifts > 0:
            return F.pad(x[..., :-shifts], (shifts, 0), mode='reflect')
        else:
            return F.pad(x[..., -shifts:], (0, -shifts), mode='reflect')



##################     Critic      ##########################
    
# See page 15 of the Wavegan paper https://openreview.net/pdf?id=ByMVTsR5KQ        
class WaveDiscriminator(nn.Module):
    """
    Generator for WaveGAN
    
    d: model size (default 64)
    c: number of channels in the data (default 1)
    inplace: boolean (defaul True) #arg for the Leaky Relu  
    
    See page 15 of the Wavegan paper https://openreview.net/pdf?id=ByMVTsR5KQ 
    
    """
    def __init__(self,d=64, c=1,inplace=True):
        super(WaveDiscriminator, self).__init__()
     
        self.d=d # model size
        self.c=c # = 1 in the paper
        self.padding=11
        leak=0.2
        
        
        self.dense= nn.Linear(256*self.d, 1)
        self.seq = nn.Sequential(
            # input is audio or WaveGenerator(z)
            nn.Conv1d( self.c, self.d, 25, 4, self.padding, bias=True), #(n,4096,d)
            nn.LeakyReLU(leak,inplace=inplace),
            PhaseShuffling(n=2),
            
            nn.Conv1d( self.d, 2*self.d, 25, 4, self.padding, bias=True),  #(n,1024,2d)
            nn.LeakyReLU(leak,inplace=inplace),
            PhaseShuffling(n=2),
            
            nn.Conv1d( self.d * 2, self.d * 4, 25, 4, self.padding, bias=True),  #(n,256,4d)
            nn.LeakyReLU(leak,inplace=inplace),
            PhaseShuffling(n=2),
            
            nn.Conv1d( self.d * 4, self.d * 8, 25, 4, self.padding, bias=True),  #(n,64,8d)
            nn.LeakyReLU(leak,inplace=inplace),
            PhaseShuffling(n=2),
            
            nn.Conv1d( self.d * 8, self.d * 16, 25, 4, self.padding, bias=True),  #(n,16,16d)
            nn.LeakyReLU(leak,inplace=inplace)
        )
               

    def forward(self, x):
        x=self.seq(x)
        x=torch.reshape(x, (-1,256*self.d)) 
        #print(x.shape)
        return self.dense(x) 


def testing():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the generator
    waveG = WaveGenerator().to(device)
    #initialize wights
    waveG.apply(initialize_weights)
    print(waveG)

    # Create the Discriminator
    waveD = WaveDiscriminator().to(device)
    #initialize wights
    waveD.apply(initialize_weights)
    print(waveD)


    N, in_channels, L = 8, 1, 16384
    noise_dim = 100
    x = torch.randn((N, in_channels,L))
    waveD = WaveDiscriminator()
    print(waveD(x).shape)
    waveG = WaveGenerator(d=64,c=1)
    z = torch.randn((N, 100))
    print(waveG(z).shape)
    print(waveD(waveG(z)))