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

class SpecGenerator(nn.Module):
    """
    Generator for SpecGAN
    
    d: model size (default 64)
    c: number of channels in the data (default 1)
    inplace: boolean (defaul True) #arg for the Relu 
    
    See page 16 of the paper https://openreview.net/pdf?id=ByMVTsR5KQ 
    
    """
    def __init__(self,d=64, c=1 ,inplace=True):
        super(SpecGenerator, self).__init__()
        self.d=d # model size
        self.c=c # = 1 in the paper
        self.dense1= nn.Linear(100, 256*self.d)
        self.padding=2
        self.seq = nn.Sequential(
            # input is dense(Z), going into a convolution
            nn.ReLU(inplace), #out (n,4,4,16d)
            nn.ConvTranspose2d( 16*self.d, self.d * 8, 5, 2, self.padding,1, bias=True), 
            nn.ReLU(inplace), #no batch norm
         
            nn.ConvTranspose2d(self.d * 8, self.d * 4, 5, 2, self.padding,1, bias=True),
            nn.ReLU(inplace),
        
            nn.ConvTranspose2d( self.d * 4,self.d * 2, 5, 2, self.padding,1, bias=True),
            nn.ReLU(inplace),
      
            nn.ConvTranspose2d( self.d * 2, self.d, 5, 2, self.padding,1, bias=True), 
            nn.ReLU(inplace),
            nn.ConvTranspose2d( self.d, self.c, 5, 2, self.padding,1, bias=True),
            nn.Tanh() # as suggested       
        )
        
        
        
       

        self.g=nn.Tanh() # as suggested  

    def forward(self, x):
        #input (n,100)
        x=self.dense1(x) # output (n,256*d)
        x=torch.reshape(x, (-1,16*self.d,4,4)) 
       
        return self.seq(x) #(n, 16384, c), c=1

    
    
##################     Critic      ##########################
    
# See page 16 of the paper https://openreview.net/pdf?id=ByMVTsR5KQ        
class SpecDiscriminator(nn.Module):
    """
    Generator for SpecGAN
    
    d: model size (default 64)
    c: number of channels in the data (default 1)
    inplace: boolean (defaul True) #arg for the Leaky Relu  
    
    See page 16 of the paper https://openreview.net/pdf?id=ByMVTsR5KQ 
    
    """
    def __init__(self,d=64, c=1,inplace=True):
        super(SpecDiscriminator, self).__init__()
     
        self.d=d # model size
        self.c=c # = 1 in the paper
        self.padding=2
        leak=0.2
        
        
        self.dense= nn.Linear(256*self.d, 1)
        self.seq = nn.Sequential(
            # input is audio or WaveGenerator(z)
            nn.Conv2d( self.c, self.d, 5, 2, self.padding, bias=True), #(n,4096,d)
            nn.LeakyReLU(leak,inplace=inplace),
            
            nn.Conv2d( self.d, 2*self.d, 5, 2, self.padding, bias=True),  #(n,1024,2d)
            nn.LeakyReLU(leak,inplace=inplace),
            
            nn.Conv2d( self.d * 2, self.d * 4, 5,2, self.padding, bias=True),  #(n,256,4d)
            nn.LeakyReLU(leak,inplace=inplace),
            
            nn.Conv2d( self.d * 4, self.d * 8, 5, 2, self.padding, bias=True),  #(n,64,8d)
            nn.LeakyReLU(leak,inplace=inplace),
            
            nn.Conv2d( self.d * 8, self.d * 16, 5, 2, self.padding, bias=True),  #(n,16,16d)
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
    specG = SpecGenerator().to(device)
    #initialize wights
    specG.apply(initialize_weights)
    print(specG)

    # Create the Discriminator
    specD = SpecDiscriminator().to(device)
    #initialize wights
    specD.apply(initialize_weights)
    print(specD)


    N, in_channels, W,H = 8, 1, 128,128
    noise_dim = 100
    x = torch.randn((N, in_channels,W,H ))
    specD = SpecDiscriminator()
    print(specD(x).shape)
    specG = SpecGenerator(d=64,c=1)
    z = torch.randn((N, 100))
    print(specG(z).shape)
    print(specD(specG(z)))
