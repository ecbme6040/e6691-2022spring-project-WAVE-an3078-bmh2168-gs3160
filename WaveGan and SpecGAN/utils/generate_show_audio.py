from IPython.display import Audio 
from IPython.core.display import display
import torchaudio
import torch
import numpy as np
from utils.utils import *
from torch.utils.data import DataLoader,Dataset
from utils.wavegan import *
from utils.specgan import *
from matplotlib import pyplot as plt
from torchaudio import transforms


import matplotlib.pyplot as plt


griffinLim = transforms.GriffinLim(
        n_fft=255,
        win_length=255,
        hop_length=128)

SAMPLING_RATE=16000


# show spectrograms in matplotlib plots from generated specgan batch
'''
show spectrograms in matplotlib plots from generated specgan batch
'''
def show_spectrograms(audio_batch):
   
    for i in range(audio_batch.shape[0]):
        plt.imshow(audio_batch[i][0].numpy(), interpolation='nearest')
        plt.show()

        
# generate in browser audio from generated specgan batch, after rescaling
def listen_spectrograms(audio_batch,autoplay=False):
    '''
    generate in browser audio from generated specgan batch, after rescaling
    Autoplay:autoplay audio, false is multiple clips
    '''
    for i in range(audio_batch.shape[0]):
        sound=audio_batch[i][0]
        sound=torch.clip(sound, min=0)
        sound=griffinLim(sound)
        display(Audio(sound, rate=16000, autoplay=autoplay))



def listen_specgan(model_path,dataset,denoise=0,batch_size=10,plot=True,d=64,c=1,device='cpu'):
    '''
    Listen to generated clips from specgan model.
    model_path: string, model path
    batch_size: number of generated clips
    dataset: sc09, drum or piano, sets different scalings that are dataset specific
    denoise: has to be a positive float. Is there is a lot of noise, it changes the rescaling and can help
    plot:boolean
    d: model capacity
    c: channels (mono by default)
    device: cpu or cuda
    '''
    # different std and mean for different datasets
    
    #sc09
    mean=0.7270409535254246
    std=16.963365074677686
    
    if dataset=='drum':
        mean=1.180594260845436
        std=37.88559632326802
    elif dataset=='piano':
        mean=0.22374595802679448
        std=2.228093965844948
        
    gen = SpecGenerator(d=d, c=c ,inplace=True).to(device)
    gen.eval()
    gen.load_state_dict(torch.load(model_path,map_location=device))
    noise = torch.randn(batch_size, 100).to(device)
    audio_batch = gen(noise)
    
    # reverse standardization
    audio_batch = (audio_batch * (3 * std)) + mean - denoise # if artifacts denoise>0 can help.
    
    
    listen_spectrograms(audio_batch.detach())
    if plot:
        show_spectrograms(audio_batch.detach())


def plot_audio_batch(audio_batch):
    for audio in audio_batch:
        plt.plot(audio[0].detach())
        plt.show()
        
def audio_player(audio_batch,autoplay=False):
    
    for i in range(audio_batch.shape[0]):
        display(Audio(audio_batch[i].detach().numpy(), rate=SAMPLING_RATE, autoplay=autoplay))


        
def listen_wavegan(model_path,batch_size=10,plot=True,d=64,c=1,device='cpu'):
    '''
    Listen to generated clips from wavegan model.
    model_path: string, model path
    batch_size: number of generated clips
    plot:boolean
    d: model capacity
    c: channels (mono by default)
    device: cpu or cuda
    '''
    wave_gen = WaveGenerator(d=d, c=c ,inplace=True).to(device)
    wave_gen.eval()
    wave_gen.load_state_dict(torch.load(model_path,map_location=device))
    noise = torch.randn(batch_size, 100).to(device)
    audio_batch = wave_gen(noise)
    audio_player(audio_batch)
    if plot:
        plot_audio_batch(audio_batch)
        