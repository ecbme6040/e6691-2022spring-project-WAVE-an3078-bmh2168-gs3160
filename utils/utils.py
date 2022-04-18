import glob
import librosa
import numpy as np
import os 
import torch
from torch.utils.data import Dataset
from scipy import signal
import matplotlib.pyplot as plt
from torch.autograd import grad

def get_number_parameters(model):
       
    """
    Prints the number of trainable parameters of the model
    
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params,'trainable parameters')
    
    

def plot_spectrogram(audio_numpy,fs=41000):
    
    """
    plots the spectrogram
    
    audio_numpy: audio sample in numpy array
    fs: sample rate frequency
    
    """
    # params : https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    f, t, Sxx = signal.spectrogram(audio_numpy, fs)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def get_all_paths(data_path,audio_extension):
    
    """
    Returns list of paths in 'data_path' that ends with the extension 'audio_extension'
    
    data_path: string
    audio_extension: string
    
    """
    files = glob.glob(data_path + '/**/*.'+audio_extension, recursive=True)
    return files

#################### Loading audio files
    
#loads from path to numpy
def load_audio_file(path,sample_rate=16000,number_samples=16384,std=False):
    
    """
    loads audio file from path, returns the normalized audio with fixed padded length (float32 numpy array)
    
    path: audio sample path
    sample_rate: sample rate of returned audio sample array
    number_samples: lengths of the return audio sample array
    std: boolean, if true normalize the audio sample
    """
    debug=False
    try:
        if debug:
            print(path)
        
        audio, _ = librosa.load(path, sr=sample_rate)
        if debug:
            print(audio)
        #'normalizing'
        if std:
            audio -=np.mean(audio)
            audio /= np.std(audio)
            
        if debug:
            print('shape',audio.shape)
        
        
        lenght = len(audio)
    except Exception as e:
        
        raise e

    # padding
    if lenght < number_samples: 
        pad = number_samples - lenght
        left = pad // 2
        right = pad - left
        audio = np.pad(audio, (left, right), mode="constant")
    
    
    # fixed length audio samples
    if lenght != number_samples :
        start_index = np.random.randint(0, (lenght - number_samples) // 2)
         
        audio=audio[start_index:start_index + number_samples]
    return audio.astype("float32")





######################### Dataset Class

class AudioDataset(Dataset):
    """
    Dataset class implementation for the audio data
    
    data_path: directory of audio data, dataset will contain all audio within the data_path (recursively)
    sample_rate: sample rate of returned audio sample array
    number_samples: lengths of the return audio sample array
    extension: extension of the audio data, only the files that matches the extension will be considered.
    std: boolean, if true normalize the audio sample
    """   
    def __init__(self, data_path,sample_rate=16000,number_samples=16384,extension='wav',std=False):
        self.std=std
        self.sample_rate=sample_rate
        self.number_samples=number_samples
        self.extension=extension
        self.data_path=data_path
        self.all_paths=get_all_paths(self.data_path,extension)
        self.n_samples=len(self.all_paths)
        
    def __getitem__(self, index):
       
        return torch.from_numpy(load_audio_file(self.all_paths[index],sample_rate=self.number_samples,number_samples=self.number_samples,std=self.std))[None,:]
   
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

    
    
    
##################### for training ####################

def wasserstein_loss(discriminator, real, generated,device,LAMBDA = 10):
    '''
    Wasserstein loss with Gradient Penalty 
    Check https://arxiv.org/pdf/1704.00028.pdf for pseudo code
    LAMBDA: penalty parameter (=10 in the paper)
    
    '''

    batch_size,C,L=real.shape
    eps=torch.rand((batch_size, 1, 1)).repeat(1, C, L).to(device)

    interpolated_sound = (1 - eps) * real + (eps) * generated

    mixed_score = discriminator(interpolated_sound)

    ones = torch.ones(mixed_score.size()).to(device)

    gradients = grad(
        outputs=mixed_score,
        inputs=interpolated_sound,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # calculate gradient penalty
    grad_penalty = (
        LAMBDA
        * ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
    )

    # normal Wasserstein loss
    loss_GP = discriminator(generated).mean() - discriminator(real).mean()
    # adding gradient penalty with param LAMBDA (=10 in paper)
    loss_GP += grad_penalty
    return loss_GP

