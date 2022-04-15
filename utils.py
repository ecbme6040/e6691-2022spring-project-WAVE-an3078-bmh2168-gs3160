import glob
import librosa
import numpy as np
import os 
import torch
from torch.utils.data import Dataset
from scipy import signal
import matplotlib.pyplot as plt


# Prints the number of trainable params in a model.
def get_number_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params,'trainable parameters')
    
    
# plots spectrogram
def plot_spectrogram(audio_numpy,fs=41000):

    # params : https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    f, t, Sxx = signal.spectrogram(audio_numpy, fs)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
plot_spectrogram(data.numpy()[0])
def get_all_paths(data_path,audio_extension):
    
    files = glob.glob(data_path + '/**/*.'+audio_extension, recursive=True)
    return files

#################### Loading audio files
    
#loads from path to numpy
def load_audio_file(path,sample_rate=16000,number_samples=16384,std=False):
    try:
        print(path)
        audio, _ = librosa.load(path, sr=sample_rate)
        print(audio)
        #'normalizing'
        if std:
            audio -=np.mean(audio)
            audio /= np.std(audio)
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
    
    def __init__(self, data_path,sample_rate=16000,number_samples=16384,extension='wav',std=False):
        self.std=std
        self.sample_rate=sample_rate
        self.number_samples=number_samples
        self.extension=extension
        self.data_path=data_path
        self.all_paths=get_all_paths(self.data_path,extension)
        self.n_samples=len(self.all_paths)
        
    def __getitem__(self, index):
        return torch.from_numpy(load_audio_file(self.all_paths[index],sample_rate=self.number_samples,number_samples=self.number_samples,std=self.std))
   
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
   

