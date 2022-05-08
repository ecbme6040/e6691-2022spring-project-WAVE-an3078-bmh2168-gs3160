import glob
import librosa
import numpy as np
import os 
import torch
from torch.utils.data import Dataset
from scipy import signal
import matplotlib.pyplot as plt
from torch.autograd import grad
from tqdm import tqdm
import random
import torchaudio
from torchaudio import transforms

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

# transforms audio tensor array to 2D spectrogram
transform_spectrogram = transforms.Spectrogram(
        n_fft=255,
        win_length=255,
        hop_length=128,
        center=True,
        pad_mode="reflect",
        power=2.0)

# inverts generated spectrogram to audio tensor array 
griffinLim = transforms.GriffinLim(
        n_fft=255,
        win_length=255,
        hop_length=128)
    
    
#loads from path to numpy
def load_audio_file(path,sample_rate=16000,number_samples=16384,std=False,start_only=True,spectrogram=False):
    
    """
    loads audio file from path, returns the normalized audio with fixed padded length (float32 numpy array)
    
    path: audio sample path
    sample_rate: sample rate of returned audio sample array
    number_samples: lengths of the return audio sample array
    std: boolean, if true normalize the audio sample
    start_only: if true start audio at the beginning. If false, start at a random point 
    spectrogram: returns spectrogram images for specGan if true. Audio tensor otherwise
    """
    debug=False
    
    try:
        if debug:
            print(path)
        
        audio, _ = librosa.load(path, sr=sample_rate)
        if debug:
            print(audio)
      
        if std:
            #audio -=np.mean(audio) makes no sense to introduce DC component to the audio
            audio /= np.std(audio)
            
        if debug:
            print('shape',audio.shape)
        
        
        lenght = len(audio)
    except Exception as e:
        
        raise e

    #choose random start
    if not start_only and (lenght-number_samples)>0:
        start=max(0,random.randrange(lenght-number_samples))
        audio=audio[start:start+number_samples]
        lenght = len(audio)
        
    # padding
    if lenght < number_samples: 
        pad = number_samples - lenght
        left = pad // 2
        right = pad - left
        audio = np.pad(audio, (left, right), mode="constant")
        lenght = len(audio)
    
    # fixed length audio samples
    if lenght != number_samples :
        start_index = np.random.randint(0, max(0,(lenght - number_samples) // 2))
         
        audio=audio[start_index:start_index + number_samples]
        
    # returns 2d spectrogram if true
    if spectrogram:
        return transform_spectrogram(torch.from_numpy(audio)).numpy()
    return audio.astype("float32")





######################### Dataset Class
class AudioDataset_ram(Dataset):
    """
    Dataset class implementation for the audio data
    
    Loads from ram
    
    data_path: directory of audio data, dataset will contain all audio within the data_path (recursively)
    sample_rate: sample rate of returned audio sample array
    number_samples: lengths of the return audio sample array
    extension: extension of the audio data, only the files that matches the extension will be considered.
    std: boolean, if true normalize the audio sample
    device: load into ram if 'cpu' pf vram if cuda is available
    spectrogram: returns spectrogram images for specGan if true. Audio tensor otherwise
    """   
    def __init__(self, data_path,sample_rate=16000,number_samples=16384,extension='wav',std=False,device='cpu',start_only=True,spectrogram=False):
        self.std=std
        self.sample_rate=sample_rate
        self.number_samples=number_samples
        self.extension=extension
        self.data_path=data_path
        self.all_paths=get_all_paths(self.data_path,extension)
        self.n_samples=len(self.all_paths)
        self.start_only=start_only 
        self.spectrogram=spectrogram
        self.mel_mean=0
        self.mel_std=0
        # stores the data
        audio_shape=len(load_audio_file(self.all_paths[0],sample_rate=self.number_samples,number_samples=self.number_samples,std=self.std))
        self.data=np.zeros((self.n_samples,audio_shape)) 
        if spectrogram:
            self.data=np.zeros((self.n_samples,128,128))
            
            
        # loads data from disk into the 'device' ram
        with tqdm(self.all_paths, unit="sample") as samples: 
            for i, path in enumerate(samples):
                self.data[i]=load_audio_file(path,sample_rate=self.number_samples,number_samples=self.number_samples,std=self.std,start_only=self.start_only,spectrogram=self.spectrogram)
                if i%10==0:
                    samples.set_description(f"loading sample {i}")
        
        #Normalize for tanh between -1 and 1 as suggested in the paper
        if spectrogram:
            self.mel_mean = np.mean(self.data)
            self.mel_std = np.std(self.data)
            self.data = (self.data - self.mel_mean) / (3.0 * self.mel_std)
            self.data=np.clip(self.data, -1.0, 1.0) #clipping
        
        self.data=torch.tensor(self.data).type(torch.FloatTensor).to(device) #store to device
    def __getitem__(self, index):
        
        return self.data[index][None,:]
   
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
class AudioDataset(Dataset):
    """
    Dataset class implementation for the audio data
    
    data_path: directory of audio data, dataset will contain all audio within the data_path (recursively)
    sample_rate: sample rate of returned audio sample array
    number_samples: lengths of the return audio sample array
    extension: extension of the audio data, only the files that matches the extension will be considered.
    std: boolean, if true normalize the audio sample
    spectrogram: returns spectrogram images for specGan if true. Audio tensor otherwise
    """   
    def __init__(self, data_path,sample_rate=16000,number_samples=16384,extension='wav',std=False,start_only=True,spectrogram=False):
        self.std=std
        self.sample_rate=sample_rate
        self.number_samples=number_samples
        self.extension=extension
        self.data_path=data_path
        self.all_paths=get_all_paths(self.data_path,extension)
        self.n_samples=len(self.all_paths)
        self.start_only=start_only
        self.spectrogram=spectrogram
    def __getitem__(self, index):
       
        return torch.from_numpy(load_audio_file(self.all_paths[index],sample_rate=self.number_samples,number_samples=self.number_samples,std=self.std,start_only=self.start_only,spectrogram=self.spectrogram))[None,:]
   
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

class InceptionDataset(Dataset):
    """
    Dataset class implementation for the inceptionv3 (inception score)
    
    data_path: directory of audio data, dataset will contain all audio within the data_path (recursively)
    sample_rate: sample rate of returned audio sample array
    number_samples: lengths of the return audio sample array
    extension: extension of the audio data, only the files that matches the extension will be considered.
    std: boolean, if true normalize the audio sample
    spectrogram: returns spectrogram images for specGan if true. Audio tensor otherwise
    """   
    def __init__(self, data_path,data_transforms,sample_rate=16000,number_samples=16384,extension='wav',std=False,start_only=True,spectrogram=True):
        self.std=std
        self.sample_rate=sample_rate
        self.number_samples=number_samples
        self.extension=extension
        self.data_path=data_path
        self.all_paths=get_all_paths(self.data_path,extension)
        self.n_samples=len(self.all_paths)
        self.start_only=start_only
        self.spectrogram=spectrogram
        self.transform=data_transforms
    def __getitem__(self, index):
        dic={'Zero': 0,'One': 1,'Two': 2,'Three': 3,'Four': 4,'Five': 5,'Six': 6,'Seven': 7,'Eight':8 ,'Nine': 9}
        file_name=os.path.basename(self.all_paths[index])
        label=dic[file_name[:file_name.index('_')]]
        
        x=torch.from_numpy(load_audio_file(self.all_paths[index],sample_rate=self.number_samples,number_samples=self.number_samples,std=self.std,start_only=self.start_only,spectrogram=self.spectrogram))[None,:]
        x = x.repeat(3, 1, 1)
        return  self.transform(x), torch.tensor(label)
   
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    
    
##################### for training ####################

def wasserstein_loss(discriminator, real, generated,device,LAMBDA = 10,spec_gan=False):
    '''
    Wasserstein loss with Gradient Penalty 
    Check https://arxiv.org/pdf/1704.00028.pdf for pseudo code
    LAMBDA: penalty parameter (=10 in the paper)
    
    '''
    eps=None
    if spec_gan:
        batch_size,C,H,W=real.shape
        eps=torch.rand((batch_size, 1, 1,1)).repeat(1, C, H,W).to(device)
    else:
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

