# %reload_ext autoreload
# %autoreload 2

from utils.utils import *
from utils.wavegan import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

GPU=True
if not GPU:
    device= 'cpu'
# Data Params
DATA_PATH='./eecs-audio-data/nsynth/nsynth-train/audio'
AUDIO_LENGTH = 16384 #[16384, 32768, 65536] 
SAMPLING_RATE = 16000
NORMALIZE_AUDIO = False 
CHANNELS = 1

#Model params
LATENT_NOISE_DIM = 100
MODEL_CAPACITY=64
LAMBDA_GP = 10

#Training params
TRAIN_DISCRIM = 5 # how many times to train the discriminator for one generator step
EPOCHS = 500
BATCH_SIZE=64
LR_GEN = 1e-4
LR_DISC = 1e-4 # alternative is bigger lr instead of high TRAIN_DISCRIM
BETA1 = 0.5
BETA2 = 0.9


# Dataset and Dataloader
train_set = AudioDataset(DATA_PATH,sample_rate=SAMPLING_RATE,number_samples=AUDIO_LENGTH,extension='wav',std=NORMALIZE_AUDIO)
print(train_set.__len__())
train_loader = DataLoader(dataset=train_set,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)


#generator and discriminator
wave_gen = WaveGenerator(d=MODEL_CAPACITY, c=CHANNELS ,inplace=True).to(device)
wave_disc = WaveDiscriminator(d=MODEL_CAPACITY, c=CHANNELS ,inplace=True).to(device)

#random weights init
initialize_weights(wave_gen)
initialize_weights(wave_disc)
wave_gen
wave_disc
wave_gen.train()
wave_disc.train()

#Adam optim for both generator iand discriminator
optimizer_gen = optim.Adam(wave_gen.parameters(), lr=LR_GEN, betas=(BETA1, BETA2))
optimizer_disc = optim.Adam(wave_disc.parameters(), lr=LR_DISC, betas=(BETA1, BETA2))

start=-1
if start>0:
    wave_disc.load_state_dict(torch.load('./save/wavedisc/wave_'+str(start)+'_93.pt'))
    wave_gen.load_state_dict(torch.load('./save/wavegen/gen_'+str(start)+'_93.pt'))

#training
import pickle
step = start+1 # for restart from saved weights
hist=[]
for epoch in range(EPOCHS):
    with tqdm(train_loader, unit="batch") as tepoch: 
        for batch_id, real_audio in enumerate(tepoch):  
            tepoch.set_description(f"Epoch {step}")
            real_audio = real_audio.to(device)
            
            #Train Discriminator 
            for train_step in range(TRAIN_DISCRIM):
                noise = torch.randn(real_audio.shape[0], LATENT_NOISE_DIM).to(device)
                #print(noise.shape)
                fake_audio = wave_gen(noise)
                disc_real = wave_disc(real_audio).reshape(-1)
                disc_fake = wave_disc(fake_audio).reshape(-1)
                loss_disc = wasserstein_loss(wave_disc, real_audio, fake_audio,device,LAMBDA = LAMBDA_GP)
                wave_disc.zero_grad()
                loss_disc.backward(retain_graph=True)
                optimizer_disc.step()

            # Train the generator!
            all_wasserstein = wave_disc(fake_audio).reshape(-1)
            loss = -torch.mean(all_wasserstein)
            wave_gen.zero_grad()
            loss.backward()
            optimizer_gen.step()

            # Print progress, save stats, and save model
            hist.append([loss.item(),loss_disc.item()])
            if batch_id % 10 == 0 and batch_id > 0:
                tepoch.set_postfix(gen_loss=loss.item(), disc_loss=loss_disc.item())
                
            if batch_id % 100 == 0 and batch_id > 0:
                with open('./save/wavehist/hist_'+str(step)+'_'+str(epoch)+'_'+str(batch_id)+'.pkl', 'wb') as f:
                    pickle.dump(hist, f)
                torch.save(wave_gen.state_dict(), './save/wavegen/gen_'+str(step)+'_'+str(batch_id)+'.pt')
                torch.save(wave_disc.state_dict(), './save/wavedisc/wave_'+str(step)+'_'+str(batch_id)+'.pt')
                with torch.no_grad():
                    fake = wave_gen(noise)
                    torch.save(fake, './save/wavefake/fake_'+str(step)+'_'+str(batch_id)+'.pt')
                 

            step += 1
