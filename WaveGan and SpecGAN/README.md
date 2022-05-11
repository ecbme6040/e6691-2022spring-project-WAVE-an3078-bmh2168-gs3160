e6691-2022spring-project-WAVE-an3078-bmh2168-gs3160
# Adversarial Audio Synthesis
This report summarizes the findings of the original Adversarial Audio Synthesis paper and shows a reproduction of the results with PyTorch [1]. SpecGAN and WaveGAN have been implemented.

[1] JDonahue, C., McAuley, J. and Puckette, M., 2018. Adversarial Audio Synthesis. 

## Jupyter notebooks descriptions

- **Generate audio.ipynb** Generates audio from a trained SpecGAN or WaveGAN model
- **Inception score.ipynb** Computes the inception score from the trained Inception-v3 model
- **Inception training.ipynb** Trains a pretrained Inception-v3 model using transfer-learning
- **SpecGan Training.ipynb** Training code for SpecGAN
- **Wavegan Training.ipynb** Training code for WaveGAN

### ./utils python files

Description is also in the ./utils directory 
- **generate_show_audio.py** functions to generates audio from a trained SpecGAN or WaveGAN model
- **split_data.py** function to slipt and save audio track into smaller shunks of the same lenghts (for the piano dataset)
- **utils.py** useful functions for data loading, processing etc
- **wavegan.py** wavegan models, with phase shuffling layer
- **specgan.py** specgan models
## Demo site
Examples of generated audio clips can be found on the demo page: https://ecbme6040.github.io/e6691-2022spring-project-WAVE-an3078-bmh2168-gs3160/

# Organization of this directory

```
│   Generate audio.ipynb
│   Inception score.ipynb
│   Inception training.ipynb
│   README.md
│   SpecGan Training.ipynb
│   Wavegan Training.ipynb
│
└───utils
        generate_show_audio.py
        specgan.py
        split_data.py
        utils.py
        wavegan.py
```


## Download the models
Model weights are ordered by dataset folders (link below) 
```
Drive directory tree
Root/
├── WaveGAN/
│   ├── drum/
|   |   ├── examples_samples.pt
│   │   ├── generator.pt
│   │   └── discriminator.pt
│   ├── piano/
|   |   ├── examples_samples.pt
│   │   ├── generator.pt
│   │   └── discriminator.pt
│   └── sc09/
|   |   ├── examples_samples.pt
│   │   ├── generator.pt
│   │   └── discriminator.pt
├── SpecGAN/
│   ├── drum/
|   |   ├── examples_samples.pt
│   │   ├── generator.pt
│   │   └── discriminator.pt
│   ├── piano/
|   |   ├── examples_samples.pt
│   │   ├── generator.pt
│   │   └── discriminator.pt
│   ├── sc09/
|   |   ├── examples_samples.pt
│   │   ├── generator.pt
│   │   └── discriminator.pt
```
[Lion drive link](https://drive.google.com/drive/folders/1CPD3boEK5Dw2LmLcUIzUJOnStPdkuBL5?usp=sharing)
(https://drive.google.com/drive/folders/1CPD3boEK5Dw2LmLcUIzUJOnStPdkuBL5?usp=sharing)



# Download the data sets
- **Drums dataset** http://deepyeti.ucsd.edu/cdonahue/wavegan/data/drums.tar.gz
- **Piano Bach dataset** http://deepyeti.ucsd.edu/cdonahue/mancini_piano.tar.gz
- **Speech sc09 dataset** http://deepyeti.ucsd.edu/cdonahue/sc09.tar.gz




