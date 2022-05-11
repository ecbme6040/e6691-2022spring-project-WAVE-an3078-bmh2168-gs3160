e6691-2022spring-project-WAVE-an3078-bmh2168-gs3160
# Adversarial Audio Synthesis
This report summarizes the findings of the original Adversarial Audio Synthesis paper and shows a reproduction of the results with PyTorch [1]. SpecGAN and WaveGAN have been implemented.

We also experimented with the GANSynth model [2].
All the required helper functions for GANSynth to complete the audio processing are from [3].

[1] JDonahue, C., McAuley, J. and Puckette, M., 2018. Adversarial Audio Synthesis. 

[2] Engel J, Agrawal KK, Chen S, Gulrajani I, Donahue C, Roberts A. Gansynth: Adversarial neural audio synthesis. arXiv preprint arXiv:1902.08710. 2019

[3 ]https://github.com/magenta/magenta/tree/main/magenta/models/gansynth/lib
## Demo site
Examples of generated audio clips can be found on the demo page: https://ecbme6040.github.io/e6691-2022spring-project-WAVE-an3078-bmh2168-gs3160/



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
- **NSynth dataset** https://magenta.tensorflow.org/datasets/nsynth



# Organization of this directory

#### './WaveGan and SpecGAN' folders contain relevant code for the GANSynth model
#### './GANSynth' folder contains relevant code for the GANSynth model
#### './docs' folder contains website code, and audio examples
```
│   .gitignore
│   README.md
│
├───docs
│   │   README.md
│   │
│   └───examples
│       ├───paper
|       |       specgan_drums.mp3
|       |       specgan_piano.mp3
|       |       specgan_sc09.mp3
|       |       wavegan_drums.mp3
|       |       wavegan_sc09.mp3
|       |
│       ├───specgan
│       │       drum denoised.mp3
│       │       drum.mp3
│       │       piano.mp3
│       │       sc09.mp3
│       │
│       └───wavegan
│               drum n=0.mp3
│               drum n=2.mp3
│               piano.mp3
│               sc09.mp3
│
├───GANSynth
│   │   gansynth_generate.py
│   │   gansynth_train.py
│   │   README.md
│   │   __init__.py
│   │
│   │
│   ├───configs
│   │       mel_prog_hires.py
│   │       __init__.py
│   │
│   └───lib
│           datasets.py
│           data_helpers.py
│           data_normalizer.py
│           flags.py
│           generate_util.py
│           layers.py
│           model.py
│           networks.py
│           network_functions.py
│           specgrams_helper.py
│           specgrams_helper_test.py
│           spectral_ops.py
│           spectral_ops_test.py
│           train_util.py
│           util.py
│           __init__.py
│
└───WaveGan and SpecGAN
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

