e6691-2022spring-project-WAVE-an3078-bmh2168-gs3160
# Adversarial Audio Synthesis
This report summarizes the findings of the original Adversarial Audio Synthesis paper and shows a reproduction of the results with PyTorch [1]. SpecGAN and WaveGAN have been implemented.

[1] JDonahue, C., McAuley, J. and Puckette, M., 2018. Adversarial Audio Synthesis. 


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

#### './WaveGan and SpecGAN' folder contain relevant code for the GANSynth model
#### './GANSynth' folder contain relevant code for the GANSynth model

```
│   .gitignore
│   README.md
│
├───docs
│   │   README.md
│   │
│   └───examples
│       ├───paper
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
│   ├───Checkpoints
│   │       temp.txt
│   │
│   ├───configs
│   │       mel_prog_hires.py
│   │       temp.txt
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
│           temp.txt
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

