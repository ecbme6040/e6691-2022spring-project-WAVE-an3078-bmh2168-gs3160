

## Demo site
Examples of generated audio clips can be found on the demo page: https://ecbme6040.github.io/e6691-2022spring-project-WAVE-an3078-bmh2168-gs3160/

# Organization of this directory 

```
│   gansynth_train.py (train GANSynth)
│   gansynth_generate.py (generate audio using trained GANSynth)
│
└───lib
        data_helpers.py (functions for data processing)
        data_normalizer.py (functions for data processing)
        datasets.py (functions to query imp info from dataset)
        flags.py (utils for handling hyperparameter states)
        generate_util.py (functions for audio generation)
        layers.py (layers for progressive GAN training)
        model.py (GANSynth model definition)
        network_functions.py (generator and discriminator)
        networks.py (generator and discriminator)
        specgrams_helper.py (signal processing helper functions)
        specgrams_helper_test.py
        spectral_ops.py (signal processing functions)
        spectral_ops_test.py
        train_util.py (utility functions for training)
        util.py
└───configs
        mel_prog_hires.py (optimal parameters for training)
```


## Download the models
Model weights are ordered by dataset folders (link below) 
```
Drive directory tree
Root/
├── GANSynth/
│   ├── NSynth/

```
[Lion drive link](https://drive.google.com/drive/folders/1CPD3boEK5Dw2LmLcUIzUJOnStPdkuBL5?usp=sharing)
(https://drive.google.com/drive/folders/1CPD3boEK5Dw2LmLcUIzUJOnStPdkuBL5?usp=sharing)



# Download the data set
- https://magenta.tensorflow.org/datasets/nsynth

# Instructions for Generation

First set up magents (https://github.com/magenta/magenta/blob/main/README.md). Download the checkpoint, and run the following command from root directory of magenta:

```

python your_directory/gansynth_generate.py --ckpt_dir=/path/to/checkpoint --output_dir=/path/to/output/dir --midi_file=/path/to/file.mid

```

The MIDI argument is optional, and is not needed if you want to generate random sequences of sounds.

