e6691-2022spring-project-WAVE-an3078-bmh2168-gs3160
# Adversarial Audio Synthesis
This report summarizes the findings of the original Adversarial Audio Synthesis paper and shows a reproduction of the results with PyTorch [1]. SpecGAN and WaveGAN have been implemented.

[1] JDonahue, C., McAuley, J. and Puckette, M., 2018. Adversarial Audio Synthesis. 
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


## Jupyter noteooks descriptions
### 0. Results overview
- **Load and Test Models.ipynb** loads all the saved models and computes the top-1,3 and 5 accuracy on the associated test data sets.
### 1. CIFAR10 
- **ResNet with CIFAR10.ipynb** Shows the training process and results of ResNet et SE-Resnet models on CIFAR-10
- **ResNeXt with CIFAR10.ipynb** Shows the training process and results of ResNeXt et SE-ResneXt models on CIFAR-10
- **InceptionV3 with CIFAR10-100.ipynb** Shows the training process and results of InceptionV3 et SE-InceptionV3 models on CIFAR-10
### 2. CIFAR 100
- **ResNet with CIFAR100.ipynb** Shows the training process and results of ResNet et SE-Resnet models on CIFAR-100
- **ResNeXt with CIFAR100.ipynb** Shows the training process and results of ResNeXt et SE-ResneXt models on CIFAR-100
- **InceptionV3 with CIFAR10-100.ipynb** Shows the training process and results of InceptionV3 et SE-InceptionV3 models on CIFAR-100
### 3. Tiny ImageNet
- **ResNet18 with tinyImageNet.ipynb** Shows the training process and results of ResNet-18 et SE-Resnet-18 models on Tiny ImageNet with and without data augmentation
- **ResNet34 with tinyImageNet.ipynb** Shows the training process and results of ResNet-34 et SE-Resnet-34 models on Tiny ImageNet with and without data augmentation
- **ResNet50 with tinyImageNet.ipynb** Shows the training process and results of ResNet-50 et SE-Resnet-50 models on Tiny ImageNet with and without data augmentation
### 4. Other tests on SE-block parameters
- **analysis_ablation.ipynb** shows the ablation study tests (Different SE block integrations)
- **analysis_activation.ipynb** shows the different activation distributions accross channels for all stages and block ids
- **analysis_inference.ipynb** shows the inference speed performance of different ResNet et SE-ResNet models with Tiny ImageNet
- **analysis_ratio.ipynb** shows the effect of the ratio parameter on the accuracies
- **analysis_stage.ipynb** shows the impact of each stage where SE blocks are added.

### ./utils python files

Description is also in the ./utils directory 

- **custom_resnet.py**: builds custom ResNet models with the specified input, output sizes, stages, block multiplicity, and kernel sizes.
- **custom_ResNeXt.py**: builds custom ResNeXt models
- **SE_resnet.py**: build custom SE-ResNet models with the specified input, output sizes, stages, block multiplicity, and kernel sizes.
- **SE_ResNeXt.py**: build custom SE-ResNeXt models
- **add_SE.py**: adds SE blocks to any models by entering a list of layers where the SE blocks go.
- **ablation_resnet.py**: builds custom SE-Resnet models with different SE-block integration methods: Standard, POST, PREO and Identity
- **evaluate_model.py**: evaluate model accuracy with Top-n accuray parameters
- **train_xxx.py**: all the training functions for the CIFAR, Tiny ImageNet and ratio tests on different models.

### ./img folder
Contains graphs of the models of the ablation test and stages test
### ./figures folder
Contains miscellaneous figures such as the GCP screenshots

# Download the data sets
- **Drums dataset** http://deepyeti.ucsd.edu/cdonahue/wavegan/data/drums.tar.gz
- **Piano Bach dataset** http://deepyeti.ucsd.edu/cdonahue/mancini_piano.tar.gz
- **Speech sc09 dataset** http://deepyeti.ucsd.edu/cdonahue/sc09.tar.gz


# Organization of this directory

```
TODO
```


