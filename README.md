# Contrast Representation Learning from Imaging Parameters for Magnetic Resonance Image Synthesis
#### Step 1. To organize your multi-contrast dataset similar to the paths in [config](https://github.com/xionghonglin/CRL_MICCAI_2024/blob/main/CRL/configs/train_crl.yaml) file, you should structure it in a systematic way that reflects the different contrasts. Below is an example of how you could structure your dataset:

```
dataset
|
├── contrast_1
│   ├── image_1.jpg
│   └── image_2.jpg
|
├── contrast_2
│   ├── image_1.jpg
│   └── image_2.jpg
|
└── contrast_3
    ├── image_1.jpg
    └── image_2.jpg
```
#### Step 2. Train the model by simply running the following command:
      
      python train.py
