| ![Denoising example on unseen data](https://drive.google.com/uc?id=1gsppC5icp9r_g6latkOu4ta5nuNk9VI7) |
|:--:|
| *Denoising example on unseen data (Di-Conv-AE-Net)* |

# Contents
This repository contains an implementation of three models [1]:
- Dilated Convolution Autoencoder Denoising Network (Di-Conv-AE-Net)
- Denoising U-Shaped Net (D-U-Net)
- BatchRenormalization U-Net (Br-U-Net)

## References:
[1] Karaoğlu, O., Bilge, H. Ş., & Uluer, İ. (2022). Removal of speckle noises from ultrasound images using five different deep learning networks. Engineering Science and Technology an International Journal, 29(101030), 101030. [doi:10.1016/j.jestch.2021.06.010](https://doi.org/10.1016/j.jestch.2021.06.010).

## Data: 
[Kaggle Ultrasound Nerve Segmentation](https://www.kaggle.com/competitions/ultrasound-nerve-segmentation/code)

# Before Starting
1. Accept: [Kaggle Competition Rules](https://www.kaggle.com/competitions/ultrasound-nerve-segmentation/rules).

2. Download the `kaggle.json` token found on your Kaggle profile.

3. Paste your `kaggle.json` at the root of this project.

# Requirements
## For Br-U-Net :
Install the dependency:
```
$ pip install git+https://github.com/ludvb/batchrenorm@master
```

# Troubleshooting
If faced with the error:
```
Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory
```
Then run the following to solve:
```
$ export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

# How to start
Run:
```
$ python main.py
```
