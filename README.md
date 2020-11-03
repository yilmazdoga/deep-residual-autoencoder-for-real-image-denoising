## Deep Residual Autoencoder for Real Image Denoising
 
Details about the project and demo images can be found at [project website](https://yilmazdoga.com/deep_residual_autoencoder_for_real_image_denoising).

![](./readme_assets/denoising_before_after.gif)

## Dependencies
* torch>=1.6.0
* torchvision>=0.7.0
* Pillow~=7.2.0
* numpy~=1.19.2
* piq~=0.5.1
* pytorch_msssim~=0.2.1
* torchsummary~=1.5.1
* tensorboard~=2.3.0
* prefetch_generator~=1.0.1
* tqdm~=4.50.2
* matplotlib~=3.3.2

## Installation
1- Clone this repo.
```shell script
git clone https://github.com/yilmazdoga/Deep_Residual_Autoencoder_for_Real_Image_Denoising.git
```
2- Go to the project directory.
```shell script
cd  <CLONED_PROJECT_DIR>
```
3- Create a virtual environment.
```shell script
python3 -m venv venv
```
4- Activate the virtual environment.
```shell script
source venv/bin/activate
```
5- Install required packages.
```shell script
pip install -r requirements.txt
```

> At this point you should be able to use the pretrained models to denoise a given image. However, if you want to train the model on your machine or run the test script on the validation data continue the installation with the following steps.

6- Download the Smartphone Image Denoising Dataset (SIDD) medium sRGB version from [here](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php).

7- Unzip the downloaded file.

8- Move the Data folder inside the unzipped file to ```PROJECT_ROOT/datasets/SIDD_Dataset/SIDD_Medium_sRGB/```

9- Download the SIDD Noisy and Ground-Truth sRGB validation data from [here](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php).

10- Move the downloaded files to ```PROJECT_ROOT/datasets/SIDD_Dataset/SIDD_Validation/```

> At this point you should be able to use all of the functionalities.

## Usage
#### Training
To start a training run the following command with desired options.
```shell script
python3 train.py YOUR_OPTIONS
```

Example:
```shell script
python3 train.py --model AEv2_0 --loss-func L1_MSSSIM --augs MixUp_Tensor --epochs 200 --batch-size 24 --patch-size 128 --gpus 2
```

For further options inspect ```options/base_options``` and ```options/base_options```.

#### Testing
To denoise an image run the following command with desired options.
```shell script
python3 test.py --noisy-image-path PATH_TO_YOUR_NOISY_IMAGE --gt-image-path PATH_TO_YOUR_GT_IMAGE --YOUR_MODEL_OPTIONS
```

> If you dont have Ground-Truth (GT) image you can give noisy image as GT image but doing so will result a wrong SSIM and PSNR calculation.

To run the test code over the validation dataset run the following command. To use this feature make sure that no Noisy or GT input image path is provided.
```shell script
python3 test.py --YOUR_MODEL_OPTIONS
```

## Contribute
Please feel free to open an issue or to send an e-mail to ```yilmaz.doga.11481@ozu.edu.tr```