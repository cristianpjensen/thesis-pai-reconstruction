# Deep Learning for Photoacoustic Imaging Reconstruction

Implementation of models for the work presented in the following articles.

Cristian P. Jensen, Kalloor Joseph Francis, and Navchetan Awasthi "Image depth improvement in photoacoustic imaging using transformer based generative adversarial networks", Proc. SPIE PC12842, Photons Plus Ultrasound: Imaging and Sensing 2024, PC128421V (13 March 2024); https://doi.org/10.1117/12.3001537

Please cite this if you find this useful in your work.

## Models

The following image-to-image translation models are implemented using PyTorch
and PyTorch Lightning:
 - Pix2Pix [(Isola et al. 2018)](https://arxiv.org/abs/1611.07004);
 - Attention U-net [(Oktay et al. 2018)](https://arxiv.org/abs/1804.03999);
 - Residual U-net with the following basic blocks:
   - Res18 [(He et al. 2015)](https://arxiv.org/abs/1512.03385);
   - Res50 [(He et al. 2015)](https://arxiv.org/abs/1512.03385);
   - ResV2 [(He et al. 2016)](https://arxiv.org/abs/1603.05027);
   - ResNeXt [(Xie et al. 2017)](https://arxiv.org/abs/1611.05431).
 - Trans U-net [(Chen et al. 2021)](https://arxiv.org/abs/2102.04306);
 - Palette [(Saharia et al. 2022)](https://arxiv.org/abs/2111.05826).

More models can easily be added by using the `UnetWrapper` class.

## Loss functions

The following loss functions are implemented:
 - GAN loss, using Pix2Pix' discriminator (to change the used adversarial
   network, you must change the `Discriminator` class in `models/wrapper.py`);
 - MSE loss;
 - SSIM loss;
 - PSNR loss.
 - Combination of SSIM and PSNR loss.

## Data organisation

The organisation of your data does not matter. The only important thing is
the data file, a [YAML](https://yaml.org/) file containing a list of 
input-ground truth entries. The input and ground truth files must be relative
to the directory of the data file. For example:
```yaml
- input: input/00001.png
  ground_truth: ground_truth/00001.png
- input: input/00002.png
  ground_truth: ground_truth/00002.png
- input: input/00003.png
  ground_truth: ground_truth/00003.png
```

## Training a model

To train a model, run the following:
```bash
python main.py <run name> <options>
```

When training, the model with the highest SSIM on the validation dataset will
be selected as the "best" checkpoint.

## Testing a model

To test a trained model, run the following:
```bash
python report.py <report name> <options>
```
It essentially takes a model checkpoint and test data file as input and outputs
metrics and information about the model. The following metrics are reported:
 - SSIM per image;
 - PSNR per image;
 - Mean SSIM;
 - Mean PSNR;
 - Mean RMSE;
 - FLOPs;
 - Parameter count;
 - SSIM over depth (vertically) of the image (this is only relevant for PAI
   reconstruction).
 - Outputs of the model.
