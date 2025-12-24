# Crack Image Segmentation

![Example segmentation](../assets/example_segmentations.png?raw=true)

## Project Description

The goal of this project is to develop an image segmentation model for detecting cracks in images. Unlike fully supervised approaches that require pixel-level annotations for training, this project explores a weakly supervised segmentation setup, where only image-level labels are available during training.

Specifically, the model is trained using images labeled as _cracked_ or _not cracked_, without providing explicit crack masks during the training process. Despite this limitation, the model learns to localize crack regions by leveraging architectural inductive biases and weak supervision techniques.

## Model Architecture

The segmentation model is based on a Weakly Supervised U-Net architecture with a ResNet-50 encoder.

## Dataset

The model was trained and evaluated using the **Crack Segmentation Dataset** available on Kaggle:

- Dataset: Crack Segmentation Dataset
- Source: https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset
- Training images: 9,603
- Testing images: 1,695

## Training Details

- Training epochs: 3
- Supervision: Image-level labels only (`cracked` / `not cracked`)
- Evaluation metric: Dice Coefficient (Dice Similarity Coefficient, DSC)

## Results

The final performance on the test set achieved a **Dice score of 0.3908**.

The Dice score measures the overlap between predicted segmentation masks and ground-truth masks and is defined as:

$$\text{Dice} = \frac{2 \cdot |A \cap B|}{|A| + |B|}$$

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015).  
   U-Net: Convolutional Networks for Biomedical Image Segmentation.  
   _arXiv preprint arXiv:1505.04597_.  
   https://arxiv.org/abs/1505.04597

2. Yuan, Y., et al. (2022).  
   Weakly Supervised Learning for Crack Segmentation.  
   _bioRxiv_.  
   https://www.biorxiv.org/content/10.1101/2022.09.09.507144v1

3. Towards Data Science.  
   Cook Your First U-Net in PyTorch.  
   https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3/

4. Turgutlu, K.  
   Weakly Supervised Transfer Learning in Medical Imaging.  
   https://medium.com/@keremturgutlu/weakly-supervised-transfer-learning-in-medical-imaging-c89c5ca2d0be
