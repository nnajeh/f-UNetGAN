# f-UNetGAN
This repository provides the official PyTorch implementation of f-UNetGAN.




## Paper:
### A Fast Residual Attention Network for Fine-grained Unsupervised Anomaly Detection and Localization	
####  Najeh Nafti, Olfa Besbes, Asma Ben Abdallah, Antoine Vacavant, Mohamed Hedi Bedoui
#### Paper Accepted in: Applied Soft Computing Journal




## Abstract
In recent years, unsupervised anomaly detection has gained a momentum tremendous in medical applications, with Generative Adversarial Networks (GANs) leading the way in deep anomaly detection. However, GAN-based methods may not always be effective in accurately detecting anomalies especially at the pixel-level, where finer features are necessary for accurate detection and localization. In this paper, we propose f-UNetGAN, a novel GAN-based fast residual attention network for fine-grained anomaly detection and localization in a fully unsupervised manner. Firstly, a novel UNet-based discriminator architecture is constructed that enables the model to learn finer details of the input image by extracting low-level features, enhancing its ability to output both global and local information. We define four variants of the new UNet discriminator: Convolutional U-Net discriminator, Convolutional Attention U-Net discriminator, Residual U-Net discriminator, and Residual Attention U-Net discriminator. Then, we add an encoder network to the GAN model to enable fast mapping from images to the latent space, facilitating the evaluation of unseen images. After, we propose new cost functions to consider the new discriminator architecture, ensuring fine-grained anomaly localization. Indeed, we introduce a per-pixel consistency regularization technique that employs the Mixup technique to encourage the discriminator to attend to the pixel-level details of the data, empowered by the per-pixel feedback of the U-Net discriminator. Finally, we incorporate attention modules into our f-UNetGAN, which allows for the capture of both spatial and channel-specific features, resulting in improved identification of important regions and extraction of more intricate features. We evaluate our method on a COVID-19 dataset, and we validate its generalization ability on two benchmark synthetic datasets. The experimental results show that the proposed method achieves more accurate anomaly localization than other state-of-the-art methods.
 

## Graphical Abstract
![GraphicalAbstract](https://github.com/user-attachments/assets/ceab6eaf-4ef9-49bd-b541-32fb545bb026)

This is a PyTorch/GPU implementation of the paper [f-UNetGAN : A Fast Residual Attention Network for Fine-grained Unsupervised Anomaly Detection and Localization]
## Results

### COVID-19 dataset:
<img src="https://github.com/user-attachments/assets/991785f3-a665-42de-a19a-cde0b2934eba" alt="COVID-Heatmap" width="300" />

### MVTEC-AD benchmarking dataset:
#### Bottle object:
##### <img src="https://github.com/user-attachments/assets/eb6a7e6f-c19c-42bc-83dd-5b319b709c71" alt="full-heatmap-MVTEC" width="300" />

#### Carpet Texture:
##### <img src="https://github.com/user-attachments/assets/a22479f9-6bb1-46d5-987e-4fc8f6b0a04c" alt="full-heatmap-MVTEC" width="300" />


This is a PyTorch/GPU implementation of the paper [A fast residual attention network for fine-grained unsupervised anomaly detection and localization]:
```
@article{nafti2024fast,
  title={A fast residual attention network for fine-grained unsupervised anomaly detection and localization},
  author={Nafti, Najeh and Besbes, Olfa and Abdallah, Asma Ben and Vacavant, Antoine and Bedoui, Mohamed Hedi},
  journal={Applied Soft Computing},
  pages={112066},
  year={2024},
  publisher={Elsevier}
}
```
