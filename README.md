# f-UNetGAN



## Abstract:
In recent years, unsupervised anomaly detection has gained a momentum tremendous in medical applications, with Generative Adversarial Networks (GANs) leading the way in deep anomaly detection. However, GAN-based methods may not always be effective in accurately detecting anomalies especially at the pixel-level, where finer features are necessary for accurate detection and localization. In this paper, we propose f-UNetGAN, a novel GAN-based fast residual attention network for fine-grained anomaly detection and localization in a fully unsupervised manner. Firstly, a novel UNet-based discriminator architecture is constructed that enables the model to learn finer details of the input image by extracting low-level features, enhancing its ability to output both global and local information. We define four variants of the new UNet discriminator: Convolutional U-Net discriminator, Convolutional Attention U-Net discriminator, Residual U-Net discriminator, and Residual Attention U-Net discriminator. Then, we add an encoder network to the GAN model to enable fast mapping from images to the latent space, facilitating the evaluation of unseen images. After, we propose new cost functions to consider the new discriminator architecture, ensuring fine-grained anomaly localization. Indeed, we introduce a per-pixel consistency regularization technique that employs the Mixup technique to encourage the discriminator to attend to the pixel-level details of the data, empowered by the per-pixel feedback of the U-Net discriminator. Finally, we incorporate attention modules into our f-UNetGAN, which allows for the capture of both spatial and channel-specific features, resulting in improved identification of important regions and extraction of more intricate features. We evaluate our method on a COVID-19 dataset, and we validate its generalization ability on two benchmark synthetic datasets. The experimental results show that the proposed method achieves more accurate anomaly localization than other state-of-the-art methods.
 

## Proposed Framework
![Framework](https://github.com/nnajeh/f-UNetGAN/assets/38373885/5c02b1c0-4cf5-4b9c-b8e7-462a7d6c28a1)

This is a PyTorch/GPU implementation of the paper [f-UNetGAN : A Fast Residual Attention Network for Fine-grained Unsupervised Anomaly Detection and Localization]:
