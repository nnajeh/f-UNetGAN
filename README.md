# F-UNetGAN



## Abstract:
In recent years, there has been a tremendous increase in the use of anomaly detection and localization techniques in industrial and medical applications, particularly in the case of lung diseases, which pose a significant challenge to global health. However, the lack of annotated anomalous image samples makes this task difficult. Generative Adversarial Networks (GANs) have emerged as a potent tool for various learning tasks, including unsupervised anomaly detection and segmentation. In this paper, we propose a novel GAN-based unsupervised mechanism called Fast UNet GAN (F-UNetGAN), that can effectively detect and segment anomalies in images without any prior knowledge. F-UNetGAN framework consists of three components : a generator, a UNet discriminator, and an encoder. The UNet-based discriminator replaces the traditional discriminator structure with a new UNet discriminator structure, which outputs per-pixel information with different levels of details, that improves image detection and localization, and in turn,image generation. And the encoder enables fast mapping from images to the latent space, which facilitates the evaluation of unseen images. To further improve the performance and ensure a better pixel-wise detection, we define a new loss functions for the generator, UNet discriminator, and encoder models that take into account the new architecture. We evaluate the proposed method on lung disease datasets, including COVID-19 and lung cancer, and we validate its capacity on two benchmark synthetic data sets, the MNIST and the MVTec AD. Experiments are evaluated using AUC-ROC, PR-AUC, f1-score, sensitivity, and recall me-
trics. The results indicate the effectiveness of the proposed unsupervised GAN-based approach for the detection and segmentation tasks. The F-UNetGAN code is available at https://github.com/nnajeh/F-UNetGAN


## Proposed Framework
![F-UNetGAN](https://user-images.githubusercontent.com/38373885/218199301-8fd50b4d-8d07-483e-96df-38c94c056c5e.png)

This is a PyTorch/GPU implementation of the paper [F-UNetGAN : Fast UNET GAN For Unsupervised Anomaly Detection And
Segmentation : Application to Lung diseases]:
