# F-UNetGAN



## Abstract:
In recent years, there has been a tremendous increase in the use of anomaly detection and localization techniques in medical applications, particularly in the case of lung diseases, which pose a significant challenge to global health. However, the lack of annotated anomalous image samples makes this task difficult.

In this paper, we propose a novel GAN-based unsupervised approach called Fast U-Net GAN (F-UNetGAN), for lung diseases detection and segmentation. The F-UNetGAN framework consists of three components: a generator, a U-Net discriminator, and an encoder. The encoder enables fast mapping from images to the latent space, facilitating the evaluation of unseen images. Besides, the traditional discriminator is replaced with a new U-Net discriminator, which outputs per-pixel information with different levels of details, that improve both anomaly detection and localization, as well as image generation. Indeed, we incorporate the Mixup consistency regularization technique, to empower the per-pixel output of the discriminator to leverage the U-Net discriminator to focus more on semantic and structural changes between real and generated images. To further enhance performance and ensure better pixel-wise detection, we propose new loss functions for the generator, U-Net discriminator, and encoder models that take into account the new discriminator architecture. We evaluate our method on two lung diseases datasets, COVID-19 and lung cancer datasets in chest computed tomography (CT), and we validate its capacity on two benchmark synthetic datasets, i.e.,  MNIST, and MVTec AD. In our experiments, we used AUC-ROC, PR-AUC, f1-score, sensitivity, and recall metrics. The obtained results indicate the effectiveness of the proposed unsupervised GAN-based approach for the anomaly detection and segmentation tasks.

## Proposed Framework
![Full-Framework](https://user-images.githubusercontent.com/38373885/222929562-dc5ef22e-cb7a-4c92-8201-38a8ff15fed0.png)

This is a PyTorch/GPU implementation of the paper [F-UNetGAN : Fast U-Net GAN for Unsupervised Anomaly Detection and Segmentation of Lung Diseasess]:
