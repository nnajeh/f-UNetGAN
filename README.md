# f-RANet



## Abstract:
In recent years, there has been a tremendous increase in the use of unsupervised anomaly detection techniques in medical applications. Where Generative Adversarial Networks (GANs) have made a great breakthrough for deep anomaly detection and finding widespread use in various practical applications. However, the GAN-based methods may not always be effective for accurate anomaly detection and lack of finer features, which is mandatory for pixel-level anomaly detection.

In this paper, we propose f-UNetGAN, a novel GAN-based fast residual attention network for fine-grained unsupervised anomaly detection and localization. Firstly, a novel UNet-based discriminator architecture is constructed to facilitate the model to learn finer details of the input image by extracting low-level features, thereby enhancing the model’s ability to output both global and local information. Then, new cost functions are proposed to consider the new discriminator architecture, thus ensuring fine-grained anomaly detection. Finally, empowered by the per-pixel feedback of the U-Net discriminator, we introduce a per-pixel consistency regularization technique that employs the Mixup technique to encourage the discriminator to attend to the pixel-level details of the data. We evaluate our method on a medical dataset (i.e., COVID-19) and we validate its capacity on two benchmark synthetic datasets (i.e.,  MNIST, and MVTec AD). The experimental results show that the proposed method has accurate anomaly localization than other advanced anomaly detection methods. The f-UNetGAN code is available at \url{https://github.com/nnajeh/f-UNetGAN}.
 

## Proposed Framework
![Full-Framework](https://user-images.githubusercontent.com/38373885/222929562-dc5ef22e-cb7a-4c92-8201-38a8ff15fed0.png)

This is a PyTorch/GPU implementation of the paper [f-RANet : A Fast Residual Attention Network for Fine-grained Unsupervised Anomaly Detection and Localization]:
