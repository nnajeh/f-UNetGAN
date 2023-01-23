# F-UNetGAN



## Abstract:
In recent years, there has been a tremendous increase in the use of anomaly detection
and localization techniques in industrial and medical applications. However, it can be
challenging due to the lack of annotated anomalous image samples. Generative Adver-
sarial Networks (GANs) have become powerful tools for various learning tasks, inclu-
ding anomaly detection and segmentation.
In this paper, we propose a novel GAN-based unsupervised mechanism called Fast
UNet GAN (F-UNetGAN), which can effectively detect and segment anomalies in
images without any prior knowledge. F-UNetGAN is a framework composed of three
components : a generator, a UNet discriminator, and an encoder. The proposed me-
chanism involves replacing the traditional discriminator structure with a UNet discri-
minator, which allows pixel level information to the generator,which improves image
generation, and in turn, improves image detection and localization. The encoder enables
fast mapping from images to the latent space, which facilitates the evaluation of unseen
images. We evaluate the proposed method on lung disease datasets, including COVID-
19 and lung cancer, and we validate its capacity on synthetic data sets (i.e., MNIST and
MVTec AD). Experiments are evaluated using AUC-ROC, PR-AUC, f1-score, sensi-
tivity, and recall metrics. The results indicate the effectiveness of the proposed GAN-
based detection approach for the detection and segmentation tasks.


## Proposed Framework
![U-F-BigGAN](https://user-images.githubusercontent.com/38373885/213512790-a85d2aec-4e8e-4390-951f-5ae7e16a3492.png)


This is a PyTorch/GPU implementation of the paper [F-UNetGAN : Fast UNET GAN For Unsupervised Anomaly Detection And
Segmentation : Application to computed tomography scans for Lung diseases]:
