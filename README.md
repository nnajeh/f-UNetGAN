# U-FGAN



#Abstract:

In recent years, there has been a tremendous increase in the use of anomaly detection
and localization techniques in industrial and medical applications. However, it can be
challenging due to the lack of annotated anomalous image samples. Generative Adver-
sarial Networks (GANs) have become powerful tools for various learning tasks, inclu-
ding anomaly detection and segmentation.
In this paper, we propose a novel GAN-based unsupervised mechanism called UNet
Fast GAN (F-AGAN), which can effectively detect and segment anomalies
in images without any prior knowledge. F-AUGAN is a framework composed of three
components : a generator, a UNet discriminator, and an encoder. The proposed approach
involves replacing the traditional discriminator with a UNet discriminator, which is able
to capture different local information of the input images. The optimized discriminator
provides useful information to the generator, which improves image generation, and
in turn, improves image detection and localization. The encoder enables fast mapping
from images to the latent space, which facilitates the evaluation of unseen images. We
evaluate the proposed method on lung disease datasets, including COVID-19 and lung
cancer, and we validate its capacity on synthetic data sets (i.e., MNIST and MVTec AD).
Experiments are evaluated using AUC-ROC, f1-score, sensitivity, and recall metrics.
The results indicate the effectiveness of the proposed GAN-based detection approach
for the detection and segmentation tasks

## Proposed Framework
![U-F-BigGAN](https://user-images.githubusercontent.com/38373885/213512790-a85d2aec-4e8e-4390-951f-5ae7e16a3492.png)


This is a PyTorch/GPU implementation of the paper [Fast Unsupervised Residual Attention GAN for COVID-19 Detection]:
