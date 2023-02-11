import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image,cmap='gray')
    plt.show()


def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage[0][0])
	ax[1].imshow(origMask[0][0])
	ax[2].imshow(predMask[0][0].cpu().data.numpy())
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	# set the layout of the figure and display it
	figure.tight_layout()
	figure.show()

	
	
## Calculate threshold	
precision, recall, thresholds = precision_recall_curve(np.asarray(y_true).ravel(), np.asarray(y_score).ravel())
a = 2 * precision * recall
b = precision + recall
f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
threshold = thresholds[np.argmax(f1)]
print(threshold)

def auc_roc(mask, score):
    """
    ref: iou https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    ref: confusion matrix https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    ref: confusion matrix https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    auc_score = roc_auc_score(mask.cpu().ravel(), score.cpu().ravel())
    fpr, tpr, thresholds = roc_curve(mask.cpu().ravel(), score.cpu().ravel(), pos_label=1)

    return auc_score, [fpr, tpr, thresholds]


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

with torch.no_grad():
for i, (imgs,labels, masks) in enumerate(test_dataloader):   
    
        real_img = imgs.to(device)
        mask = masks.to('cuda')
        fake_imgs = pretrained_G(pretrained_E(real_img))
    
        #Discriminator encoder Output
        real_img_feat,_ = pretrained_D.extract_all_features(real_img)
        fake_img_feat,_ = pretrained_D.extract_all_features(fake_imgs)
                
    
        ###Discriminator Decoder Output
        _,real_pixel_feat = pretrained_D.extract_all_features(real_img)
        _,fake_pixel_feat = pretrained_D.extract_all_features(fake_imgs)
        
 
        anomaly_score=  torch.abs(fake_imgs-real_img)+ torch.mean(torch.abs(fake_pixel_feat - real_pixel_feat))  
        
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = anomaly_score.squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().detach().numpy()
        
        # filter out the weak predictions and convert them to integers
        predMask = (predMask > threshold) * 255
        predMask = predMask.astype(np.uint8)
    

            
        visualize(
           original_image = real_img[0][0].cpu().data.numpy(),
           ground_truth_mask = masks[0][0].cpu().data.numpy(),
           predicted_mask_real = anomaly_score[0][0].cpu().data.numpy(),
           predicted_mask_fake = fake_pixel_feat[0][0].cpu().data.numpy(),
           outputs = (torch.argmax(torch.softmax(anomaly_score, dim=1), dim=1, keepdim=True))[0][0].cpu().data.numpy(),
           predMask = predMask[0],

            #predicted_building_heatmap = pred_building_heatmap
          )
                
        x =plt.imshow(anomaly_score[0][0].cpu().data.numpy(), extent=(-3, 3, 3, -3), interpolation='bilinear', cmap='jet')
        plt.colorbar(x);


def au_prc(true_mask, pred_mask):

    # Calculate pr curve and its area
    precision, recall, threshold = precision_recall_curve(true_mask, pred_mask)
    au_prc = auc(recall, precision)

    # Search the optimum point and obtain threshold via f1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0

    th = threshold[np.argmax(f1)]

    return au_prc, th
