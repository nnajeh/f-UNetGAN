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

	
	

def measure_all(gtsavepath,prepath):

    seg = sitk.GetArrayFromImage(sitk.ReadImage(gtsavepath, sitk.sitkInt16))
    label = sitk.GetArrayFromImage(sitk.ReadImage(prepath, sitk.sitkInt16))

    #### dice
    zeros =np.zeros(seg.shape) 
    ones = np.ones(seg.shape)  
    tp =((seg == ones) & (label == ones)).sum()
    fp=((seg==zeros) & (label==ones)).sum()
    tn=((seg==zeros) & (label==zeros)).sum()
    fn=((seg==ones) & (label==zeros)).sum()
    core=0.000000000000000001
    dice = (tp*2)/(fp+tp*2+fn)
    # mcc = (tp*tn-fp*fn)/(((tp+fp+core)*(tp+fn+core)*(tn+fp+core)*(tn+fn+core))**0.5)
    acc=(tp+tn+core)/(tp+fp+tn+fn+core)
    precision=(tp+core)/(tp+fp+core)
    recall_sen=(tp+core)/(tp+fn+core)
    spc=(tn+core)/(tn+fp+core)
    jac = tp/(fp+tp+fn)
    #### hausdorff
    quality = dict()
    seg1 = sitk.ReadImage(gtsavepath, sitk.sitkInt16)
    label1 = sitk.ReadImage(prepath, sitk.sitkInt16)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    if ((sitk.GetArrayFromImage(seg1).sum() > 0) and (sitk.GetArrayFromImage(label1).sum() > 0)):
        hausdorffcomputer.Execute(seg1, label1)  # (labelTrue > 0.5, labelPred > 0.5)
        quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
        quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
    else:
        quality["avgHausdorff"] = "max"
        quality["Hausdorff"] = "max"

    with open(r"D:\TGN_AVP_FVN\CODE\CNTSeg\final_results\result1_excel\LOSS\dice_FVN.txt", 'a+') as f:
        f.writelines("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(prediction,dice,jac,acc,precision,recall_sen,spc,quality["Hausdorff"],quality["avgHausdorff"]))
    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(prediction,dice,jac,acc,precision,recall_sen,spc,quality["Hausdorff"],quality["avgHausdorff"]))


def au_prc(true_mask, pred_mask):

    # Calculate pr curve and its area
    precision, recall, threshold = precision_recall_curve(true_mask, pred_mask)
    au_prc = auc(recall, precision)

    # Search the optimum point and obtain threshold via f1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0

    th = threshold[np.argmax(f1)]

    return au_prc, th

## Calculate threshold	
precision, recall, thresholds = precision_recall_curve(np.asarray(y_true).ravel(), np.asarray(y_score).ravel())
a = 2 * precision * recall
b = precision + recall
f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
threshold = thresholds[np.argmax(f1)]
print(threshold)





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


