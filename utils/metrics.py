def auc_roc(mask, score):
   # mask = np.asarray(mask)
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    auc_score = roc_auc_score(mask.ravel(), score.ravel())
    fpr, tpr, thresholds = roc_curve(mask.ravel(), score.ravel(), pos_label=1)

    return auc_score, [fpr, tpr, thresholds]


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())
  
  
  
def au_prc(true_mask, pred_mask):

    # Calculate pr curve and its area
    precision, recall, threshold = precision_recall_curve(true_mask, pred_mask)
    au_prc = auc(recall, precision)

    # Search the optimum point and obtain threshold via f1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0

    th = threshold[np.argmax(f1)]

    return au_prc, th
  
  
  
intersection = np.logical_and(target, prediction)
union = np.logical_or(target, prediction)
iou_score = np.sum(intersection) / np.sum(union)




