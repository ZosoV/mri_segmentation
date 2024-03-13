import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score

def plot_frequencies(reference_img, reference_labels, labels2plot = list(range(6))):

    label_names = ['air', 'skin/scalp', 'skull', 'CSF', 'Gray Matter', 'White Matter']

    # Initialize an empty list to store the histograms
    histograms = []

    # Iterate over unique labels
    for label in labels2plot:
        # Select the pixels with the current label
        pixels = reference_img[reference_labels == label]
        
        # Plot the histogram of the selected pixels
        hist = plt.hist(pixels.flatten(), bins=100, alpha=0.5, label=f'Label {label}: {label_names[label]}')
        
        # Append the histogram to the list
        histograms.append(hist)

    # Show the legend
    plt.legend()

    # Set the title and labels
    plt.title('Histogram of Reference Image by Label')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()

def plot_masks(temporal_masks, rows=1, cols=2):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))

    flatten_axes = axes.flat
    for ax, (key, mask) in zip(flatten_axes, temporal_masks.items()):
        ax.imshow(mask, cmap='gray')
        ax.set_title(f'Mask: {key}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def temporal_masks2final_segmented_mask(temporal_masks, labels = range(6)):
    segmented_labels = np.zeros_like(temporal_masks["0"])

    # Accumulate all the temporal masks in the segmented_labels
    for label in labels:
        mask = temporal_masks[str(label)]
        segmented_labels[mask == 1] = label

    return segmented_labels

def kmeans_segmentation(image, n_clusters=4):
    # Reshape the image to a 2D array
    X = image.reshape(-1, 1)

    # Fit KMeans to the data
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    # Predict the labels for the data
    labels = kmeans.predict(X)

    # Reshape the labels to the original image shape
    labels = labels.reshape(image.shape)

    # Get the centroids of the clusters
    centroids = kmeans.cluster_centers_

    # Order the centroids and return the indices
    order = np.argsort(centroids, axis=0)

    return labels, order

def calculate_auprc(y_true, y_score, experiment_name = None):

    # Binarize the labels for One-vs-Rest computation
    y_true_binarized = label_binarize(y_true, classes=list(range(6)))
    n_classes = y_true_binarized.shape[1]

    # Compute Precision-Recall and AUPRC for each class
    auprc_scores = []
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], y_score[:, i])
        auprc = auc(recall, precision)
        auprc_scores.append(auprc)
        # print(f'Class {i} AUPRC: {auprc}')

    # Calculate the average AUPRC
    average_auprc = np.mean(auprc_scores)

    return average_auprc, auprc_scores


def create_normal_distribution_mask(length=10, center_index=0, mean=None, std=1.0):
    # Set the mean to the center_index if not specified
    if mean is None:
        mean = center_index

    # Generate x values starting from 0 increasing to the end
    x = np.linspace(0, 2, length)  # Adjust the range to control the spread of the distribution

    # Shift the x values based on the desired center index
    x -= x[center_index]  # This will make the peak at the desired center index

    # Calculate the Gaussian mask using the specified mean and standard deviation
    gaussian_mask = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    
    # Normalize by a softmax
    gaussian_mask /= np.sum(gaussian_mask)
    
    return gaussian_mask


# def dice_coefficient(y_true, y_pred):
#     # Flatten the arrays to simplify the intersection and union calculations
#     y_true_flatten = y_true.flatten()
#     y_pred_flatten = y_pred.flatten()
    
#     # Calculate intersection and union
#     intersection = np.sum(y_true_flatten * y_pred_flatten)
#     union = np.sum(y_true_flatten) + np.sum(y_pred_flatten)
    
#     # Calculate Dice coefficient
#     dice = 2 * intersection / union if union != 0 else 1
#     return dice

# def macro_dice_coefficient(y_true, y_pred):
#     dice_scores = []
#     for i in range(6):
#         dice = dice_coefficient(y_true == i, y_pred == i)
#         dice_scores.append(dice)

#     macro_dice = np.mean(dice_scores)

#     return macro_dice, dice_scores

def dice_coefficients(y_true, y_pred):
    # Flatten y_true and y_pred if they are multidimensional
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # True positives, false positives, and false negatives per class
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP

    # Dice score per class
    dice_scores = 2 * TP / (2 * TP + FP + FN)

    # Micro Dice coefficient (overall)
    micro_dice = 2 * TP.sum() / (2 * TP.sum() + FP.sum() + FN.sum())

    return micro_dice, dice_scores


# segmented_labels is my prediction and reference_labels is the ground truth
def calculate_metrics(y_true, y_pred, experiment_name = None):

    if experiment_name:
        print(f"Metrics for {experiment_name}", "\n")

    # Flatten the arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Transform to one-hot
    n_values = np.max(y_pred) + 1  # This finds the number of unique values

    # Create one-hot encoded matrix
    y_score = np.eye(n_values)[y_pred]

    # Compute auc and auprc
    average_auprc, auprc_scores = calculate_auprc(y_true, y_score)
    auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')

    # Dice metrics
    micro_dice, dice_scores = dice_coefficients(y_true, y_pred)

    # Compute the metrics by label
    jaccard_index = jaccard_score(y_true, y_pred, average=None)
    # precision = precision_score(y_true, y_pred, average=None)
    # recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    # Compute weighted metrics
    accuracy = accuracy_score(y_true, y_pred)
    # weighted_jaccard = jaccard_score(y_true, y_pred, average='weighted')
    # weighted_precision = precision_score(y_true, y_pred, average='weighted')
    # weighted_recall = recall_score(y_true, y_pred, average='weighted')
    # weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    # Compute micro metrics
    micro_jaccard = jaccard_score(y_true, y_pred, average='micro')
    # micro_precision = precision_score(y_true, y_pred, average='micro')
    # micro_recall = recall_score(y_true, y_pred, average='micro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    # Compute macro metrics
    # macro_jaccard = jaccard_score(y_true, y_pred, average='macro')
    # macro_precision = precision_score(y_true, y_pred, average='macro')
    # macro_recall = recall_score(y_true, y_pred, average='macro')
    # macro_f1 = f1_score(y_true, y_pred, average='macro')

    # Save the average=none metrics in a dictionary by label from 0 to 
    metrics_by_label = {}
    
    for i in range(6):
        metrics_by_label[f"jaccard label {i}"] = np.round(jaccard_index[i], 4)
        # metrics_by_label[f"precision label {i}"] = np.round(precision[i], 4)
        # metrics_by_label[f"recall label {i}"] = np.round(recall[i], 4)
        metrics_by_label[f"f1 label {i}"] = np.round(f1[i], 4)
        # metrics_by_label[f"auc label {i}"] = np.round(auc_scores[i], 4)
        metrics_by_label[f"auprc label {i}"] = np.round(auprc_scores[i], 4)


    # Save the metrics in a dictionary
    metrics = {
        "micro_jaccard": round(micro_jaccard, 4),
        "micro_dice": round(micro_dice, 4),
        "auprc" : round(average_auprc, 4),
        "micro_f1": round(micro_f1, 4),
        "accuracy": round(accuracy, 4),
        # "weighted_jaccard": round(weighted_jaccard, 4),
        # "macro_auc" : round(auc, 4), # NOTE: over optimistic metric
        # "micro_precision": round(micro_precision, 4),
        # "micro_recall": round(micro_recall, 4),
        # "weighted_precision": round(weighted_precision, 4),
        # "macro_jaccard": round(macro_jaccard, 4),
        # "macro_precision": round(macro_precision, 4),
        # "macro_recall": round(macro_recall, 4),
        # "macro_f1": round(macro_f1, 4)
    }

    return metrics,  metrics_by_label

def plot_confusion_matrix(y_true, y_pred):

    # Flatten the arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    label_names = ['air', 'skin/scalp', 'skull', 'CSF', 'Gray Matter', 'White Matter']

    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(conf_matrix, cmap='Blues')

    # Add the matrix values
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, str(conf_matrix[i, j]), va='center', ha='center')

    # Set the title and labels
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    # Set the ticks
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))

    # Set the labels
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)

    # Show the plot
    plt.show()
