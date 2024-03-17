Practical Considerations

Pixel Imbalance: In segmentation, often there is a large imbalance between the number of pixels belonging to the segment of interest and the background. This should be taken into account when interpreting the ROC curve and AUC score.

- I have to mention the pixel imbalance
- TODO:

Spatial Information: ROC and AUC primarily assess pixel-wise classification performance and do not directly consider the spatial contiguity or the shape of the segmented regions. Therefore, these metrics should be complemented with other measures that take into account the spatial coherence and accuracy of the segmented shapes (e.g., Dice coefficient, IoU (Intersection over Union)).

- Check message with Kaylen to make decision but try to remember I don't have mind to thing about this now

Micro-average: Use this if you want to weight each instance or prediction equally. This is useful if you're concerned about the performance across all classes collectively and especially if you have class imbalance.

Weighted-average: Use this when class imbalance is present and you want to weight the performance of each class by its presence in the dataset. This gives a better representation of the model's overall performance.
- Weight is as a macro that address unbalanced problems

Yes, micro-averaged metrics can implicitly handle data imbalance. The reason for this is that micro-averaging combines the contributions of all classes to compute the overall metric. This means that the metric is calculated by summing the individual true positives, false positives, and false negatives across all classes before calculating the metric. Here's how this property of micro-averaging addresses data imbalance:

Equal weighting of each instance: In micro-averaging, each instance contributes equally to the final metric, regardless of the class it belongs to. This means that classes with more instances have a greater impact on the metric, naturally reflecting the class distribution in the dataset.

Effect on larger classes: Because larger classes contribute more to the confusion matrix sums in micro-averaging, their performance has a bigger impact on the calculated metric. This is suitable for datasets where the performance on the common classes is more important.

Handling class imbalance: In imbalanced datasets, where some classes have much fewer samples than others, micro-averaging ensures that the abundant classes don’t dominate the evaluation metric unfairly. This is because the performance across all samples, including those from minority classes, is accounted for in the overall calculation.

In summary, micro-averaged metrics naturally take into account the class distribution of the dataset, making them particularly useful for evaluating models on imbalanced datasets. They focus on the overall performance across all classes, giving a more comprehensive view of the model's ability to predict all classes correctly.


# NOTE:
Connected Component Labeling (CCL) is a technique used in computer vision to detect connected regions in binary digital images. Here are some popular CCL algorithms:

Two-Pass Algorithm: This is the most common CCL algorithm. It scans the image twice. In the first pass, it assigns temporary labels and records equivalences. In the second pass, it replaces each temporary label by the label of its equivalence class.