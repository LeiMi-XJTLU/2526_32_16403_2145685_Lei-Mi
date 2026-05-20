# 2526_32_16403_2145685_Lei-Mi
                   TransUNet: A CNN-Transformer Hybrid Model for Radiotherapy Image Segmentation
This project implements brain tumor MRI image segmentation using TransUNet, completes data preprocessing, model training, result visualization and quantitative evaluation, and conducts experimental verification based on public medical imaging datasets.

Project Introduction
Brain tumor segmentation is an important task in medical image analysis, which helps doctors quickly locate tumor areas and assist clinical diagnosis. In this study, we adopt TransUNet combining Transformer and U-Net architecture to perform automatic segmentation on brain contrast-enhanced MRI images.
The experiment completes the whole process from original NIfTI medical data processing, dataset construction, model training, performance verification to result analysis. Although the final segmentation effect is not ideal due to limited data and category imbalance, this project builds a complete medical image segmentation experimental pipeline and summarizes practical problems and improvement directions in actual training.

Main Functions
Read and preprocess .nii.gz format brain MRI medical images
Realize one-to-one matching of original images and segmentation labels
Build TransUNet segmentation network model
Model training, verification and epoch curve drawing
Generate ROC curve, confusion matrix and segmentation contour overlay diagram
Calculate Dice, IoU, Precision, Recall, Accuracy and other evaluation indicators
Complete experimental result statistics and error analysis

Environment & Dependencies
This project was developed under the following environment:
- Python: `3.7.16`
- Conda: `23.7.4`
- Platform: `Linux x86_64`

Experimental Environment & Dependencies
 pip install -r requirements.txt

Dataset
Data source: TCIA public brain tumor MRI dataset (https://www.cancerimagingarchive.net)
Data format: .nii.gz multi-modal brain medical images + tumor segmentation labels
Since the original dataset is too large, this repository only uploads project code.
Users can download the original data from the official TCIA website and place it into the specified folder for running experiments.

How to Run
1.Download the TCIA dataset and place it in the tcia_data/ folder.
2.Preprocess the data and match image-label pairs using the provided script.
3.Train the TransUNet model
  python train.py
4.Run the test script to evaluate the model and generate metrics.
5.Execute the visualization script to reproduce the training curves, ROC curve, confusion matrix, and contour overlays.

Project Significance & Value
Built a complete and reusable medical image segmentation experimental pipeline, which can be directly migrated to other organ medical segmentation tasks
Verified the actual application limitations of TransUNet in small-sample medical datasets
Distinguished the difference between pixel-level classification ability and actual segmentation effect, which provides reference for subsequent related research
Summarized common problems and optimization ideas in brain tumor segmentation experiments, which is convenient for subsequent researchers to carry out improvement experiments

Future Improvement Directions
Expand the training dataset size and introduce more multi-center sample data
Use weighted loss function and sample balance strategy to alleviate category imbalance
Optimize prediction threshold and add morphological post-processing operations to remove wrong segmentation areas
Add more data augmentation methods suitable for medical images
Try lightweight improvement or mixed training strategy to enhance model generalization ability
