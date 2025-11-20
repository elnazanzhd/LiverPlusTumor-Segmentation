# Liver & Liver Tumor Segmentation with U-Net (3D CT)

<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJ5q0HW1gRLwAE6hxmp3tfqo1ROw3gQmFkfQ&s" 
       alt="Liver Segmentation" 
       width="500">
</p>



This notebook explores ** liver and liver tumor segmentation**
on 3D CT data using  **U-Net convolutional neural network**.

The main takeaway:

-   The model **learns liver anatomy well** (validation Dice up to
    **0.92** for liver),
-   but **struggles to localize tumors** (validation Dice for tumor
    stays low, best around **0.12**).

This contrast is intentional to show both what the model can do **and**
its current limitations on a challenging, highly imbalanced medical
task.

------------------------------------------------------------------------

## Dataset

-   **Source:** [3D Liver and Liver Tumor Segmentation
    (Kaggle)](https://www.kaggle.com/datasets/gauravduttakiit/3d-liver-and-liver-tumor-segmentation)\
-   **Modality:** CT scans\
-   **Targets:**
    -   Class 1 -- Liver\
    -   Class 2 -- Liver tumor

The dataset is used to build a **two-class segmentation problem**
(background vs. organ vs. tumor), with evaluation focused separately on:

-   **Liver Dice**
-   **Tumor Dice**
-   **Mean Dice** (average of liver and tumor)

------------------------------------------------------------------------

## Method

The notebook implements a **U-Net-style encoder--decoder architecture**:

-   **Encoder:** convolution + nonlinearity + downsampling\
-   **Decoder:** upsampling with skip connections\
-   **Loss:** Cross-Entropy + Dice loss\
-   **Training:** logs CE, DiceLoss, and per-class Dice across epochs

------------------------------------------------------------------------

## Results

### Key validation metrics across 15 epochs:

-   **Best validation mean Dice:** \~**0.52**\
-   **Best liver Dice:** \~**0.93**\
-   **Best tumor Dice:** \~**0.12**

#### Examples from training:

-   **Epoch 1**
    -   Liver Dice: 0.75\
    -   Tumor Dice: 0.00\
    -   Mean Dice: 0.38
-   **Epoch 6**
    -   Liver Dice: 0.91\
    -   Tumor Dice: 0.00\
    -   Mean Dice: 0.46
-   **Epoch 13**
    -   Liver Dice: 0.93\
    -   Tumor Dice: 0.07\
    -   Mean Dice: 0.50
-   **Epoch 15 (best mean Dice)**
    -   Liver Dice: 0.92\
    -   Tumor Dice: 0.12\
    -   Mean Dice: 0.52

### Interpretation

-   The model **reliably segments the liver**, reaching Dice scores
    above 0.9.
-   **Tumor segmentation remains poor**, with very low Dice values due
    to:
    -   extreme class imbalance,\
    -   small and irregular tumor regions,\
    -   limited training schedule and baseline architecture.

The project intentionally highlights both strengths and limitations to
provide a realistic baseline.

------------------------------------------------------------------------

## How to Use the Notebook

1.  Place the Kaggle dataset in the expected directory.\
2.  Open:

``` bash
jupyter notebook liver_segmentation.ipynb
```

3.  Run all cells:
    -   preprocessing\
    -   model building\
    -   training + validation\
    -   visualization

------------------------------------------------------------------------


## Future Improvements

-   Class-balanced, focal, or weighted losses\
-   Tumor-focused patches or ROI extraction\
-   3D U-Net or 2.5D hybrid approaches\
-   Augmentations tailored to tumor appearance\
-   Longer training or hyperparameter tuning

------------------------------------------------------------------------

## Acknowledgements

-   Dataset: [3D Liver and Liver Tumor Segmentation --
    Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/3d-liver-and-liver-tumor-segmentation)\
-   U-Net inspired by the original **U-Net for Biomedical Image
    Segmentation** paper.
