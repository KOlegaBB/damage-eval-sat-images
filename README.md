# Assessment of Damage to Ukrainian Infrastructure Using Satellite Imagery
This repository contains code and resources for training and evaluating models for building segmentation 
and damage classification from satellite imagery, focused on disaster response to war in Ukraine.

## Project Structure
```
damage-eval-sat-images/
│
├── classification/                      # Classification task (damage assessment)
│   ├── scripts/                         # Scripts for training, evaluation, transforms, etc.
│   │   ├── evaluation.py                # Evaluation utilities for classification
│   │   ├── k_folds.py                   # K-fold cross-validation logic
│   │   ├── loader.py                    # Data loading for classification
│   │   ├── models.py                    # Model architectures
│   │   ├── train.py                     # Training loop
│   │   ├── transforms.py                # Data augmentations
│   │   └── __init__.py
│   ├── building_classification.ipynb    # Experiments on damage classification
│   └── building_classification_finetuning.ipynb  # Classification fine-tuning on Ukrainian dataset
│
├── segmentation/                        # Segmentation task (building footprints)
│   ├── scripts/                         # Scripts for training, evaluation, transforms, etc.
│   │   ├── evaluation.py                # Evaluation metrics (e.g., IoU)
│   │   ├── instance_segm.py             # Instance segmentation utilities
│   │   ├── k_folds.py                   # K-fold logic for segmentation
│   │   ├── loader.py                    # Dataset loaders
│   │   ├── loss.py                      # Loss functions
│   │   ├── models.py                    # Segmentation model architectures
│   │   ├── regularize.py                # Regularization strategies
│   │   ├── train.py                     # Training pipeline
│   │   ├── transforms.py                # Augmentation and preprocessing
│   │   ├── utils.py                     # Helper functions
│   │   └── __init__.py
│   ├── building_segmentation.ipynb      # Segmentation experiment notebook
│   └── building_segmentation_finetuning.ipynb  # Segmentation fine-tuning on Ukrainian dataset
│
├── datasheet.md                         # Ukrainian Infrustructure dataset description
└── README.md                            # Project description (this file)
```

## Usage
All experiments were conducted on **Google Colab**. The experiment notebooks are available in the `classification` and `segmentation` directories.

### 1. **Download the Datasets**

Before using the notebooks, you need to download the following datasets:

- **xBD (XView2 Coding Challenge Dataset)** is available at:  
  [https://xview2.org/dataset](https://xview2.org/dataset)

- **Custom Ukrainian Damage Assessment Dataset** is available at:  
  [https://huggingface.co/datasets/KOlegaBB/damage_assessment_ukraine](https://huggingface.co/datasets/KOlegaBB/damage_assessment_ukraine)

### 2. **Running the Notebooks**

After downloading the datasets, you can run the notebooks. Please note that you may need to adjust the file paths in the notebooks to point to the directory where you store the code and data.

- **Classification Experiments**:  
  - `building_classification.ipynb`  
  - `building_classification_finetuning.ipynb`

- **Segmentation Experiments**:  
  - `building_segmentation.ipynb`  
  - `building_segmentation_finetuning.ipynb`

You can modify and run these notebooks for your experiments.
