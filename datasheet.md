Datasheet
=========


Motivation
----------

-   **For what purpose was the dataset created?**

    The dataset was created to support research on the impact of the war on Ukrainian infrastructure. It contains satellite and aerial images from before and after disaster events, with the goal of enabling the development and evaluation of models for automated damage assessment.
    There is a notable lack of publicly available, labeled datasets representing real-world post-disaster scenarios in Ukraine — particularly those capturing both urban and rural environments. This dataset aims to help fill that gap and foster research in humanitarian computer vision, with a focus on rapid response in crisis zones and equitable access to tools for affected communities.
    
Composition
-----------

-   **What do the instances that comprise the dataset represent (e.g.,
    documents, photos, people, countries)?** 

    The dataset consists of two parts, each designed for a different computer vision task related to the impact of war on Ukrainian infrastructure.
    The first part is intended for semantic segmentation of buildings. Each instance represents a pre-disaster satellite image
    from one of three Ukrainian locations — Kamianka, Popasna, and Yakovlivka — along with a corresponding binary mask indicating the location of buildings in the image.
    The second part is designed for building damage assessment. Each instance includes a pair of cropped images showing a single
    building before and after a disaster, a building mask, and a damage label assigned to one of four categories: undamaged, minor damage, major damage, or destroyed.
    Together, these instances support tasks such as segmentation, change detection, and multi-class classification in humanitarian response scenarios.

-   **How many instances are there in total (of each type, if
    appropriate)?**

    In total, the dataset includes 169 pre-disaster satellite images annotated for building segmentation and 2,219 individual building instances for damage assessment. 
    The table below summarizes the number of images and annotated buildings per location:
     
    | Location   | Oblast   | Pre-Disaster Date | Post-Disaster Date | Pre-Disaster Images | Annotated Buildings |
    |------------|----------|-------------------|---------------------|---------------------|----------------------|
    | Kamianka   | Kharkiv  | 09.07.2020        | 29.08.2022          | 62                  | 855                  |
    | Yakovlivka | Donetsk  | 06.09.2019        | 01.08.2022          | 40                  | 760                  |
    | Popasna    | Luhansk  | 15.10.2020        | 04.07.2023          | 67                  | 604                  |

-   **Does the dataset contain all possible instances or is it a sample
    (not necessarily random) of instances from a larger set?**

    This dataset represents a sample of building instances from the locations of Kamianka, Popasna, and Yakovlivka, and does not include all possible instances. Specifically, only part of the Popasna region was annotated, and images that did not contain buildings were excluded from the dataset. Additionally, while the goal was to annotate as many buildings as possible in each image, we cannot guarantee that every building in the selected regions was annotated during the annotation process. Therefore, the dataset may not fully represent all buildings or every possible disaster scenario in these regions.
    The sample was selected to represent key locations impacted by the conflict, but it should be noted that it is not fully exhaustive of all buildings or areas within the broader conflict zone. The annotations were primarily based on available satellite imagery

-   **Are there recommended data splits (e.g., training,
    development/validation, testing)?**

    Yes, both the segmentation and classification subsets of the dataset are split into folds for k-fold cross-validation. Each subset is divided into 10 folds, with each fold containing approximately 10% of the data.
    
    For the segmentation task, folds were constructed to ensure that the majority of instances in each fold contain non-blank (i.e., building-present) masks, minimizing the chance of evaluation on irrelevant or empty regions.
    
    For the classification (damage assessment) task, folds were stratified to maintain an approximately equal distribution of damage labels (undamaged, minor damage, major damage, destroyed) across all splits, enabling fair and balanced model evaluation.
-   **Are there any errors, sources of noise, or redundancies in the
    dataset?**

    Yes, first, pre-disaster images are of lower quality than post-disaster images, due to limitations in acquiring historical data.
    Additionally, the annotation process was manually carried out by a small team without the involvement of subject matter experts, which led to inconsistencies in the labeling of damage levels. 
    As a result, some labels may not fully represent the actual conditions on the ground.

Collection process
------------------

-   **How was the data associated with each instance acquired?**

    The dataset was created using publicly available satellite imagery.
    Post-disaster images were extracted using QGIS from Google Maps, which provides georeferenced recent imagery.
    Pre-disaster images were acquired from Google Earth’s historical imagery archive, which required manual georeferencing due to the lack of embedded spatial metadata.
    Damage labels were manually assigned to each building using visual inspection, following label definitions from the xBD dataset.
    No automated or model-based inference was used during labeling. While the process aimed for accuracy, the annotations were not validated by subject matter experts.
    
-   **What mechanisms or procedures were used to collect the data (e.g.,
    hardware apparatus or sensor, manual human curation, software
    program, software API)?**
    Satellite images were collected using QGIS and Google Earth software.QGIS was used to extract and georeference post-disaster imagery from Google Maps.
    For pre-disaster imagery, historical snapshots from Google Earth were manually georeferenced using QGIS.
    All images were cropped and scaled to a spatial resolution of approximately 0.33 meters per pixel.
    Building segmentation and damage annotations were created manually by a small team, based on visual assessment of pre- and post-disaster building crops.

Preprocessing / cleaning / labeling
-----------------------------------

-   **Was any preprocessing/cleaning/labeling of the data done (e.g.,
    discretization or bucketing, tokenization, part-of-speech tagging,
    SIFT feature extraction, removal of instances, processing of missing
    values)?**

    Yes. All images were manually georeferenced and cropped to 1024×1024 pixels with a consistent spatial resolution of approximately 0.33 meters per pixel.
    Building footprints were manually annotated to create binary segmentation masks.
    Each building instance was also assigned a discrete damage level based on the xBD label definitions: Undamaged, Minor Damage, Major Damage, or Destroyed.
    Some images not containing buildings were excluded during preprocessing. Additionally, images or buildings with low visibility or severe occlusion were manually
    filtered out where appropriate.
-   **Was the "raw" data saved in addition to the
    preprocessed/cleaned/labeled data (e.g., to support unanticipated
    future uses)?**

    No.
-   **Is the software used to preprocess/clean/label the instances
    available?** 
    Preprocessing was performed using QGIS, an open-source geographic information system,
    and manual annotation was conducted using CVAT (Computer Vision Annotation Tool), available at: https://www.cvat.ai/
Uses
----

-   **Has the dataset been used for any tasks already?** 

    Yes. The dataset has been used to inference, fine-tune, and evaluate deep learning models for damage assessment from satellite imagery.
    Specifically, it was used for two primary tasks:
    
    1. Building segmentation – identifying and extracting building footprints from satellite images to localize relevant structures for further analysis.
    
    2. Damage assessment – classifying the level of damage for each segmented building into four categories (Undamaged, Minor Damage, Major Damage, or Destroyed) or predicting a continuous severity score using regression.
-   **Is there a repository that links to any or all papers or systems
    that use the dataset?**

    Yes. Associated code for model training, evaluation, and experimental results are publicly available at:
    https://github.com/KOlegaBB/damage-eval-sat-images/
-   **Is there anything about the composition of the dataset or the way
    it was collected and preprocessed/cleaned/labeled that might impact
    future uses?**

    Yes. The dataset was manually annotated, and damage assessments are subject to human interpretation and visual clarity of the satellite imagery.
    Differences in image resolution, lighting, or occlusion may affect label quality. 
Distribution
------------

-   **Will the dataset be distributed to third parties outside of the
    entity (e.g., company, institution, organization) on behalf of which
    the dataset was created?**

    Yes. The dataset will be distributed publicly for research and development purposes.
    It is intended to support disaster response, damage assessment research, and benchmarking of computer vision models in remote sensing contexts.
-   **How will the dataset will be distributed (e.g., tarball on
    website, API, GitHub)?**

    The dataset is available via Hugging Face at https://huggingface.co/datasets/KOlegaBB/damage_assessment_ukraine. 

-   **Will the dataset be distributed under a copyright or other
    intellectual property (IP) license, and/or under applicable terms of
    use (ToU)?**

    The underlying satellite imagery is sourced from Google Maps, and is governed by Google's Terms of Service,
    which restrict redistribution and commercial use of the imagery. For details, see Google Maps/Google Earth Additional Terms of Service.
-   **Have any third parties imposed IP-based or other restrictions on
    the data associated with the instances?**

    
    Yes. The satellite imagery used in the dataset originates from Google Maps, and use of this imagery is subject to Google’s licensing terms. 
