# PPE Detection on Construction Sites using improved YOLOv8

## Overview
This project is part of the research titled *"Enhancing Worker Safety: Real-time Automated Detection of Personal Protective Equipment to Prevent Falls from Heights at Construction Sites Using Improved YOLOv8 and Edge Devices"*, authored by **Kim D.** and **Xiong S.** (2024). The research has been accepted and is in press at the *Journal of Construction Engineering and Management*.
ref: 

This work focuses on enhancing construction site safety by utilizing portable AI and computer vision techniques for real-time automated detection of personal protective equipment (PPE) usage and fall incidents. I introduce an improved YOLOv8 model, optimized for edge devices, to accurately detect proper PPE usage (helmet, nohelmet, harness, noharness, lanyard) and prevent injuries from falls.

### Key Features:
- **YOLOv8 Model Improvements**: Integration of Coordinate Attention module, Ghost Convolution module, Transfer Learning, and Merge-Non-Maximum Suppression to achieve a balance between detection accuracy and lightweight design.
- **Custom Dataset**: A large-scale, multi-class PPE dataset has been constructed for model training, though it is not publicly available. Sample images are provided for reference.
- **Real-Time Detection**: The model is designed to run on edge devices, enabling real-time detection of PPE usage and fall incidents on construction sites.

## Directory Structure

### 1. `ppe_custom_detect` Folder
This folder contains the custom configurations, datasets, model weights, and code for training, testing, and prediction.

- **`custom_cfg/`**: YOLOv8 model configuration YAML files.
    - `original_yolov8/`: YOLOv8s with a custom number of classes.
    - `yolov8_combined/`: Improved YOLOv8 with Coordinate Attention and Ghost Convolution modules.
    - `yolov8_etc/`: Experimental changes to YOLOv8.
    - `yolov8_lightweight/`: Lightweight version of YOLOv8.
    - `yolov8_with_attention/`: YOLOv8 with various attention modules for better accuracy.

- **`custom_dataset/`**: Custom dataset for PPE detection.
    - `train/`, `val/`, `test/`: Folders containing training, validation, and test images.
    - `custom_ppe_dataset.yaml`: Path configuration for the custom dataset.

- **`custom_weight/`**: Model weight files for the improved YOLOv8.
    - `coco_custom_yolov8s_add_ca_backbone_g_neck.pt`: improved YOLOv8s trained on the COCO dataset.
    - `ppe_custom_yolov8s_add_ca_backbone_g_neck.pt`: improved YOLOv8 trained on the custom PPE dataset with transfer learning from COCO.

- **`custom_yolov8_ppe_train_detect.ipynb`**: Jupyter notebook containing code for training, testing, and predicting PPE detection using YOLOv8.

- **`sample_ppe.jpg`**: A sample image from the custom dataset.

### 2. `ultralytics/nn` Folder
This folder contains the modules used in YOLOv8. You can add or modify modules as needed.

- **`modules.py`**: Contains various modules used in YOLOv8, including custom modules like C2fGhost, CA, and ECA. If you add new modules here, ensure to include the module names in `tasks.py`.

### 3. `ultralytics/yolo` Folder
This folder contains configuration files, utilities, and other important scripts for YOLOv8.

- **`cfg/`**: Contains `default.yaml` file, which includes default training settings and hyperparameters. You can modify hyperparameters here as needed.

- **`utils/`**: Utility functions used in YOLOv8.
    - `metrics.py`: Contains the Intersection over Union (IoU) function. Custom IoU functions like SIoU can be added here.
    - `ops.py`: Contains the Non-Maximum Suppression (NMS) function. You can enable merge-NMS by setting `merge=True` in line 197.
    - `loss.py`: Contains the bounding box loss function. You can modify the loss function (e.g., CIoU, SIoU) here.

### 4. `yolov8_eigen_cam` Folder
This folder contains scripts for visualizing model predictions using Eigen CAM.

- **`yolov8_heatmap.py`**: Execute this script in Anaconda Prompt to generate visualized images using Eigen CAM. Parameters can be adjusted in the `get_params()` function.

- **`yolov8_visualize.ipynb`**: Jupyter notebook for visualizing images and model predictions.

## How to Use

### 1. Training the Model
Use the `custom_yolov8_ppe_train_detect.ipynb` notebook to train the YOLOv8 model on the custom PPE dataset. Ensure the paths in `custom_ppe_dataset.yaml` are correctly set to your dataset location.

### 2. Testing and Prediction
After training, you can test and predict using the same notebook. The model weights (`ppe_custom_yolov8s_add_ca_backbone_g_neck.pt`) will be used for inference.

### 3. Visualization
To visualize model predictions, use the `yolov8_visualize.ipynb` notebook. You can generate heatmaps using the `yolov8_heatmap.py` script.

### 4. Modifying Modules
If you need to add or modify modules, edit the `modules.py` file in the `ultralytics/nn/` folder. Ensure to update `tasks.py` if new modules are added.

### 5. Hyperparameter Tuning
Modify the `default.yaml` file in the `cfg/` folder to adjust training settings and hyperparameters.

## Notes
- This project utilizes YOLOv8 version Ultralytics-8.0.40.
- The custom dataset is proprietary and only sample images are provided in the repository.

## Acknowledgments
We acknowledge the use of the YOLOv8 framework from Ultralytics and thank the contributors to the open-source modules integrated into this project. This work is based on the research by **Kim D.** and **Xiong S.** (2024) as part of the paper titled *"Enhancing Worker Safety: Real-time Automated Detection of Personal Protective Equipment to Prevent Falls from Heights at Construction Sites Using Improved YOLOv8 and Edge Devices,"* accepted in the *Journal of Construction Engineering and Management*.