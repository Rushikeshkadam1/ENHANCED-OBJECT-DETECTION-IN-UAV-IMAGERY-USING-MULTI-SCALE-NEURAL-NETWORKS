# Drone-YOLO: A Multi-Scale Neural Network for Enhanced Object Detection in UAV Imagery

## Aim
The objective of this research is to improve object detection in unmanned aerial vehicle (UAV) images by proposing **Drone-YOLO**, a set of neural network methods based on the YOLOv8 architecture. The study focuses on enhancing detection performance for small objects and densely distributed targets in drone images, addressing challenges like large scene sizes and complex backgrounds.

## Dataset

- **Name:** VisDrone2019 Dataset
- **Size:**
  - 6471 images (training)
  - 548 images (validation)
  - 1610 images (test)
  - 1580 images (competition)
- **Image Characteristics:**
  - Image sizes ranging from 2000x1500 to 480x360 pixels
  - Captured from UAVs with outdoor scenes such as streets, parks, and residential areas
- **Objects:** 10 types including pedestrians, cars, buses, motorcycles, and bicycles
- **Conditions:** Images captured under various lighting conditions (day, night, glare, etc.)

## Methodology

The research involves a series of enhancements to the YOLOv8 architecture to address specific challenges in drone image detection.

- **Backbone:** 
  - Uses **RepVGG modules** in place of traditional convolution layers for improved multi-scale feature learning
  - The backbone consists of **C2f modules** for effective gradient flow and multi-scale feature extraction
  
- **Neck:**
  - Features a three-layer **PAFPN (Path Aggregation Network)** and a **sandwich-fusion module** for enhanced detection of small objects
  - Combines spatial information from different layers for better object localization

- **Detection Heads:**
  - Separate heads for detecting objects of different sizes (tiny, small, medium, and large)

- **Training Setup:**
  - Models were trained on resized images (640x640 pixels) for 300 epochs using the PyTorch framework
  - Hardware support provided by **NVIDIA 3080ti GPUs**

## Model Architecture

**Drone-YOLO** is based on YOLOv8 with several architectural improvements:

- **Backbone:** 
  - **RepVGG modules** for downsampling, with **C2f structures** at each stage for feature extraction
  
- **Neck:** 
  - **PAFPN** with three-layer sandwich-fusion modules for enhanced multi-scale detection, especially for small objects

- **Detection Heads:** 
  - Multi-scale detection heads with layers handling tiny, small, medium, and large objects

- **Variations:** 
  - Five versions—Drone-YOLO (large, medium, small, tiny, and nano)—with decreasing parameters and complexity but still achieving high accuracy

## Python Libraries Used

- PyTorch
- CUDA
- Pandas
- NumPy
- OpenCV
- scikit-learn
- Matplotlib
- ultralytics (for YOLOv8 implementation)

## Evaluation Metrics

- **Precision (P):** Proportion of correctly detected objects among all detected objects
- **Recall (R):** Proportion of correctly detected objects among all actual objects
- **AP (Average Precision):** Measures precision and recall for different Intersection over Union (IoU) thresholds
- **mAP (mean Average Precision):** Average of AP across all object categories at different IoU thresholds (e.g., mAP@0.5, mAP@0.75, mAP@0.95)

## Results

- **Drone-YOLO (large)** achieved **mAP@0.5 of 40.7%** on the VisDrone2019-test dataset, a **9.1% improvement** over the previous best method (TPH-YOLOv5)
- **Drone-YOLO (tiny),** with only **5.35M parameters,** performed comparably to YOLOv8 models, making it highly suitable for edge computing devices like drones
- Ablation experiments showed significant improvements in small object detection with the added small-size detection head and sandwich-fusion module

## Conclusion

**Drone-YOLO** models provide a significant improvement in UAV image object detection, particularly for small and densely packed objects. The models are scalable for both desktop post-flight processing (high accuracy) and real-time embedded UAV systems (efficient with fewer parameters). The proposed enhancements make it highly suitable for drone-based object detection tasks in various environments.

## Future Work

- Integrating attention mechanisms like **CBAM** or **Swin Transformer** could further enhance detection accuracy, especially in complex environments
- Optimizing models for real-time performance on next-gen embedded platforms like **NVIDIA Jetson Xavier**
- Expanding the methodology to handle additional tasks like **object tracking** or **semantic segmentation** in UAV imagery
