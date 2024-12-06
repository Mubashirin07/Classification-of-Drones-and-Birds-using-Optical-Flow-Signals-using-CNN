# Classification-of-Drones-and-Birds-using-Optical-Flow-Signals-using-CNN
This repository contains a machine learning-based system to classify birds and drones using optical flow signals extracted from video data. The project focuses on leveraging motion-based features to enable accurate classification, with applications in airspace security and wildlife monitoring.

##Overview
The system employs optical flow analysis to generate one-dimensional motion signals from video data, which are then classified using a 1D Convolutional Neural Network (CNN). The entire pipeline includes:

1.Data preprocessing with OpenCV to extract frames and calculate optical flow.
2.Creation of a synthetic dataset with 500 samples (250 birds, 250 drones) from 161 video inputs.
3.Model training and evaluation, achieving 77% accuracy.
4.Flask-based frontend for seamless video upload and real-time predictions.

##Features
*Data Preprocessing:
Extracted motion signals from 161 videos using OpenCV's frame processing and the Farneback optical flow method.
Standardized signal lengths for compatibility with machine learning models.

*Synthetic Dataset:
Generated a balanced dataset of 500 samples to improve model robustness.
*1D CNN Model:
Designed a convolutional neural network optimized for sequential motion data.
Achieved 77% classification accuracy.

*Frontend Integration:
Built a Flask-based user interface for video upload and real-time classification.
Outputs predictions along with a visualized spectrogram for interpretability.

##Pipeline

1.Data Acquisition:
Extract frames from uploaded video files.
Compute optical flow signals to capture motion patterns.

2.Model Training:
Train a 1D CNN model with the synthetic dataset.
Optimize using binary cross-entropy loss and the Adam optimizer.

3.Real-Time Inference:
Upload a video via the web interface.
Process the video, classify it as a bird or drone, and display results.

##Results
*Classification Accuracy: 77%
*Processing Time: 20–30 seconds per video for upload, processing, and prediction.

##Applications

*Airspace Security: Distinguishing drones from birds in restricted areas.
*Wildlife Monitoring: Monitoring natural habitats by identifying aerial entities.
*Surveillance: Enhancing automation in aerial object detection.

##Future Goals
*Expand the dataset with more diverse bird and drone videos.
*Explore advanced architectures like CNN-LSTM and Transformers for improved accuracy.
*Optimize for real-time performance on edge devices.
*Integrate multi-sensor data for enhanced detection in challenging conditions.
*Broaden scope to multi-class classification for various aerial objects.
