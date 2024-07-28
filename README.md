# Lane-Detection-System

This project is a lane detection and classification system designed to identify and classify lane markings on the road. The system utilizes deep learning models to process video input, detecting lanes and classifying them into various types, such as "Double Solid Yellow Lane," "Dashed White Lane," and "Solid White Lane."

https://github.com/user-attachments/assets/fe7927a8-4efb-4aba-8aeb-68d8e5407c84

## Overview

The project consists of two main components:

1. **Lane Detection**: Uses deep learning models like U-Net and LaneNet to detect lane markings in real-time from video frames.
2. **Lane Classification**: A separate deep learning model classifies the detected lanes into different categories based on their markings.

## Features

- **Real-time Processing**: The system processes video frames in real-time, overlaying detected lanes and their classifications on the original video.
- **Multi-Class Classification**: It can classify lanes into multiple categories, enhancing the understanding of road conditions.
- **Integrated Output**: The final output video includes both lane detection visualizations and lane type annotations.

Deployment on Hardware
The system is also deployed on a Jetson Nano, a compact microprocessor designed for AI applications. This deployment enables real-time lane detection and classification directly on hardware, making it suitable for embedded systems and automotive applications.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lanedetection.git

2. Navigate to the project directory:
   ```bash
   cd lanedetection
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

To run the system on a video file, use the following command:

```bash
python Lane_detection_System.py
## change model paths and input output video setting according to your computer
   
