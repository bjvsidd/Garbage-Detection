# Garbage Detection & Classification App

A Streamlit-based web application that combines YOLO object detection with deep learning classification to identify and categorize different types of waste in images and videos.

## Features

- **Dual Input Support**: Process both images and videos
- **Real-time Detection**: Uses YOLO for fast garbage detection
- **Intelligent Classification**: VGG16-based classifier for waste categorization
- **Interactive Web Interface**: User-friendly Streamlit interface
- **Video Processing**: Frame-by-frame analysis with progress tracking
- **Download Support**: Save processed videos locally

## Waste Categories

The application can classify waste into 9 categories:
- Cardboard
- Food Organic
- Glass
- Metal
- Miscellaneous Trash
- Paper
- Plastic
- Textile Trash
- Vegetation

## Prerequisites

### System Requirements
- Python 3.8 or higher
- Sufficient RAM (4GB+ recommended for video processing)
- GPU support recommended for faster processing

### Required Models
Before running the application, ensure you have:
1. **YOLO Detection Model**: `Garbage_detection.pt`
2. **Classification Model**: `waste_classifier_vgg16 (1).h5`

Place these models in your desired directory and update the paths in the code.

## Installation

1. **Clone or download the application code**

2. **Install required dependencies:**
```bash
pip install streamlit opencv-python numpy pillow ultralytics tensorflow
```

3. **Update model paths in the code:**
```python
detection_model = YOLO(r"path/to/your/Garbage_detection.pt")
classifier_model = tf.keras.models.load_model(r"path/to/your/waste_classifier_vgg16.h5")
```

## Usage

### Starting the Application
```bash
streamlit run app.py
```

### Processing Images
1. Select "Image" from the input type dropdown
2. Upload an image file (PNG, JPG, JPEG)
3. View the original image and processed results with bounding boxes and labels

### Processing Videos
1. Select "Video" from the input type dropdown
2. Upload a video file (MP4, AVI, MOV)
3. Monitor processing progress
4. View the processed video in the browser
5. Download the processed video using the download button

## How It Works

### Detection Pipeline
1. **Input Processing**: Images/video frames are loaded and preprocessed
2. **Object Detection**: YOLO model identifies potential garbage objects
3. **Crop Extraction**: Detected regions are cropped from the original image
4. **Classification**: Each crop is classified using the VGG16-based model
5. **Annotation**: Results are overlaid on the original image with bounding boxes and labels

### Technical Details
- **Detection Confidence**: 0.3 threshold for YOLO predictions
- **Image Preprocessing**: Crops resized to 256x256 pixels and normalized
- **Video Output**: Maintains original frame rate and resolution
- **Color Space**: Handles BGR/RGB conversions for proper display

## Configuration

### Adjusting Detection Sensitivity
Modify the confidence threshold in the detection function:
```python
results = detection_model.predict(rgb, conf=0.3, verbose=False)  # Change 0.3 to desired value
```

### Customizing Class Names
Update the class_names list to match your classifier's output:
```python
class_names = ['your', 'custom', 'class', 'names']
```

## File Structure
```
project/
│
├── app.py                          # Main Streamlit application
├── models/
│   ├── Garbage_detection.pt        # YOLO detection model
│   └── waste_classifier_vgg16.h5   # Classification model
└── README.md                       # This file
```

## Performance Considerations

### For Images
- Processing time: 1-3 seconds per image
- Memory usage: Moderate

### For Videos
- Processing time: Depends on video length and resolution
- Memory usage: Higher due to frame-by-frame processing
- Recommendation: Use shorter videos or reduce resolution for faster processing

## Troubleshooting

### Common Issues

**Model Loading Errors**
- Verify model file paths are correct
- Ensure models are compatible with installed library versions

**Memory Issues**
- Reduce video resolution or length
- Close other applications to free up RAM

**Slow Processing**
- Consider using GPU acceleration
- Reduce detection confidence threshold
- Process smaller image/video sizes

**Import Errors**
```bash
pip install --upgrade streamlit opencv-python ultralytics tensorflow
```

## Dependencies

- `streamlit`: Web interface framework
- `opencv-python`: Computer vision operations
- `numpy`: Numerical computing
- `pillow`: Image processing
- `ultralytics`: YOLO model implementation
- `tensorflow`: Deep learning framework

## Model Information

### YOLO Detection Model
- Purpose: Detect garbage objects in images
- Input: RGB images
- Output: Bounding box coordinates and confidence scores

### VGG16 Classification Model
- Purpose: Classify detected garbage into categories
- Input: 256x256 RGB image crops
- Output: Probability distribution over waste categories

