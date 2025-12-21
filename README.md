# Histolab
## LC25000 Histopathology Image Classifier

A deep learning-powered web application for automated classification of lung and colon histopathology images. Built with Streamlit and TensorFlow, this tool provides real-time tissue classification with confidence scoring and detailed probability distributions.

## Overview

This application uses a convolutional neural network trained on the LC25000 dataset to classify histopathology images into five distinct categories:

- **Lung Adenocarcinoma** - Most common type of lung cancer
- **Lung Squamous Cell Carcinoma** - Cancer in squamous cells of airways
- **Lung Benign Tissue** - Normal, healthy lung tissue
- **Colon Adenocarcinoma** - Cancer in glandular cells of colon
- **Colon Benign Tissue** - Normal, healthy colon tissue

## Features

- **Real-time Classification** - Upload and analyze histopathology images instantly
- **Confidence Thresholding** - 90% confidence threshold to ensure reliable predictions
- **Probability Distribution** - View classification probabilities for all tissue types
- **Model Performance** - Achieves 96% accuracy on the LC25000 dataset
- **Professional UI** - Clean, modern interface with intuitive navigation
- **Hugging Face Integration** - Seamless model loading from Hugging Face Hub

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd histopathology-classifier
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit>=1.28.0
tensorflow>=2.13.0
numpy>=1.24.0
Pillow>=10.0.0
huggingface-hub>=0.17.0
```

## Configuration

### Hugging Face Model Setup

Edit the following variables in the code to point to your model:

```python
HF_REPO_ID = "jeremysean/histolab"  # Your Hugging Face repository
HF_FILENAME = "lc25000_best.keras"   # Your model filename
```

### Confidence Threshold

The default confidence threshold is set to 90%. To modify this, change the following constant:

```python
CONFIDENCE_THRESHOLD = 0.90  # Adjust between 0.0 and 1.0
```

## Usage

1. **Upload Image** - Click the upload button and select a histopathology image (JPEG or PNG format)
2. **Analysis** - The system automatically processes and classifies the image
3. **Review Results** - Examine the predicted tissue type, confidence score, and probability distribution
4. **Interpret** - Results below the confidence threshold will indicate unrecognized tissue

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- Recommended size: 224×224 pixels

## Model Information

### Architecture

- Base architecture: Convolutional Neural Network
- Input size: 224×224×3 (RGB images)
- Output classes: 5 tissue types
- Model format: Keras (.keras)

### Performance Metrics

- Training accuracy: 96%
- Dataset size: 25,000 images
- Image resolution: 224×224 pixels
- Color channels: RGB (3 channels)

### Dataset

The LC25000 dataset consists of 25,000 histopathological images of lung and colon tissue, distributed equally across five classes:

- 5,000 images per class
- High-resolution microscopy images
- Pre-labeled by medical professionals
- Balanced class distribution

## Technical Details

### Image Preprocessing

Images undergo the following preprocessing steps:

1. Resize to 224×224 pixels
2. Convert to RGB format (if grayscale or RGBA)
3. Normalize pixel values to float32
4. Add batch dimension for model input

### Classification Pipeline

1. Image upload and validation
2. Preprocessing and normalization
3. Model inference
4. Confidence thresholding
5. Result presentation with probability distribution

## Disclaimer

**IMPORTANT**: This application is designed for research and educational purposes only. It is not intended for clinical diagnosis or medical decision-making. All predictions should be verified by qualified medical professionals. Do not use this tool as a substitute for professional medical advice, diagnosis, or treatment.

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Troubleshooting

### Model Loading Issues

If the model fails to load from Hugging Face:

- Verify your internet connection
- Check that the repository ID and filename are correct
- Ensure you have the latest version of `huggingface-hub` installed
- Verify the model file exists in the specified Hugging Face repository

### Image Upload Errors

If images fail to process:

- Ensure the image format is supported (JPEG or PNG)
- Check that the image is not corrupted
- Verify the image has sufficient resolution
- Try converting the image to RGB format before upload

## Contributing

Contributions are welcome. Please ensure all code follows Python best practices and includes appropriate documentation.

## License

This project is provided as-is for research and educational purposes. Please refer to the LC25000 dataset license for data usage terms.

## Acknowledgments

- LC25000 Dataset creators
- Hugging Face for model hosting infrastructure
- Streamlit for the web framework
- TensorFlow team for the deep learning framework

## Contact

For questions, issues, or contributions, please open an issue in the repository or contact the maintainers.

---

**Version**: 1.0.0  
**Last Updated**: December 2025  
**Status**: Active Development