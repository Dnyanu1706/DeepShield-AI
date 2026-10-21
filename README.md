# DeepShield AI — AI-Based Deepfake Image Detection System

DeepShield AI is a web-based deepfake image detection application that uses deep learning to classify uploaded images as **Real** or **Fake**. The system uses a fine-tuned MobileNetV2 convolutional neural network (CNN) and provides prediction results, confidence scores, prediction history, dashboard analytics, and PDF report generation.

## Features

- **Image Upload:** Upload an image for analysis.
- **Deepfake Detection:** Classify images as Real or Fake using a trained deep learning model.
- **Confidence Score:** Display a model-generated confidence score alongside the prediction.
- **Prediction History:** Store and view previous prediction records using SQLite.
- **Dashboard Analytics:** Display prediction statistics and historical results.
- **PDF Report:** Generate and download a report of prediction records.
- **Web Interface:** Access the system through a Flask-based web application.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Flask | Web application and routing |
| TensorFlow / Keras | Deep learning model |
| MobileNetV2 | Image classification architecture |
| OpenCV / Pillow | Image processing, as used in the application |
| NumPy | Numerical operations |
| SQLite | Store prediction history |
| HTML, CSS, JavaScript | Frontend interface |
| ReportLab | PDF report generation |

## Project Structure

```text
DeepShield-AI/
├── app.py
├── requirements.txt
├── README.md
├── database.db
├── models/
│   └── image_model_mobilenet_finetune_v2.keras
├── training/
│   └── train_mobilenet_finetune.py
├── Dataset/
│   ├── Train/
│   │   ├── fake/
│   │   └── real/
│   └── Test/
│       ├── fake/
│       └── real/
├── templates/
├── static/
│   └── uploads/
└── .venv/
```

*Note: This is the expected structure. Adjust the paths and filenames to match the actual project. Do not upload the virtual environment or large datasets unnecessarily.*

## System Requirements

- Python 3.10 or another version compatible with the installed TensorFlow release
- pip package manager
- Visual Studio Code or another Python IDE
- A modern web browser
- Sufficient memory and disk space for the trained model

## Installation and Setup

### 1. Clone or Download the Project

If the project is hosted on GitHub:

```bash
git clone <YOUR_REPOSITORY_URL>
cd DeepShield-AI
```

Alternatively, open the existing project folder in VS Code.

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt or your existing activated virtual environment.

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that the installed packages are compatible with your Python and TensorFlow versions.

### 4. Verify the Model File

Confirm that this file exists:

```text
models/image_model_mobilenet_finetune_v2.keras
```

The application needs the trained model to perform predictions. If the model is not included in the repository, obtain it from the project's authorized storage.

### 5. Run the Application

```powershell
python app.py
```

Open the local address shown in the terminal, typically:

```text
http://127.0.0.1:5000
```

## How to Use the Application

1. Start the Flask application.
2. Open the application in your browser.
3. Upload a supported image.
4. Submit the image for analysis.
5. View the predicted class and confidence score.
6. Open the History page to review saved predictions.
7. Open the Dashboard to view prediction statistics.
8. Download the PDF report when required.

## Model Architecture

DeepShield AI uses MobileNetV2, a convolutional neural network architecture designed for efficient image classification.

The model was fine-tuned using a labeled dataset containing Real and Fake images. Images are resized to 160 × 160 pixels before prediction. The model includes an internal rescaling layer that maps pixel values to the range expected by the network.

### Prediction Process

1. The user uploads an image.
2. The application validates and reads the image.
3. The image is converted to RGB and resized to 160 × 160 pixels.
4. The processed image is passed to the trained model.
5. The model produces a prediction score.
6. The application maps the score to the Real or Fake class.
7. The result and associated confidence score are displayed.
8. The prediction record is stored in the database.

**Important:** The displayed confidence score is model-derived and should not be interpreted as a calibrated probability that the prediction is correct.

## Dataset

The current dataset is organized into training and testing directories:

```text
Dataset/
├── Train/
│   ├── fake/
│   └── real/
└── Test/
    ├── fake/
    └── real/
```

The test set contains 857 images:

- Fake images: 457
- Real images: 400

The dataset should contain correctly labeled images. Dataset sources and licenses should be documented before public distribution.

## Model Evaluation Results

The fine-tuned model was evaluated on the separate test dataset.

| Metric | Result |
|---|---:|
| Test Accuracy | 69.08% |
| AUC | 0.7752 |
| Fake Precision | 72.12% |
| Fake Recall | 68.49% |
| Fake F1-score | 70.26% |
| Real Precision | 65.96% |
| Real Recall | 69.75% |
| Real F1-score | 67.80% |

### Confusion Matrix

Rows represent the actual class, and columns represent the predicted class.

| Actual / Predicted | Fake | Real |
|---|---:|---:|
| Fake | 313 | 144 |
| Real | 121 | 279 |

The model achieved 69.08% accuracy on this test set. These results indicate that further improvements may be needed for reliable real-world use. Performance can vary with image quality, dataset composition, compression, and unseen generation methods.

## Database and Reporting

The application uses SQLite to store prediction history. The database can include details such as:

- Image filename
- Media type
- Predicted class
- Confidence score
- Prediction timestamp

The dashboard summarizes prediction activity, while the PDF feature provides a downloadable report of stored records.

## Testing

The following application features have been tested:

- Image upload and prediction
- Prediction history
- Dashboard analytics
- PDF report generation

Additional validation should cover unsupported file types, oversized files, corrupted images, empty uploads, and database persistence after restarting the application.

## Limitations

- The model may misclassify genuine or manipulated images.
- Performance depends on the quality and diversity of the training dataset.
- New image-generation techniques may not be represented in the dataset.
- A confidence score does not guarantee correctness.
- Results should not be used as the sole basis for legal, security, or other high-impact decisions.

## Future Enhancements

- Expand the dataset with diverse real and synthetic images.
- Improve model generalization and evaluate additional architectures.
- Add explainability tools such as Grad-CAM.
- Improve file validation and application security.
- Add more detailed analytics and export options.
- Extend detection to video or audio in a future version.

## Conclusion

DeepShield AI demonstrates how deep learning can be integrated into a web application for deepfake image classification. It combines image upload, model inference, prediction history, dashboard analytics, and PDF reporting in one system. The current evaluation provides a baseline for future improvements in accuracy and reliability.

## Disclaimer

DeepShield AI is an academic project intended for educational and research purposes. Its predictions are not definitive proof that an image is authentic or manipulated.

## Author

**Project:** AI-Based Deepfake Image Detection System  
**Application Name:** DeepShield AI

Update this section with the project team's names, institution, guide, and repository URL before final submission.
