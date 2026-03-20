
# CodeAlpha Tasks

This repository contains three projects developed as part of CodeAlpha tasks. Each project focuses on a distinct application of Artificial Intelligence, Natural Language Processing, and Computer Vision.

## Projects Overview

### 1. CodeAlpha FAQ Chatbot
A Streamlit-based FAQ Chatbot that uses Natural Language Processing to find the most relevant answers to user queries.
- **Technologies Used**: Python, Streamlit, NLTK, Scikit-learn (TF-IDF, Cosine Similarity).
- **How it works**: The app vectorizes the predefined FAQs and the user's question using TF-IDF, then computes the cosine similarity to retrieve the best-matching answer.
- **Location**: [`CodeAlpha_FAQ_Chatbot/`](./CodeAlpha_FAQ_Chatbot)
- **Usage**: Run `streamlit run app.py` inside the directory.

### 2. CodeAlpha Language Translation Tool
A simple, yet powerful language translation web application.
- **Technologies Used**: Python, Streamlit, Googletrans.
- **How it works**: Uses the Google Translate API to translate text between various supported languages including English, Tamil, Hindi, French, German, and Spanish.
- **Location**: [`CodeAlpha_Language_Translation_Tool/`](./CodeAlpha_Language_Translation_Tool)
- **Usage**: Run `streamlit run app.py` inside the directory.

### 3. CodeAlpha Object Detection and Tracking
A real-time computer vision script that detects and tracks objects using a webcam feed.
- **Technologies Used**: Python, OpenCV, PyTorch, Torchvision, Scipy, Numpy.
- **How it works**: Utilizes a pre-trained `FasterRCNN_ResNet50_FPN` model to detect objects (from the 91 COCO categories) and a custom Centroid Tracker to track their movements across frames.
- **Location**: [`CodeAlpha_object_detection_and _tracking/`](./CodeAlpha_object_detection_and _tracking)
- **Usage**: Run `python object_detection.py` inside the directory.

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd "CodeAlpha Tasks"
   ```

2. **Setup virtual environment** (Recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   Navigate to each project directory and install the respective requirements. For example:
   ```bash
   cd CodeAlpha_FAQ_Chatbot
   pip install -r requirements.txt
   ```
   *Note: For Object Detection and Tracking, ensure you have PyTorch and OpenCV installed.*
