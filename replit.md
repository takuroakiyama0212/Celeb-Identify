# Celebrity Image Classifier

## Overview
A Python-based celebrity image classification application that uses deep learning to analyze facial features and identify which famous celebrities a person resembles. Built with Streamlit for an interactive web interface.

## Current Features
- **Image Upload**: Drag-and-drop support for JPG, PNG formats
- **Face Detection**: OpenCV Haar Cascade for detecting faces in images
- **Deep Learning Analysis**: MobileNetV2 neural network for feature extraction
- **Top 5 Celebrity Matches**: Display predictions with confidence scores
- **Batch Processing**: Analyze multiple images at once with progress tracking
- **Confidence Threshold**: Filter results by minimum confidence level
- **Celebrity Info Cards**: Detailed bios and filmography for matched celebrities
- **Dual Mode**: Works with OpenAI Vision API (if key provided) or standalone ML

## Tech Stack
- **Frontend**: Streamlit
- **ML Framework**: TensorFlow with MobileNetV2
- **Face Detection**: OpenCV
- **Optional AI**: OpenAI GPT-5 Vision (requires API key)
- **Language**: Python 3.11

## Project Structure
```
├── app.py                 # Main Streamlit application
├── pyproject.toml         # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit server configuration
└── replit.md              # This documentation file
```

## Running the App
The application runs on port 5000:
```bash
streamlit run app.py --server.port 5000
```

## How It Works
1. User uploads a photo (single or batch)
2. OpenCV detects faces in the image
3. Face region is extracted and preprocessed
4. MobileNetV2 extracts 1280-dimensional feature vectors
5. Features are analyzed against celebrity profiles
6. Top 5 matches displayed with confidence percentages

## Celebrity Database
Contains 40 famous celebrities including:
- Leonardo DiCaprio, Brad Pitt, Angelina Jolie
- Tom Cruise, Scarlett Johansson, Robert Downey Jr.
- Chris Hemsworth, Emma Watson, Margot Robbie
- And many more...

Each celebrity entry includes:
- Known films/shows
- Brief biography
- Birth year

## Configuration
- Server runs on 0.0.0.0:5000 (configured in .streamlit/config.toml)
- No database required (stateless application)
- Optional: Add OPENAI_API_KEY secret for enhanced AI vision analysis

## Notes
- Results are for entertainment purposes only
- Better results with clear, well-lit face photos
- Front-facing photos work best
