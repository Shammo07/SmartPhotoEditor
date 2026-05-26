# Smart Photo Editor

A web-based photo editing application built with **Django** that allows users to upload images and apply a wide range of filters, adjustments, and AI-powered effects directly in the browser.

---

## Features

### Basic Adjustments (with real-time preview slider)
- **Blur** – Gaussian blur with adjustable radius
- **Sharpen** – Sharpness enhancement with adjustable intensity
- **Brightness** – Brighten or darken the image
- **Contrast** – Increase or decrease contrast
- **Rotate** – Rotate the image by any angle
- **Crop** – Interactively crop a region of the image

### Artistic Filters
- Black & White
- Invert Colors
- Sepia
- Vintage
- Noir
- Polaroid
- Neon
- Watercolor
- Lomography
- Black & Gold
- Cool Tone
- Warm Tone
- Vibrant
- Red Highlight

### AI-Powered Features
- **Face Detection** – Detects and highlights faces using OpenCV's Haar Cascade classifier
- **Portrait Mode** – Blurs the background while keeping the subject sharp, powered by **MediaPipe** selfie segmentation
- **Object Detection** – Detects and labels objects in an image using **YOLOv8** (Ultralytics)

---

## Tech Stack

| Layer            | Technology                        |
|------------------|-----------------------------------|
| Backend          | Django 5.1                        |
| Image Processing | Pillow, OpenCV (`cv2`), NumPy     |
| AI / ML          | MediaPipe, Ultralytics YOLOv8     |
| Frontend         | Bootstrap 5, HTML/CSS/JavaScript  |
| Database         | SQLite (default Django DB)        |

---

## Project Structure

```
SmartPhotoEditor/
├── manage.py
├── photoeditor/          # Django project configuration
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
└── editor/               # Main application
    ├── models.py         # Photo model (original, edited, preview fields)
    ├── views.py          # All image processing logic
    ├── forms.py          # Photo upload form
    ├── urls.py           # App URL routing
    ├── templates/
    │   └── editor/
    │       └── index.html  # Single-page UI
    └── migrations/
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shammo07/SmartPhotoEditor.git
   cd SmartPhotoEditor
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply migrations**
   ```bash
   python manage.py migrate
   ```

5. **Run the development server**
   ```bash
   python manage.py runserver
   ```

6. **Open in browser**
   ```
   http://127.0.0.1:8000/
   ```

> **Note:** The YOLOv8 model (`yolov8n.pt`) will be downloaded automatically on first use.

---

## Usage

1. Click **Upload Image** and select a photo from your device.
2. Use the **Apply Filter** dropdown to choose an artistic filter or AI effect.
3. Use the **Adjustments** section (blur, sharpen, brightness, contrast, rotate, crop) with the real-time preview slider to fine-tune your edits.
4. Click **Reset to Original** at any time to undo all changes.

---

## Notes

- Uploaded and processed images are stored in the `media/` directory (`uploads/`, `edited/`, `previews/` sub-folders).
- This project is configured for **development only** (`DEBUG = True`). Do not deploy as-is to production.
- The secret key in `settings.py` should be changed and kept secret before any production deployment.
