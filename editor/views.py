from django.shortcuts import render, redirect
from .models import Photo
from .forms import PhotoForm
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from django.core.files.base import ContentFile
from io import BytesIO
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import time
import mediapipe as mp
from ultralytics import YOLO
import random

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Initialize YOLO model
object_detection_model = YOLO('yolov8n.pt')  # This will download the model automatically

def index(request):
    if request.method == "POST":
        # Check if this is a filter application with slider values
        if 'apply_filter' in request.POST:
            photo_id = request.POST.get('photo_id')
            action = request.POST.get('action')
            intensity = float(request.POST.get('intensity', 1.0))
            
            photo = Photo.objects.get(id=photo_id)
            img_path = photo.edited.path if photo.edited else photo.original.path
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)

            if action == "blur":
                # Apply GaussianBlur with intensity-based radius
                radius = max(1, int(intensity))
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            elif action == "sharpen":
                # Apply sharpen with adjustable intensity
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(intensity)
            elif action == "brightness":
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(intensity)
            elif action == "contrast":
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(intensity)
            elif action == "rotate":
                angle = intensity  
                img = img.rotate(-angle, expand=True)
            elif action == "crop":
                # Get crop coordinates from the request
                x = int(request.POST.get('x', 0))
                y = int(request.POST.get('y', 0))
                width = int(request.POST.get('width', img.width))
                height = int(request.POST.get('height', img.height))
                
                # Ensure coordinates are within image bounds
                x = max(0, min(x, img.width - 1))
                y = max(0, min(y, img.height - 1))
                width = min(width, img.width - x)
                height = min(height, img.height - y)
                
                # Perform the crop
                img = img.crop((x, y, x + width, y + height))
            
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            photo.edited.save(f"processed_{photo_id}.png", ContentFile(buffer.getvalue()))
            return redirect('index')
        
        # Original file upload handling
        form = PhotoForm(request.POST, request.FILES)
        if form.is_valid():
            photo = form.save()
            request.session['photo_id'] = photo.id
            return redirect('index')

    else:
        form = PhotoForm()
        photo = None
        if 'photo_id' in request.session:
            try:
                photo = Photo.objects.get(id=request.session['photo_id'])
            except Photo.DoesNotExist:
                del request.session['photo_id']

    return render(request, 'editor/index.html', {'form': form, 'photo': photo})

def process_image(request, photo_id, action):
    photo = Photo.objects.get(id=photo_id)
    img_path = photo.edited.path if photo.edited else photo.original.path
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)

    if action == "black_white":
        img = img.convert("L")
    elif action == "invert":
        # Invert colors
        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            rgb = Image.merge('RGB', (r, g, b))
            inverted = ImageOps.invert(rgb)
            r2, g2, b2 = inverted.split()
            img = Image.merge('RGBA', (r2, g2, b2, a))
        else:
            img = ImageOps.invert(img)
    elif action == "cool_tone":
        # Cool tone (blue tint)
        img = img.convert("RGB")
        r, g, b = img.split()
        b = b.point(lambda x: x * 1.3)
        img = Image.merge('RGB', (r, g, b))
    elif action == "warm_tone":
        # Warm tone (red/yellow tint)
        img = img.convert("RGB")
        r, g, b = img.split()
        r = r.point(lambda x: x * 1.2)
        g = g.point(lambda x: x * 1.1)
        img = Image.merge('RGB', (r, g, b))
    elif action == "vibrant":
        # Vibrant colors (increase saturation)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(2.0)
    elif action == "red_highlight":
        # Grayscale with red highlight
        img = img.convert("RGB")
        width, height = img.size
        pixels = img.load()
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                # Keep red pixels, convert others to grayscale
                if not (r > g * 1.3 and r > b * 1.3):
                    gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                    pixels[x, y] = (gray, gray, gray)
    elif action == "sepia":
        # Convert to RGB if not already
        img = img.convert("RGB")
        
        # Sepia transformation matrix
        sepia_matrix = (
            0.393, 0.769, 0.189, 0,
            0.349, 0.686, 0.168, 0,
            0.272, 0.534, 0.131, 0
        )
        
        # Apply the matrix transformation
        img = img.convert("RGB", sepia_matrix)
        
        # Slightly increase contrast for better effect
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
    elif action == "vintage":
        img = img.convert("RGB")
        width, height = img.size
        pixels = img.load()
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                
                # Add vintage effect
                r = min(255, int(r * 0.9 + 30))
                g = min(255, int(g * 0.9 + 10))
                b = min(255, int(b * 0.8))
                
                # Add slight vignette
                dx = (x - width/2) / (width/2)
                dy = (y - height/2) / (height/2)
                vignette = 1.0 - (dx*dx + dy*dy) * 0.3
                r = int(r * vignette)
                g = int(g * vignette)
                b = int(b * vignette)
                
                pixels[x, y] = (r, g, b)
    elif action == "noir":
        img = img.convert("L")
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.5)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.8)
    elif action == "polaroid":
        img = img.convert("RGB")
        # Add white border
        img = ImageOps.expand(img, border=30, fill='white')
        # Add slight warm tint
        r, g, b = img.split()
        r = r.point(lambda x: min(255, x + 15))
        g = g.point(lambda x: min(255, x + 5))
        img = Image.merge('RGB', (r, g, b))
    elif action == "neon":
        img = img.convert("RGB")
        width, height = img.size
        pixels = img.load()
        
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                
                # Boost colors and add glow
                r = min(255, int(r ** 1.5 * 0.8))
                g = min(255, int(g ** 1.5 * 0.9))
                b = min(255, int(b ** 1.5 * 1.2))
                
                # Increase contrast between colors
                if max(r, g, b) == r:
                    r = min(255, r + 50)
                elif max(r, g, b) == g:
                    g = min(255, g + 30)
                else:
                    b = min(255, b + 70)
                    
                pixels[x, y] = (r, g, b)
    elif action == "watercolor":
        img = img.convert("RGB")
        # Apply multiple effects
        img = img.filter(ImageFilter.SMOOTH_MORE)
        img = img.filter(ImageFilter.SMOOTH_MORE)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
    elif action == "lomography":
        img = img.convert("RGB")
        width, height = img.size
        pixels = img.load()
        
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                
                # Increase contrast and saturation
                r = min(255, int(r * 1.3 - 20))
                g = min(255, int(g * 1.1))
                b = min(255, int(b * 1.1 - 30))
                
                # Add vignette
                dx = (x - width/2) / (width/2)
                dy = (y - height/2) / (height/2)
                vignette = 0.7 + 0.3 * (1.0 - (dx*dx + dy*dy))
                r = int(r * vignette)
                g = int(g * vignette)
                b = int(b * vignette)
                
                pixels[x, y] = (r, g, b)
    elif action == "black_gold":
        img = img.convert("RGB")
        width, height = img.size
        pixels = img.load()
        
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                brightness = (r + g + b) / 3
                
                if brightness > 160:  # Light areas become gold
                    r = min(255, int(r * 1.8))
                    g = min(255, int(g * 1.5))
                    b = min(255, int(b * 0.3))
                else:  # Dark areas become black
                    r = g = b = max(0, int(brightness * 0.3))
                
                pixels[x, y] = (r, g, b)
    elif action == "face_detect":
        img_cv = cv2.imread(img_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (255, 0, 0), 2)
        img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    elif action == "portrait":
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Process the image with Mediapipe segmentation
        results = segment.process(img_cv)
        
        # Extract the mask (foreground probability)
        mask = results.segmentation_mask
        mask = (mask > 0.5).astype(np.uint8)
        
        # Blur the background (using a fixed blur intensity of 55)
        blur_intensity = 55
        blurred_background = cv2.GaussianBlur(img_cv, (blur_intensity, blur_intensity), 0)
        
        # Combine foreground and blurred background
        output_image = img_cv * mask[:, :, None] + blurred_background * (1 - mask[:, :, None])
        output_image = output_image.astype(np.uint8)
        
        # Convert back to PIL Image
        img = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    elif action == "detect_objects":
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Run object detection
        results = object_detection_model(img_cv)
        
        # Generate random colors for each class
        class_colors = {}
        for class_id in results[0].boxes.cls.unique():
            class_colors[int(class_id)] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        
        # Draw bounding boxes and labels
        for result in results:
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class ID and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = object_detection_model.names[class_id]
                
                # Get color for this class
                color = class_colors[class_id]
                
                # Draw rectangle
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                
                # Create label text
                label = f"{class_name}: {confidence:.2f}"
                
                # Calculate text size
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw background rectangle for text
                cv2.rectangle(img_cv, (x1, y1 - text_height - 10),
                            (x1 + text_width, y1), color, -1)
                
                # Put text
                cv2.putText(img_cv, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert back to PIL Image
        img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    elif action == "default":
        # Reset to original
        if photo.edited:
            photo.edited.delete()
        return redirect('index')

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    photo.edited.save(f"processed_{photo_id}.png", ContentFile(buffer.getvalue()))
    return redirect('index')

@csrf_exempt
def preview_image(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            photo_id = data.get('photo_id')
            action = data.get('action')
            value = float(data.get('value', 1.0))
            
            photo = Photo.objects.get(id=photo_id)
            img_path = photo.edited.path if photo.edited else photo.original.path
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)

            # Apply the same processing as your final filters
            if action == "blur":
                radius = max(1, int(value))
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            elif action == "sharpen":
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(value)
            elif action == "brightness":
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(value)
            elif action == "contrast":
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(value)
            elif action == "rotate":
                angle = value
                img = img.rotate(-angle, expand=True)


            # Save to buffer
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            
            # Create a temporary preview file
            preview_filename = f"preview_{photo_id}_{int(time.time())}.png"
            photo.preview.save(preview_filename, ContentFile(buffer.read()))
            
            return JsonResponse({
                'preview_url': photo.preview.url,
                'success': True
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e), 'success': False}, status=400)
    
    return JsonResponse({'error': 'Invalid request', 'success': False}, status=400)
    