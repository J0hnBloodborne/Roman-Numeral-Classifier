from PIL import Image
import numpy as np
import base64
import io

def crop_whitespace(image):
    img_array = np.array(image)
    
    if len(img_array.shape) > 2:
        gray_img = image.convert('L')
        img_array = np.array(gray_img)
    
    rows = np.where(np.min(img_array, axis=1) < 240)[0]
    cols = np.where(np.min(img_array, axis=0) < 240)[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return image
    
    top, bottom = rows[0], rows[-1] + 1
    left, right = cols[0], cols[-1] + 1
    
    padding = max(3, int(min(img_array.shape) * 0.05))
    
    top = max(0, top - padding)
    bottom = min(img_array.shape[0], bottom + padding)
    left = max(0, left - padding)
    right = min(img_array.shape[1], right + padding)
    
    cropped_img = image.crop((left, top, right, bottom))
    
    return cropped_img

def preprocess_image(image_data):
    if isinstance(image_data, str) and image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    image = image.convert('L')
    image = crop_whitespace(image)
    image = image.resize((28, 28))
    
    image_array = np.array(image)
    image_array = image_array.reshape(1, 28, 28, 1)
    image_array = image_array.astype('float32') / 255.0
    
    return image_array