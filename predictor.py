import numpy as np
from PIL import Image
import base64
import tensorflow as tf
import io

class Predictor:
    def __init__(self, main_model_path="Except2.keras", subnet_model_path="2-5-9th.keras"):
        print(f"[Predictor] Initializing. Main model: {main_model_path}, Subnet model: {subnet_model_path}")
        self.roman_classes = ['I', 'II', 'III', 'IV', 'IX', 'V', 'VI', 'VII', 'VIII', 'X']
        self.model = self.load_model(main_model_path) if main_model_path else None
        self.subnet_model = self.load_model(subnet_model_path) if subnet_model_path else None

    def load_model(self, model_path):
        print(f"[Predictor] Loading model from: {model_path}")
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"[Predictor] Model loaded successfully: {model_path}")
            return model
        except Exception as e:
            print(f"[Predictor] Error loading model: {e}")
            return None

    def crop_whitespace(self, image):
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

    def preprocess_image(self, image_data):
        try:
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('L')
            image = self.crop_whitespace(image)
            image = image.resize((28, 28))
            image_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0
            return image_array
        except Exception as e:
            print(f"[Predictor] Error in preprocessing image: {e}")
            return None

    def predict(self, image_data):
        processed_image = self.preprocess_image(image_data)

        if processed_image is None or self.model is None:
            print("[Predictor] Prediction failed: processed_image is None or model is None")
            return None
        prediction = self.model.predict(processed_image)
        print(f"[Predictor] Raw model prediction: {prediction}")
        predicted_index = np.argmax(prediction[0])
        predicted_roman = self.roman_classes[predicted_index]
        confidence = float(prediction[0][predicted_index])
        print(f"[Predictor] Predicted: {predicted_roman}, Confidence: {confidence}")
        if predicted_roman == "V" and self.subnet_model is not None:
            # Enhanced subnet logic for distinguishing II vs V
            img_data = processed_image
            subnet_pred = self.subnet_model.predict(img_data)
            subnet_conf = float(subnet_pred[0][0])
            print(f"[Predictor] Subnet prediction: {subnet_pred}, Conf: {subnet_conf}")
            is_two = subnet_conf > 0.5  # Adjust threshold if needed
            if not is_two:
                flip_img = np.fliplr(img_data[0])  # remove batch dim, flip, then add batch back
                flip_img = flip_img.reshape(1, 28, 28, 1)
                pred = self.subnet_model.predict(flip_img)[0][0]
                print(f"[Predictor] Subnet prediction (flipped): {pred}")
                prediction = "V" if pred > 0.5 else "II"
                if prediction == "V":
                    from scipy.ndimage import rotate
                    img_rotated = rotate(img_data[0], angle=45, reshape=False)
                    img_rotated = img_rotated.reshape(1, 28, 28, 1)
                    pred = self.subnet_model.predict(img_rotated)[0][0]
                    print(f"[Predictor] Subnet prediction (rotated): {pred}")
                    prediction = "V" if pred > 0.5 else "II"
                predicted_roman = prediction
        return predicted_roman