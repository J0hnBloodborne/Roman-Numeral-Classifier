from tensorflow import keras
import numpy as np
import base64
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RomanModel:
    def __init__(self):
        self.model = self.load_model('Except2.keras')
        self.subnet_model = self.load_model('2-5-9th.keras')
        self.roman_classes = ['I', 'II', 'III', 'IV', 'IX', 'V', 'VI', 'VII', 'VIII', 'X']

    def load_model(self, model_path):
        try:
            model = keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None

    def preprocess_image(self, image_data):
        try:
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('L')
            image = image.resize((28, 28))
            image_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0
            logger.debug(f"Processed image shape: {image_array.shape}")
            return image_array
        except Exception as e:
            logger.error(f"Error in preprocessing image: {e}")
            raise

    def predict(self, processed_image):
        if self.model is None:
            logger.error("Main model not loaded")
            return None, None

        prediction = self.model.predict(processed_image)
        predicted_index = np.argmax(prediction[0])
        predicted_roman = self.roman_classes[predicted_index]
        confidence = float(prediction[0][predicted_index])
        return predicted_roman, confidence

    def predict_with_subnet(self, processed_image):
        if self.subnet_model is None:
            logger.error("Subnet model not loaded")
            return None

        subnet_prediction = self.subnet_model.predict(processed_image)
        return float(subnet_prediction[0][0]) > 0.5  # Returns True if predicted as "II"