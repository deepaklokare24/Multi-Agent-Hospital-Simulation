from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoProcessor
)
from PIL import Image
import requests
from typing import Dict, Tuple
import torch

class PatientImageAnalysis:
    def __init__(self):
        # Initialize gender classification
        self.gender_model_name = "rizvandwiki/gender-classification"
        self.gender_model = AutoModelForImageClassification.from_pretrained(self.gender_model_name)
        self.gender_processor = AutoProcessor.from_pretrained(self.gender_model_name)

        # Initialize age classification
        self.age_model_name = "nateraw/vit-age-classifier"
        self.age_model = AutoModelForImageClassification.from_pretrained(self.age_model_name)
        self.age_processor = AutoProcessor.from_pretrained(self.age_model_name)

        # Initialize X-ray classification
        self.xray_model_name = "lxyuan/vit-xray-pneumonia-classification"
        self.xray_model = AutoModelForImageClassification.from_pretrained(self.xray_model_name)
        self.xray_processor = AutoProcessor.from_pretrained(self.xray_model_name)

    def analyze_patient_photo(self, image: Image.Image) -> Dict[str, str]:
        """Analyze patient photo for gender and age."""
        # Gender analysis
        gender_inputs = self.gender_processor(images=image, return_tensors="pt")
        gender_outputs = self.gender_model(**gender_inputs)
        gender_prediction = gender_outputs.logits.argmax(-1).item()
        gender_label = self.gender_model.config.id2label[gender_prediction]

        # Age analysis
        age_inputs = self.age_processor(images=image, return_tensors="pt")
        age_outputs = self.age_model(**age_inputs)
        age_prediction = age_outputs.logits.argmax(-1).item()
        age_label = self.age_model.config.id2label[age_prediction]

        return {
            "gender": gender_label,
            "age_range": age_label
        }

    def analyze_xray(self, image: Image.Image) -> Dict[str, float]:
        """Analyze chest X-ray for pneumonia."""
        inputs = self.xray_processor(images=image, return_tensors="pt")
        outputs = self.xray_model(**inputs)
        
        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert to dictionary of label: probability
        result = {
            self.xray_model.config.id2label[i]: prob.item()
            for i, prob in enumerate(probabilities[0])
        }
        
        return result

    @staticmethod
    def load_image_from_url(url: str) -> Image.Image:
        """Load an image from a URL."""
        return Image.open(requests.get(url, stream=True).raw)

    @staticmethod
    def load_image_from_path(path: str) -> Image.Image:
        """Load an image from a local file path."""
        return Image.open(path) 