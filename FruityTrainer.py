import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import cv2
from pathlib import Path
import pickle
import gc


class FruitVegetableClassifier:
    def __init__(self, img_size=(100, 100)):
        self.img_size = img_size
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.label_encoder = LabelEncoder()
        self.category_map = self._create_category_map()

    def _create_category_map(self):
        """Map fruit/vegetable names to categories - comprehensive list"""
        vegetables = [
            # Root vegetables
            'beetroot', 'carrot', 'radish', 'turnip', 'ginger', 'garlic',
            'potato', 'sweet potato', 'sweetpotato',

            # Cruciferous vegetables
            'cabbage', 'cauliflower', 'broccoli', 'kohlrabi', 'kohirabi',

            # Nightshades (botanically fruits, culinarily vegetables)
            'tomato', 'eggplant', 'brinjal', 'pepper', 'capsicum',
            'bell pepper', 'chilli pepper', 'chillipepper', 'paprika',
            'jalapeño', 'jalapeno',

            # Cucurbits (botanically fruits, culinarily vegetables)
            'cucumber', 'zucchini', 'zuchini', 'gourd', 'pumpkin', 'pumkin',

            # Leafy greens
            'lettuce', 'spinach',

            # Legumes
            'bean', 'beans', 'pea', 'peas', 'soybean', 'soy beans', 'soybeans',

            # Other vegetables
            'onion', 'corn', 'sweetcorn'
        ]
        # Convert all to lowercase for case-insensitive matching
        return {name.lower(): 'vegetable' for name in vegetables}

    def _load_image(self, img_path):
        """Load and preprocess image"""
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.flatten()

    def load_dataset_batch(self, dataset_path, max_images_per_class=None, sample_ratio=None):
        """Load dataset with memory optimization"""
        images = []
        labels = []
        dataset_path = Path(dataset_path)

        total_classes = sum(1 for f in dataset_path.iterdir() if f.is_dir())
        current_class = 0

        for class_folder in dataset_path.iterdir():
            if not class_folder.is_dir():
                continue

            current_class += 1
            class_name = class_folder.name
            print(f"[{current_class}/{total_classes}] Loading {class_name}...")

            # Get all image files
            img_files = list(class_folder.glob('*.jpg'))

            # Apply sampling if specified
            if sample_ratio and sample_ratio < 1.0:
                sample_size = int(len(img_files) * sample_ratio)
                img_files = np.random.choice(img_files, size=sample_size, replace=False)
            elif max_images_per_class:
                img_files = img_files[:max_images_per_class]

            # Process images
            for img_file in img_files:
                img_data = self._load_image(img_file)
                if img_data is not None:
                    images.append(img_data)
                    labels.append(class_name)

            # Periodically free memory
            if current_class % 10 == 0:
                gc.collect()

        return np.array(images, dtype=np.float32), np.array(labels)

    def train(self, dataset_path, test_size=0.2, max_images_per_class=None, sample_ratio=None):
        """Train the classifier with memory optimization"""
        print("Loading dataset with memory optimization...")
        print(f"Image size: {self.img_size}")

        if max_images_per_class:
            print(f"Limiting to {max_images_per_class} images per class")
        if sample_ratio:
            print(f"Using {sample_ratio * 100}% of available images")

        X, y = self.load_dataset_batch(dataset_path, max_images_per_class, sample_ratio)

        print(f"\nDataset loaded: {len(X)} images, {len(np.unique(y))} classes")
        print(f"Memory usage: ~{X.nbytes / (1024 ** 2):.2f} MB")

        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Free memory
        del X, y, y_encoded
        gc.collect()

        print("\nTraining model...")
        self.model.fit(X_train, y_train)

        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0
        ))

        return accuracy

    def save_model(self, filepath):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'img_size': self.img_size,
            'category_map': self.category_map
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")


if __name__ == "__main__":
    # Initialize classifier with smaller image size for memory efficiency
    classifier = FruitVegetableClassifier(img_size=(50, 50))  # Reduced from 100x100

    # Train on dataset with memory constraints
    dataset_path = "fruitNveggies/train"

    print("Starting training process with memory optimization...")

    # Option 1: Limit images per class (e.g., use only 100 images per class)
    classifier.train(dataset_path, max_images_per_class=100)

    # Option 2: Use a percentage of all images (e.g., 30% of dataset)
    # classifier.train(dataset_path, sample_ratio=0.3)

    # Option 3: Use full dataset (if you have enough RAM)
    # classifier.train(dataset_path)

    # Save the trained model
    classifier.save_model("fruit_classifier_model.pkl")

    print("\nTraining complete! Model saved as 'fruit_classifier_model.pkl'")