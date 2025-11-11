import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import cv2
from pathlib import Path
import pickle
import gc

# Import shared classes
from fruit_classifier_utils import FeatureExtractor, DataAugmentation
import random


class ImprovedFruitClassifier:
    """Improved binary classifier with proper feature extraction"""

    def __init__(self, img_size=(100, 100), augmentation_factor=3):
        self.img_size = img_size
        self.augmentation_factor = augmentation_factor

        # More powerful model
        self.model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
            verbose=1
        )

        self.scaler = StandardScaler()
        self.augmenter = DataAugmentation()
        self.feature_extractor = FeatureExtractor()

    def _load_image(self, img_path):
        """Load and preprocess image - returns 100x100x3 array in [0,1]"""
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        return img

    def _augment_image(self, image):
        """Generate augmented versions - works with 100x100x3 images"""
        augmented_images = [image]

        transformations = [
            lambda img: self.augmenter.rotate(img),
            lambda img: self.augmenter.flip_horizontal(img),
            lambda img: self.augmenter.adjust_brightness(img),
            lambda img: self.augmenter.adjust_saturation(img),
            lambda img: self.augmenter.add_noise(img),
            lambda img: self.augmenter.random_crop_and_resize(img),
        ]

        for _ in range(self.augmentation_factor):
            num_transforms = random.randint(1, 3)
            selected_transforms = random.sample(transformations, num_transforms)

            aug_img = image.copy()
            for transform in selected_transforms:
                aug_img = transform(aug_img)

            augmented_images.append(aug_img)

        return augmented_images

    def load_dataset(self, dataset_path):
        """Load all images from dataset"""
        dataset_path = Path(dataset_path)

        all_images = []
        all_labels = []

        categories = ['Frutas', 'Verduras']

        for category in categories:
            category_folder = dataset_path / category
            if not category_folder.exists():
                print(f"Warning: Folder not found: {category_folder}")
                continue

            img_files = (list(category_folder.glob('*.jpg')) +
                         list(category_folder.glob('*.png')) +
                         list(category_folder.glob('*.jpeg')))

            print(f"  Loading {len(img_files)} images from {category}...")

            for img_path in img_files:
                img_data = self._load_image(img_path)
                if img_data is not None:
                    all_images.append(img_data)
                    all_labels.append(1 if category == 'Verduras' else 0)

        return np.array(all_images, dtype=np.float32), np.array(all_labels)

    def train(self, dataset_path, test_size=0.2, val_size=0.1):
        """Train the classifier with validation set"""
        print("\n" + "=" * 70)
        print("IMPROVED FRUIT/VEGETABLE CLASSIFIER")
        print("=" * 70)
        print(f"Augmentation factor: {self.augmentation_factor}x")
        print(f"Model: Gradient Boosting Classifier")

        # Load dataset
        print("\n" + "=" * 70)
        print("Loading dataset...")
        print("=" * 70)

        X_all, y_all = self.load_dataset(dataset_path)

        if len(X_all) == 0:
            raise ValueError("No images found. Check dataset path.")

        print(f"\nTotal: {len(X_all)} images")
        print(f"  Fruits: {np.sum(y_all == 0)}")
        print(f"  Vegetables: {np.sum(y_all == 1)}")

        # Split: train/val/test
        print("\n" + "=" * 70)
        print("Splitting dataset...")
        print("=" * 70)

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_all, y_all,
            test_size=test_size,
            random_state=42,
            stratify=y_all
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=42,
            stratify=y_temp
        )

        print(f"  Training: {len(X_train)} images")
        print(f"  Validation: {len(X_val)} images")
        print(f"  Test: {len(X_test)} images")

        del X_all, y_all, X_temp, y_temp
        gc.collect()

        # Augment training set
        print("\n" + "=" * 70)
        print("Augmenting training set...")
        print("=" * 70)

        X_train_aug = []
        y_train_aug = []

        for i, (img, label) in enumerate(zip(X_train, y_train)):
            augmented = self._augment_image(img)
            X_train_aug.extend(augmented)
            y_train_aug.extend([label] * len(augmented))

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(X_train)}...")

        X_train_aug = np.array(X_train_aug, dtype=np.float32)
        y_train_aug = np.array(y_train_aug)

        print(f"\nAugmented training set: {len(X_train_aug):,} images")

        del X_train, y_train
        gc.collect()

        # Extract features
        print("\n" + "=" * 70)
        print("Extracting features...")
        print("=" * 70)

        print("  Training features...")
        X_train_features = np.array([
            self.feature_extractor.extract_all_features(img)
            for img in X_train_aug
        ])

        print("  Validation features...")
        X_val_features = np.array([
            self.feature_extractor.extract_all_features(img)
            for img in X_val
        ])

        print("  Test features...")
        X_test_features = np.array([
            self.feature_extractor.extract_all_features(img)
            for img in X_test
        ])

        print(f"\nFeatures extracted: {X_train_features.shape[1]} per image")

        del X_train_aug, X_val, X_test
        gc.collect()

        # Normalize
        print("\nNormalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        X_val_scaled = self.scaler.transform(X_val_features)
        X_test_scaled = self.scaler.transform(X_test_features)

        del X_train_features, X_val_features, X_test_features
        gc.collect()

        # Train
        print("\n" + "=" * 70)
        print("Training model...")
        print("=" * 70)

        self.model.fit(X_train_scaled, y_train_aug)
        print("\nTraining complete!")

        del X_train_scaled, y_train_aug
        gc.collect()

        # Evaluate on validation set
        print("\n" + "=" * 70)
        print("VALIDATION SET EVALUATION")
        print("=" * 70)

        y_val_pred = self.model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        print(f"\nValidation Accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")

        # Evaluate on test set
        print("\n" + "=" * 70)
        print("TEST SET EVALUATION")
        print("=" * 70)

        y_test_pred = self.model.predict(X_test_scaled)
        y_test_proba = self.model.predict_proba(X_test_scaled)

        test_accuracy = accuracy_score(y_test, y_test_pred)

        print(f"\n{'=' * 70}")
        print(f"TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
        print(f"{'=' * 70}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)

        print("\nConfusion Matrix:")
        print("\n                    Predicted")
        print("                Fruit    Vegetable")
        print(f"    Real Fruit    {cm[0, 0]:5d}    {cm[0, 1]:5d}")
        print(f"         Vegetable {cm[1, 0]:5d}    {cm[1, 1]:5d}")

        # Metrics
        tn, fp, fn, tp = cm.ravel()

        precision_fruit = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_fruit = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_fruit = 2 * (precision_fruit * recall_fruit) / (precision_fruit + recall_fruit) if (
                                                                                                          precision_fruit + recall_fruit) > 0 else 0

        precision_veg = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_veg = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_veg = 2 * (precision_veg * recall_veg) / (precision_veg + recall_veg) if (
                                                                                                precision_veg + recall_veg) > 0 else 0

        print(f"\nMetrics by class:")
        print(f"\n  Fruits:")
        print(f"     Precision: {precision_fruit:.3f}")
        print(f"     Recall:    {recall_fruit:.3f}")
        print(f"     F1-Score:  {f1_fruit:.3f}")

        print(f"\n  Vegetables:")
        print(f"     Precision: {precision_veg:.3f}")
        print(f"     Recall:    {recall_veg:.3f}")
        print(f"     F1-Score:  {f1_veg:.3f}")

        # Confidence analysis
        print("\n" + "=" * 70)
        print("CONFIDENCE ANALYSIS")
        print("=" * 70)

        confidence = np.max(y_test_proba, axis=1)

        very_high = np.sum(confidence >= 0.9)
        high = np.sum((confidence >= 0.8) & (confidence < 0.9))
        medium = np.sum((confidence >= 0.7) & (confidence < 0.8))
        low = np.sum(confidence < 0.7)

        print(f"\n  Very high (>=90%): {very_high:4d} ({very_high / len(confidence) * 100:5.1f}%)")
        print(f"  High (80-90%):     {high:4d} ({high / len(confidence) * 100:5.1f}%)")
        print(f"  Medium (70-80%):   {medium:4d} ({medium / len(confidence) * 100:5.1f}%)")
        print(f"  Low (<70%):        {low:4d} ({low / len(confidence) * 100:5.1f}%)")

        print(f"\n  Average confidence: {np.mean(confidence):.3f}")
        print(f"  Median confidence:  {np.median(confidence):.3f}")

        # Full report
        print("\n" + "=" * 70)
        print("Full Classification Report:")
        print("=" * 70)
        print(classification_report(
            y_test, y_test_pred,
            target_names=['Fruits', 'Vegetables'],
            zero_division=0
        ))

        return val_accuracy, test_accuracy, np.mean(confidence)

    def save_model(self, filepath):
        """Save model - don't save feature_extractor, save only class reference"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'img_size': self.img_size,
            'type': 'improved_binary'
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\nModel saved: {filepath}")


if __name__ == "__main__":

    # RECOMMENDED CONFIGURATION
    classifier = ImprovedFruitClassifier(
        img_size=(100, 100),
        augmentation_factor=3  # Creates 3 augmented versions per image
    )

    dataset_path = "VeggiesFruits"

    print(f"\nDataset: {dataset_path}")

    try:
        val_acc, test_acc, avg_conf = classifier.train(dataset_path)

        classifier.save_model("improved_classifier.pkl")

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
        print(f"Average Confidence: {avg_conf * 100:.1f}%")

        print("\nInterpretation:")
        if test_acc >= 0.85:
            print("  EXCELLENT - High accuracy achieved")
        elif test_acc >= 0.75:
            print("  GOOD - Acceptable accuracy")
        elif test_acc >= 0.65:
            print("  FAIR - Room for improvement")
        else:
            print("  NEEDS IMPROVEMENT - Consider:")
            print("    - Checking data quality")
            print("    - Increasing augmentation_factor")
            print("    - Collecting more diverse images")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()