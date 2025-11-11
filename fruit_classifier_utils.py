"""
Shared classes for Fruit/Vegetable Classification
Used by both trainer and tester
"""

import numpy as np
import cv2
from scipy import ndimage
import random


class FeatureExtractor:
    """Extract meaningful features from images"""

    @staticmethod
    def extract_color_histogram(image):
        """Color histogram - critical for fruits/vegetables"""
        # image is already 100x100x3 in range [0,1]
        
        # Convert to uint8 for OpenCV
        img_uint8 = (image * 255).astype(np.uint8)
        
        # More bins for better color discrimination
        hist_r = cv2.calcHist([img_uint8], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([img_uint8], [1], None, [16], [0, 256])
        hist_b = cv2.calcHist([img_uint8], [2], None, [16], [0, 256])
        
        # Normalize
        hist_r = hist_r.flatten() / (hist_r.sum() + 1e-7)
        hist_g = hist_g.flatten() / (hist_g.sum() + 1e-7)
        hist_b = hist_b.flatten() / (hist_b.sum() + 1e-7)
        
        return np.concatenate([hist_r, hist_g, hist_b])

    @staticmethod
    def extract_hsv_features(image):
        """HSV color space features"""
        img_uint8 = (image * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        
        features = []
        for i in range(3):  # H, S, V
            channel = hsv[:, :, i].astype(np.float32)
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75)
            ])
        
        return np.array(features)

    @staticmethod
    def extract_color_moments(image):
        """Statistical color moments"""
        features = []
        for i in range(3):  # RGB
            channel = image[:, :, i]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75),
            ])
        
        return np.array(features)

    @staticmethod
    def extract_texture_features(image):
        """Texture features using edge detection"""
        img_uint8 = (image * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        features = []
        
        # Sobel edges
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        features.extend([
            np.mean(np.abs(sobelx)),
            np.std(np.abs(sobelx)),
            np.mean(np.abs(sobely)),
            np.std(np.abs(sobely))
        ])
        
        # Laplacian (texture roughness)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        features.extend([
            np.mean(np.abs(laplacian)),
            np.std(np.abs(laplacian))
        ])
        
        return np.array(features)

    @staticmethod
    def extract_all_features(image):
        """Extract all features - image should be 100x100x3 in range [0,1]"""
        color_hist = FeatureExtractor.extract_color_histogram(image)
        hsv_features = FeatureExtractor.extract_hsv_features(image)
        color_moments = FeatureExtractor.extract_color_moments(image)
        texture = FeatureExtractor.extract_texture_features(image)
        
        return np.concatenate([color_hist, hsv_features, color_moments, texture])


class DataAugmentation:
    """Data augmentation for images in 100x100x3 format"""

    @staticmethod
    def rotate(image, angle_range=(-20, 20)):
        """Rotate image"""
        angle = random.uniform(angle_range[0], angle_range[1])
        rotated = ndimage.rotate(image, angle, reshape=False, order=1)
        return np.clip(rotated, 0, 1)

    @staticmethod
    def flip_horizontal(image):
        """Horizontal flip"""
        return cv2.flip(image, 1)

    @staticmethod
    def adjust_brightness(image, factor_range=(0.6, 1.4)):
        """Adjust brightness"""
        factor = random.uniform(factor_range[0], factor_range[1])
        return np.clip(image * factor, 0, 1)

    @staticmethod
    def adjust_saturation(image, factor_range=(0.6, 1.4)):
        """Adjust saturation"""
        img_uint8 = (image * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        factor = random.uniform(factor_range[0], factor_range[1])
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb.astype(np.float32) / 255.0

    @staticmethod
    def add_noise(image, noise_level=0.02):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level, image.shape)
        return np.clip(image + noise, 0, 1)

    @staticmethod
    def random_crop_and_resize(image, crop_factor_range=(0.8, 1.0)):
        """Random crop and resize back"""
        h, w = image.shape[:2]
        factor = random.uniform(crop_factor_range[0], crop_factor_range[1])
        
        new_h, new_w = int(h * factor), int(w * factor)
        start_h = random.randint(0, h - new_h)
        start_w = random.randint(0, w - new_w)
        
        cropped = image[start_h:start_h+new_h, start_w:start_w+new_w]
        resized = cv2.resize(cropped, (w, h))
        
        return resized
