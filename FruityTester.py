import numpy as np
import cv2
import pickle
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Import shared classes
from fruit_classifier_utils import FeatureExtractor


class ImprovedFruitPredictorUI:
    def __init__(self, model_path):
        """Initialize the improved predictor with UI"""
        self.model = None
        self.scaler = None
        self.feature_extractor = FeatureExtractor()
        self.img_size = None
        self.current_image_path = None

        self.load_model(model_path)
        self.setup_ui()

    def load_model(self, filepath):
        """Load the trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.img_size = model_data['img_size']

            print(f"Model loaded: {filepath}")
            print(f"Model type: {model_data.get('type', 'unknown')}")
        except FileNotFoundError:
            messagebox.showerror("Error",
                                 f"File '{filepath}' not found!\n"
                                 "Run ImprovedFruitTrainer.py first.")
            raise
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {e}")
            raise

    def setup_ui(self):
        """Create the user interface"""
        self.root = tk.Tk()
        self.root.title("Fruit vs Vegetable Classifier - Improved")
        self.root.geometry("900x750")
        self.root.configure(bg='#ecf0f1')

        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title = tk.Label(
            title_frame,
            text="Fruit or Vegetable\nImproved Binary Classifier",
            font=("Arial", 18, "bold"),
            bg='#2c3e50',
            fg='white',
            justify=tk.CENTER
        )
        title.pack(expand=True)

        # Main frame
        main_frame = tk.Frame(self.root, bg='#ecf0f1')
        main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Image display
        img_frame = tk.LabelFrame(
            main_frame,
            text="Selected Image",
            font=("Arial", 12, "bold"),
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        img_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        self.image_display = tk.Frame(img_frame, bg='white', relief=tk.SUNKEN, bd=2)
        self.image_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.image_label = tk.Label(
            self.image_display,
            text="\n\n\nSelect an image",
            font=("Arial", 14),
            bg='white',
            fg='#95a5a6'
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Results display
        result_frame = tk.LabelFrame(
            main_frame,
            text="Prediction Result",
            font=("Arial", 12, "bold"),
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        result_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)

        self.result_display = tk.Frame(result_frame, bg='white', relief=tk.SUNKEN, bd=2)
        self.result_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.result_icon = tk.Label(
            self.result_display,
            text="?",
            font=("Arial", 70),
            bg='white',
            fg='#95a5a6'
        )
        self.result_icon.pack(pady=(20, 10))

        self.result_category = tk.Label(
            self.result_display,
            text="No prediction",
            font=("Arial", 20, "bold"),
            bg='white',
            fg='#95a5a6'
        )
        self.result_category.pack()

        self.result_confidence = tk.Label(
            self.result_display,
            text="",
            font=("Arial", 14),
            bg='white',
            fg='#7f8c8d'
        )
        self.result_confidence.pack(pady=10)

        # Confidence bar
        self.confidence_frame = tk.Frame(self.result_display, bg='white')
        self.confidence_frame.pack(pady=15, padx=40, fill=tk.X)

        self.confidence_bar_bg = tk.Canvas(
            self.confidence_frame,
            height=35,
            bg='#ecf0f1',
            highlightthickness=0
        )
        self.confidence_bar_bg.pack(fill=tk.X)

        # Probability details
        self.prob_frame = tk.Frame(self.result_display, bg='white')
        self.prob_frame.pack(pady=10, padx=20, fill=tk.X)

        self.fruit_prob_label = tk.Label(
            self.prob_frame,
            text="Fruit: ---%",
            font=("Arial", 11),
            bg='white',
            fg='#7f8c8d',
            anchor='w'
        )
        self.fruit_prob_label.pack(fill=tk.X)

        self.veg_prob_label = tk.Label(
            self.prob_frame,
            text="Vegetable: ---%",
            font=("Arial", 11),
            bg='white',
            fg='#7f8c8d',
            anchor='w'
        )
        self.veg_prob_label.pack(fill=tk.X)

        # Buttons
        button_frame = tk.Frame(self.root, bg='#ecf0f1')
        button_frame.pack(pady=20)

        self.browse_btn = tk.Button(
            button_frame,
            text="Browse Image",
            command=self.browse_file,
            font=("Arial", 12, "bold"),
            bg='#3498db',
            fg='white',
            padx=30,
            pady=15,
            relief=tk.RAISED,
            cursor="hand2"
        )
        self.browse_btn.grid(row=0, column=0, padx=10)

        self.predict_btn = tk.Button(
            button_frame,
            text="Classify",
            command=self.predict_image,
            font=("Arial", 12, "bold"),
            bg='#2ecc71',
            fg='white',
            padx=30,
            pady=15,
            relief=tk.RAISED,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.predict_btn.grid(row=0, column=1, padx=10)

        self.clear_btn = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_all,
            font=("Arial", 12, "bold"),
            bg='#e74c3c',
            fg='white',
            padx=30,
            pady=15,
            relief=tk.RAISED,
            cursor="hand2"
        )
        self.clear_btn.grid(row=0, column=2, padx=10)

        # Status bar
        self.status = tk.Label(
            self.root,
            text="Ready to classify",
            font=("Arial", 10),
            bg='#34495e',
            fg='white',
            anchor=tk.W,
            padx=15,
            pady=8
        )
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def browse_file(self):
        """Open file dialog to select image"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_btn.config(state=tk.NORMAL)
            self.status.config(text=f" {Path(file_path).name}")
            self.reset_result_display()

    def display_image(self, img_path):
        """Display the selected image"""
        try:
            img = Image.open(img_path)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Could not display image: {e}")

    def reset_result_display(self):
        """Reset the result display"""
        self.result_icon.config(text="?", fg='#95a5a6')
        self.result_category.config(text="No prediction", fg='#95a5a6')
        self.result_confidence.config(text="")
        self.confidence_bar_bg.delete("all")
        self.fruit_prob_label.config(text="Fruit: ---%")
        self.veg_prob_label.config(text="Vegetable: ---%")

    def _load_and_process_image(self, img_path):
        """Load image and extract features"""
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # Extract features
        features = self.feature_extractor.extract_all_features(img)

        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        return features_scaled

    def predict_image(self):
        """Perform prediction"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Select an image first")
            return

        self.status.config(text="Analyzing...")
        self.root.update()

        try:
            # Process image
            features = self._load_and_process_image(self.current_image_path)
            if features is None:
                messagebox.showerror("Error", "Could not process image")
                return

            # Predict
            prediction = self.model.predict(features)
            probabilities = self.model.predict_proba(features)[0]

            category = 'Vegetable' if prediction[0] == 1 else 'Fruit'
            confidence = probabilities[prediction[0]]

            fruit_prob = probabilities[0]
            veg_prob = probabilities[1]

            # Display result
            self.display_result(category, confidence, fruit_prob, veg_prob)
            self.status.config(text=f"Classification complete: {category} ({confidence:.1%})")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.status.config(text="Error in prediction")
            import traceback
            traceback.print_exc()

    def display_result(self, category, confidence, fruit_prob, veg_prob):
        """Display the prediction result visually"""
        # Icon and color based on category
        if category == 'Fruit':
            icon = "[FRUIT]"
            color = "#e74c3c"
            bar_color = "#e74c3c"
        else:
            icon = "[VEGETABLE]"
            color = "#27ae60"
            bar_color = "#27ae60"

        # Update display
        self.result_icon.config(text=icon, fg=color)
        self.result_category.config(text=category.upper(), fg=color)

        # Confidence indicator
        if confidence >= 0.9:
            conf_text = "Confidence: VERY HIGH"
            conf_color = "#27ae60"
        elif confidence >= 0.8:
            conf_text = "Confidence: HIGH"
            conf_color = "#2ecc71"
        elif confidence >= 0.7:
            conf_text = "Confidence: MEDIUM"
            conf_color = "#f39c12"
        elif confidence >= 0.6:
            conf_text = "Confidence: MODERATE"
            conf_color = "#e67e22"
        else:
            conf_text = "Confidence: LOW"
            conf_color = "#e74c3c"

        self.result_confidence.config(
            text=f"{confidence:.1%}\n{conf_text}",
            fg=conf_color
        )

        # Update probability labels
        self.fruit_prob_label.config(
            text=f"Fruit:     {fruit_prob:.1%}",
            fg='#e74c3c' if fruit_prob > veg_prob else '#7f8c8d'
        )
        self.veg_prob_label.config(
            text=f"Vegetable: {veg_prob:.1%}",
            fg='#27ae60' if veg_prob > fruit_prob else '#7f8c8d'
        )

        # Confidence bar
        self.confidence_bar_bg.delete("all")
        bar_width = self.confidence_bar_bg.winfo_width()
        if bar_width < 10:
            bar_width = 350

        fill_width = int(bar_width * confidence)

        # Background
        self.confidence_bar_bg.create_rectangle(
            0, 0, bar_width, 35,
            fill='#ecf0f1',
            outline=''
        )

        # Progress bar
        self.confidence_bar_bg.create_rectangle(
            0, 0, fill_width, 35,
            fill=bar_color,
            outline=''
        )

        # Text on bar
        self.confidence_bar_bg.create_text(
            bar_width // 2, 17,
            text=f"{confidence:.1%}",
            font=("Arial", 12, "bold"),
            fill='white' if confidence > 0.3 else '#7f8c8d'
        )

        # Console output
        print(f"\n{'=' * 50}")
        print(f"Result: {category}")
        print(f"{'=' * 50}")
        print(f"Fruit:     {fruit_prob:.1%}")
        print(f"Vegetable: {veg_prob:.1%}")
        print(f"Confidence: {confidence:.1%}")
        print(f"{'=' * 50}\n")

    def clear_all(self):
        """Clear everything"""
        self.current_image_path = None
        self.image_label.config(image='', text="\n\n\n\nSelect an image")
        self.image_label.image = None
        self.reset_result_display()
        self.predict_btn.config(state=tk.DISABLED)
        self.status.config(text="Ready to classify")

    def run(self):
        """Start the UI"""
        self.root.mainloop()


if __name__ == "__main__":
    try:
        app = ImprovedFruitPredictorUI(model_path="improved_classifier.pkl")
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")