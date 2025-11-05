import numpy as np
import cv2
import pickle
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class FruitPredictorUI:
    def __init__(self, model_path):
        """Initialize the predictor with UI"""
        self.model = None
        self.label_encoder = None
        self.img_size = None
        self.category_map = None
        self.current_image_path = None

        # Load model
        self.load_model(model_path)

        # Create UI
        self.setup_ui()

    def load_model(self, filepath):
        """Load trained model from pickle file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.img_size = model_data['img_size']
            self.category_map = model_data['category_map']

            print(f"Model loaded successfully from {filepath}")
            print(f"Model can classify {len(self.label_encoder.classes_)} different types")
        except FileNotFoundError:
            messagebox.showerror("Error", f"Model file '{filepath}' not found!\nPlease run train_model.py first.")
            raise
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {e}")
            raise

    def setup_ui(self):
        """Create the user interface"""
        self.root = tk.Tk()
        self.root.title("Fruit & Vegetable Classifier")
        self.root.geometry("900x750")
        self.root.configure(bg='#f0f0f0')

        # Title
        title_label = tk.Label(
            self.root,
            text="Fruit & Vegetable Classifier",
            font=("Arial", 22, "bold"),
            bg='#f0f0f0',
            fg='black'
        )
        title_label.pack(pady=15)

        # Main content frame
        content_frame = tk.Frame(self.root, bg='#f0f0f0')
        content_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        # Left side - Image display
        left_frame = tk.Frame(content_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        image_title = tk.Label(
            left_frame,
            text="Selected Image",
            font=("Arial", 14, "bold"),
            bg='#f0f0f0',
            fg='black'
        )
        image_title.pack(pady=(0, 10))

        # Image display frame with border
        self.image_frame = tk.Frame(left_frame, bg='white', relief=tk.RIDGE, bd=3)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(
            self.image_frame,
            text="\n\nNo image selected\n\nClick 'Browse Image' to start",
            font=("Arial", 14),
            bg='white',
            fg='black',
            justify=tk.CENTER
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Right side - Results
        right_frame = tk.Frame(content_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)

        results_title = tk.Label(
            right_frame,
            text="Prediction Results",
            font=("Arial", 14, "bold"),
            bg='#f0f0f0',
            fg='black'
        )
        results_title.pack(pady=(0, 10))

        # Results frame
        results_frame = tk.Frame(right_frame, bg='#ecf0f1', relief=tk.RIDGE, bd=3)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Results display
        self.results_text = tk.Text(
            results_frame,
            font=("Arial", 12),
            bg='white',
            height=20,
            width=40,
            relief=tk.FLAT,
            padx=15,
            pady=15,
            wrap=tk.WORD
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.results_text.config(state=tk.DISABLED)

        # Buttons frame
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=20)

        # Browse button
        self.browse_btn = tk.Button(
            button_frame,
            text="Browse Image",
            command=self.browse_file,
            font=("Arial", 12, "bold"),
            bg='#c9c9c9',
            fg='black',
            padx=25,
            pady=12,
            relief=tk.RAISED,
            cursor="hand2",
            borderwidth=2
        )
        self.browse_btn.grid(row=0, column=0, padx=10)

        # Predict button
        self.predict_btn = tk.Button(
            button_frame,
            text="Predict",
            command=self.predict_image,
            font=("Arial", 12, "bold"),
            bg='#c9c9c9',
            fg='black',
            padx=25,
            pady=12,
            relief=tk.RAISED,
            cursor="hand2",
            state=tk.DISABLED,
            borderwidth=2
        )
        self.predict_btn.grid(row=0, column=1, padx=10)

        # Clear button
        self.clear_btn = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_all,
            font=("Arial", 12, "bold"),
            bg='#c9c9c9',
            fg='black',
            padx=25,
            pady=12,
            relief=tk.RAISED,
            cursor="hand2",
            borderwidth=2
        )
        self.clear_btn.grid(row=0, column=2, padx=10)

        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Ready to classify | Select an image to begin",
            font=("Arial", 10),
            bg='#c9c9c9',
            fg='black',
            anchor=tk.W,
            padx=10,
            pady=5
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def browse_file(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Image loaded: {Path(file_path).name}")

            # Clear previous results
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, "\n\n\n    Click 'Predict' to\n    classify this image")
            self.results_text.tag_add("center", "1.0", tk.END)
            self.results_text.tag_config("center", justify='center', foreground='#95a5a6')
            self.results_text.config(state=tk.DISABLED)

    def display_image(self, img_path):
        """Display the selected image in the UI"""
        try:
            # Open and resize image for display
            img = Image.open(img_path)

            # Calculate size to fit in frame while maintaining aspect ratio
            display_width = 400
            display_height = 400

            img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)

            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
        except Exception as e:
            messagebox.showerror("Error", f"Could not display image: {e}")

    def _get_category(self, class_name):
        """Determine if item is fruit or vegetable"""
        base_name = class_name.split()[0].lower()
        return self.category_map.get(base_name, 'fruit')

    def _load_image_for_prediction(self, img_path):
        """Load and preprocess image for prediction"""
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.flatten()

    def predict_image(self):
        """Predict the category and type of the selected image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return

        self.status_label.config(text="Analyzing image...")
        self.root.update()

        try:
            # Load and predict
            img_data = self._load_image_for_prediction(self.current_image_path)
            if img_data is None:
                messagebox.showerror("Error", "Could not process the image!")
                return

            # Reshape for prediction
            img_data = img_data.reshape(1, -1)

            # Predict
            prediction = self.model.predict(img_data)
            probabilities = self.model.predict_proba(img_data)

            # Get results
            class_name = self.label_encoder.inverse_transform(prediction)[0]
            category = self._get_category(class_name)
            confidence = np.max(probabilities)

            # Display results
            self.display_results(category, class_name, confidence, probabilities[0])
            self.status_label.config(text="Prediction complete!")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.status_label.config(text="Prediction failed")

    def display_results(self, category, class_name, confidence, all_probs):
        """Display prediction results in the text area"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        # Get top 3 predictions
        top_indices = np.argsort(all_probs)[-3:][::-1]
        top_predictions = [(self.label_encoder.classes_[i], all_probs[i]) for i in top_indices]

        # Confidence indicator
        if confidence >= 0.8:
            conf_indicator = "HIGH"
        elif confidence >= 0.5:
            conf_indicator = "MEDIUM"
        else:
            conf_indicator = "LOW"

        # Format results
        results = f"""
{'=' * 40}
      PREDICTION RESULTS
{'=' * 40}

Category:    {category.upper()}

Type:        {class_name}

Confidence:  {confidence:.1%}
             {conf_indicator}

{'=' * 40}
TOP 3 PREDICTIONS:
{'=' * 40}

"""

        for i, (name, prob) in enumerate(top_predictions, 1):
            results += f"{i}. {name}\n   {prob:.1%}\n\n"

        self.results_text.insert(1.0, results)

        # Add color coding for category
        if category.lower() == 'fruit':
            self.results_text.tag_add("category", "5.13", "5.18")
            self.results_text.tag_config("category", foreground="#e74c3c", font=("Arial", 12, "bold"))
        else:
            self.results_text.tag_add("category", "5.13", "5.22")
            self.results_text.tag_config("category", foreground="#27ae60", font=("Arial", 12, "bold"))

        # Highlight the predicted type
        self.results_text.tag_add("type", "7.13", f"7.{13 + len(class_name)}")
        self.results_text.tag_config("type", foreground="#2980b9", font=("Arial", 12, "bold"))

        self.results_text.config(state=tk.DISABLED)

    def clear_all(self):
        """Clear the image and results"""
        self.current_image_path = None
        self.image_label.config(
            image='',
            text="\n\nNo image selected\n\nClick 'Browse Image' to start"
        )
        self.image_label.image = None

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)

        self.predict_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Ready to classify | Select an image to begin")

    def run(self):
        """Start the UI"""
        self.root.mainloop()


if __name__ == "__main__":
    try:
        # Initialize predictor with UI
        app = FruitPredictorUI("fruit_classifier_model.pkl")
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")