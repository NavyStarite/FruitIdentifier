import os
import json
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from pathlib import Path
import shutil
from datetime import datetime
import platform

# Try to import winsound for Windows, but don't fail if not available
try:
    import winsound

    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class DatasetCleanerApp:
    def __init__(self, dataset_path="VeggiesFruits"):
        self.dataset_path = Path(dataset_path)
        self.session_file = self.dataset_path / "_cleaning_session.json"
        self.current_category = None
        self.current_class = None
        self.current_images = []
        self.current_index = 0
        self.deleted_count = 0
        self.kept_count = 0
        self.trash_folder = self.dataset_path / "_DELETED"

        # Create trash folder
        self.trash_folder.mkdir(exist_ok=True)

        # Load or create session
        self.reviewed_images = set()  # Set of reviewed image paths
        self.load_session()

        # Get all classes and filter out reviewed ones
        self.classes_list = self.get_all_classes()
        self.class_index = 0

        print(f"\nSession Status:")
        print(f"   Previously reviewed: {len(self.reviewed_images)} images")
        print(f"   Classes to review: {len(self.classes_list)}")

        # Setup UI
        self.setup_ui()

        # Load first class
        if self.classes_list:
            self.load_class(0)
        else:
            messagebox.showinfo("Completado", "¡No hay más imágenes para revisar!")
            exit()

    def play_sound(self, sound_type="class_change"):
        """Play system sound for feedback"""
        try:
            system = platform.system()

            if system == "Windows" and HAS_WINSOUND:
                if sound_type == "class_change":
                    # Class change - high frequency beep
                    winsound.Beep(800, 150)  # 800Hz for 150ms
                elif sound_type == "delete":
                    # Delete - low frequency beep
                    winsound.Beep(400, 100)  # 400Hz for 100ms
                elif sound_type == "keep":
                    # Keep - quick high beep
                    winsound.Beep(600, 80)  # 600Hz for 80ms
                elif sound_type == "save":
                    # Save - double beep
                    winsound.Beep(700, 100)
                    winsound.Beep(900, 100)
            elif system == "Darwin":  # macOS
                os.system('afplay /System/Library/Sounds/Pop.aiff &')
            elif system == "Linux":  # Linux
                # Use system bell as fallback
                print('\a')  # Terminal bell
        except Exception:
            # Silent fail if sound can't play
            pass

    def load_session(self):
        """Load previous cleaning session if exists"""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)

                self.reviewed_images = set(session_data.get('reviewed_images', []))
                self.deleted_count = session_data.get('deleted_count', 0)
                self.kept_count = session_data.get('kept_count', 0)

                last_saved = session_data.get('last_saved', 'Unknown')
                print(f"\nSesión anterior cargada (guardada: {last_saved})")
                print(f"   Imágenes ya revisadas: {len(self.reviewed_images)}")

            except Exception as e:
                print(f"No se pudo cargar sesión anterior: {e}")
                self.reviewed_images = set()
        else:
            print("\nNueva sesión iniciada")

    def save_session(self):
        """Save current cleaning session"""
        try:
            session_data = {
                'reviewed_images': list(self.reviewed_images),
                'deleted_count': self.deleted_count,
                'kept_count': self.kept_count,
                'last_saved': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Error guardando sesión: {e}")
            return False

    def get_all_classes(self):
        """Get all class folders from dataset, filtering out reviewed images"""
        classes = []
        total_new_images = 0

        for category_folder in self.dataset_path.iterdir():
            if not category_folder.is_dir() or category_folder.name.startswith("_"):
                continue

            category_name = category_folder.name

            for class_folder in category_folder.iterdir():
                if not class_folder.is_dir():
                    continue

                class_name = class_folder.name

                # Get all image files
                all_img_files = (
                        list(class_folder.glob('*.jpg')) +
                        list(class_folder.glob('*.jpeg')) +
                        list(class_folder.glob('*.png')) +
                        list(class_folder.glob('*.webp')) +
                        list(class_folder.glob('*.gif'))
                )

                # Filter out already reviewed images
                unreviewed_imgs = [
                    img for img in all_img_files
                    if str(img) not in self.reviewed_images
                ]

                if unreviewed_imgs:  # Only include classes with unreviewed images
                    classes.append({
                        'category': category_name,
                        'name': class_name,
                        'path': class_folder,
                        'images': unreviewed_imgs,
                        'total_images': len(all_img_files),
                        'reviewed_images': len(all_img_files) - len(unreviewed_imgs)
                    })
                    total_new_images += len(unreviewed_imgs)

        print(f"   Imágenes nuevas/sin revisar: {total_new_images}")

        return classes

    def setup_ui(self):
        """Create the Tinder-style UI"""
        self.root = tk.Tk()
        self.root.title("Dataset Cleaner - Tinder Style (Con Sesión)")
        self.root.geometry("1000x850")
        self.root.configure(bg='#2c3e50')

        # Bind keyboard shortcuts
        self.root.bind('<Left>', lambda e: self.delete_image())
        self.root.bind('<Right>', lambda e: self.keep_image())
        self.root.bind('<space>', lambda e: self.keep_image())
        self.root.bind('<Delete>', lambda e: self.delete_image())
        self.root.bind('<Escape>', lambda e: self.skip_class())
        self.root.bind('<n>', lambda e: self.next_class())
        self.root.bind('<p>', lambda e: self.prev_class())
        self.root.bind('<s>', lambda e: self.manual_save())

        # Session info bar
        session_frame = tk.Frame(self.root, bg='#16a085', height=40)
        session_frame.pack(fill=tk.X)
        session_frame.pack_propagate(False)

        self.session_label = tk.Label(
            session_frame,
            text="Sesión guardada automáticamente | Presiona 'S' para guardar manualmente",
            font=("Arial", 10),
            bg='#16a085',
            fg='white'
        )
        self.session_label.pack(expand=True)

        # Top bar - Stats
        stats_frame = tk.Frame(self.root, bg='#34495e', height=60)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        stats_frame.pack_propagate(False)

        self.stats_label = tk.Label(
            stats_frame,
            text="",
            font=("Arial", 14, "bold"),
            bg='#34495e',
            fg='white'
        )
        self.stats_label.pack(expand=True)

        # Class info bar
        class_frame = tk.Frame(self.root, bg='#34495e', height=100)
        class_frame.pack(fill=tk.X, pady=(0, 10))
        class_frame.pack_propagate(False)

        self.class_label = tk.Label(
            class_frame,
            text="",
            font=("Arial", 18, "bold"),
            bg='#34495e',
            fg='#3498db'
        )
        self.class_label.pack(pady=5)

        self.progress_label = tk.Label(
            class_frame,
            text="",
            font=("Arial", 12),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.progress_label.pack()

        self.class_progress_label = tk.Label(
            class_frame,
            text="",
            font=("Arial", 10),
            bg='#34495e',
            fg='#95a5a6'
        )
        self.class_progress_label.pack()

        # Image display area
        image_container = tk.Frame(self.root, bg='#2c3e50')
        image_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.image_frame = tk.Frame(image_container, bg='white', relief=tk.RIDGE, bd=3)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(
            self.image_frame,
            text="Cargando...",
            font=("Arial", 16),
            bg='white',
            fg='#95a5a6'
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Control buttons
        button_frame = tk.Frame(self.root, bg='#2c3e50')
        button_frame.pack(pady=15)

        # Delete button (Left)
        self.delete_btn = tk.Button(
            button_frame,
            text="ELIMINAR\n(← o Del)",
            command=self.delete_image,
            font=("Arial", 14, "bold"),
            bg='#e74c3c',
            fg='white',
            width=15,
            height=3,
            cursor="hand2"
        )
        self.delete_btn.grid(row=0, column=0, padx=20)

        # Keep button (Right)
        self.keep_btn = tk.Button(
            button_frame,
            text="MANTENER\n(→ o Space)",
            command=self.keep_image,
            font=("Arial", 14, "bold"),
            bg='#27ae60',
            fg='white',
            width=15,
            height=3,
            cursor="hand2"
        )
        self.keep_btn.grid(row=0, column=1, padx=20)

        # Navigation buttons
        nav_frame = tk.Frame(self.root, bg='#2c3e50')
        nav_frame.pack(pady=10)

        tk.Button(
            nav_frame,
            text="◄ Anterior (P)",
            command=self.prev_class,
            font=("Arial", 10),
            bg='#95a5a6',
            fg='white',
            padx=10,
            pady=5
        ).grid(row=0, column=0, padx=5)

        tk.Button(
            nav_frame,
            text="Saltar Clase (ESC)",
            command=self.skip_class,
            font=("Arial", 10),
            bg='#95a5a6',
            fg='white',
            padx=10,
            pady=5
        ).grid(row=0, column=1, padx=5)

        tk.Button(
            nav_frame,
            text="Siguiente (N) ►",
            command=self.next_class,
            font=("Arial", 10),
            bg='#95a5a6',
            fg='white',
            padx=10,
            pady=5
        ).grid(row=0, column=2, padx=5)

        tk.Button(
            nav_frame,
            text="Guardar (S)",
            command=self.manual_save,
            font=("Arial", 10),
            bg='#16a085',
            fg='white',
            padx=10,
            pady=5
        ).grid(row=0, column=3, padx=5)

        # Help bar
        help_frame = tk.Frame(self.root, bg='#34495e')
        help_frame.pack(fill=tk.X, side=tk.BOTTOM)

        help_text = "← Eliminar | → Mantener | Space Mantener | Del Eliminar | ESC Saltar | N/P Nav | S Guardar | Q Salir"
        tk.Label(
            help_frame,
            text=help_text,
            font=("Arial", 9),
            bg='#34495e',
            fg='#bdc3c7',
            pady=8
        ).pack()

        # Quit shortcut
        self.root.bind('<q>', lambda e: self.quit_app())

        # Auto-save on window close
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    def load_class(self, class_idx):
        """Load a specific class"""
        if class_idx < 0 or class_idx >= len(self.classes_list):
            self.save_session()
            self.play_sound("save")
            messagebox.showinfo("Completado",
                                f"¡Revisión completa!\n\n"
                                f"Total eliminadas: {self.deleted_count}\n"
                                f"Total conservadas: {self.kept_count}\n"
                                f"Total revisadas: {len(self.reviewed_images)}\n\n"
                                f"Sesión guardada.")
            self.root.quit()
            return

        # Play sound when changing class (but not on first load)
        if hasattr(self, 'current_class') and self.current_class is not None:
            self.play_sound("class_change")
            # Flash the class label
            self.flash_class_label()

        self.class_index = class_idx
        class_info = self.classes_list[class_idx]

        self.current_category = class_info['category']
        self.current_class = class_info['name']
        self.current_images = class_info['images']
        self.current_index = 0

        # Update class label
        self.class_label.config(
            text=f"{self.current_category} > {self.current_class}"
        )

        # Show class progress
        total_in_class = class_info['total_images']
        reviewed_in_class = class_info['reviewed_images']
        self.class_progress_label.config(
            text=f"Clase: {reviewed_in_class}/{total_in_class} ya revisadas | "
                 f"{len(self.current_images)} pendientes"
        )

        # Show first image
        self.show_current_image()

    def flash_class_label(self):
        """Flash the class label to indicate class change"""
        original_bg = self.class_label.cget('bg')
        original_fg = self.class_label.cget('fg')

        # Flash to bright color
        self.class_label.config(bg='#f39c12', fg='white')

        # Return to normal after 200ms
        self.root.after(200, lambda: self.class_label.config(bg=original_bg, fg=original_fg))

    def show_current_image(self):
        """Display current image"""
        if self.current_index >= len(self.current_images):
            # Finished this class, move to next
            self.next_class()
            return

        img_path = self.current_images[self.current_index]

        # Update progress
        total_images = len(self.current_images)
        remaining = total_images - self.current_index
        self.progress_label.config(
            text=f"Imagen {self.current_index + 1} de {total_images} (pendientes) | Quedan {remaining}"
        )

        # Update stats
        total_classes = len(self.classes_list)
        total_reviewed = len(self.reviewed_images)
        self.stats_label.config(
            text=f"Clase {self.class_index + 1}/{total_classes} | "
                 f"Eliminadas: {self.deleted_count} | Conservadas: {self.kept_count} | "
                 f"Total revisadas: {total_reviewed}"
        )

        try:
            # Load and display image
            img = Image.open(img_path)

            # Resize to fit display (max 700x450)
            display_width = 700
            display_height = 450
            img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)

            self.image_label.config(image=photo, text="")
            self.image_label.image = photo

        except Exception as e:
            self.image_label.config(
                text=f"Error cargando imagen:\n{img_path.name}\n{str(e)}",
                image=""
            )

    def delete_image(self):
        """Move current image to trash"""
        if self.current_index >= len(self.current_images):
            return

        img_path = self.current_images[self.current_index]

        try:
            # Create category/class folder in trash
            trash_class_folder = self.trash_folder / self.current_category / self.current_class
            trash_class_folder.mkdir(parents=True, exist_ok=True)

            # Move file to trash
            trash_path = trash_class_folder / img_path.name
            shutil.move(str(img_path), str(trash_path))

            # Play delete sound
            self.play_sound("delete")

            # Flash delete button
            self.flash_button(self.delete_btn, '#c0392b')

            self.deleted_count += 1
            self.reviewed_images.add(str(img_path))
            self.current_index += 1

            # Auto-save every 10 actions
            if (self.deleted_count + self.kept_count) % 10 == 0:
                self.save_session()

            self.show_current_image()

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo eliminar la imagen:\n{e}")

    def keep_image(self):
        """Keep current image and move to next"""
        if self.current_index >= len(self.current_images):
            return

        img_path = self.current_images[self.current_index]

        # Play keep sound
        self.play_sound("keep")

        # Flash keep button
        self.flash_button(self.keep_btn, '#229954')

        self.kept_count += 1
        self.reviewed_images.add(str(img_path))
        self.current_index += 1

        # Auto-save every 10 actions
        if (self.deleted_count + self.kept_count) % 10 == 0:
            self.save_session()

        self.show_current_image()

    def flash_button(self, button, flash_color):
        """Flash a button to provide visual feedback"""
        original_bg = button.cget('bg')
        button.config(bg=flash_color)
        self.root.after(100, lambda: button.config(bg=original_bg))

    def next_class(self):
        """Go to next class"""
        # Save before moving to next class
        self.save_session()
        self.load_class(self.class_index + 1)

    def prev_class(self):
        """Go to previous class"""
        if self.class_index > 0:
            self.save_session()
            self.load_class(self.class_index - 1)

    def skip_class(self):
        """Skip all remaining images in current class"""
        if messagebox.askyesno("Saltar Clase",
                               f"¿Marcar las {len(self.current_images) - self.current_index} imágenes restantes como conservadas?"):
            # Mark remaining as kept and reviewed
            for i in range(self.current_index, len(self.current_images)):
                img_path = self.current_images[i]
                self.reviewed_images.add(str(img_path))
                self.kept_count += 1

            self.save_session()
            self.next_class()

    def manual_save(self):
        """Manually save session"""
        if self.save_session():
            self.play_sound("save")
            self.session_label.config(
                text=f"Sesión guardada - {datetime.now().strftime('%H:%M:%S')}",
                bg='#27ae60'
            )
            # Reset message after 2 seconds
            self.root.after(2000, lambda: self.session_label.config(
                text="Sesión guardada automáticamente | Presiona 'S' para guardar manualmente",
                bg='#16a085'
            ))

    def quit_app(self):
        """Quit the application"""
        self.save_session()
        if messagebox.askyesno("Salir",
                               f"¿Salir de la aplicación?\n\n"
                               f"Eliminadas: {self.deleted_count}\n"
                               f"Conservadas: {self.kept_count}\n"
                               f"Revisadas: {len(self.reviewed_images)}\n\n"
                               f"Sesión guardada."):
            self.root.quit()

    def run(self):
        """Start the application"""
        self.root.mainloop()


def main():
    print("=" * 70)
    print("DATASET CLEANER - TINDER STYLE (CON SESIÓN)")
    print("=" * 70)
    print("\nControles:")
    print("  ← (Flecha izquierda) - Eliminar imagen")
    print("  → (Flecha derecha)   - Mantener imagen")
    print("  Space                - Mantener imagen")
    print("  Delete               - Eliminar imagen")
    print("  ESC                  - Saltar clase (marcar resto como conservadas)")
    print("  N                    - Siguiente clase")
    print("  P                    - Clase anterior")
    print("  S                    - Guardar sesión manualmente")
    print("  Q                    - Salir")
    print("\nLa sesión se guarda automáticamente cada 10 acciones")
    print("Imágenes eliminadas se mueven a: FrutasVerduras/_DELETED/")
    print("Archivo de sesión: FrutasVerduras/_cleaning_session.json")
    print("=" * 70)

    dataset_path = "FrutasVerduras"

    if not Path(dataset_path).exists():
        print(f"\nError: No se encontró la carpeta '{dataset_path}'")
        return

    app = DatasetCleanerApp(dataset_path)
    app.run()

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"Imágenes eliminadas: {app.deleted_count}")
    print(f"Imágenes conservadas: {app.kept_count}")
    print(f"Total revisadas: {len(app.reviewed_images)}")
    print(f"\nSesión guardada en: {app.session_file}")
    print("Puedes retomar la limpieza ejecutando el script nuevamente.")
    print("=" * 70)


if __name__ == "__main__":
    main()