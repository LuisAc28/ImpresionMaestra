import tkinter as tk
from tkinter import filedialog, messagebox
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from PIL import Image

# Global variable to store image paths
image_paths = []

def upload_files():
    """
    Opens a file dialog to select multiple image files (JPG, PNG),
    stores their paths, and enables the PDF generation button.
    """
    global image_paths
    # Open file dialog
    file_paths = filedialog.askopenfilenames(
        title="Seleccionar imágenes",
        filetypes=(
            ("Archivos de imagen", "*.jpg *.jpeg *.png"),
            ("Todos los archivos", "*.*")
        )
    )

    if file_paths:
        image_paths = list(file_paths)
        print("Archivos seleccionados:")
        for path in image_paths:
            print(path)
        # Enable the generate button if images are selected
        generate_button.config(state=tk.NORMAL)
    else:
        # Disable if no files are selected
        image_paths = []
        generate_button.config(state=tk.DISABLED)

def generate_pdf():
    """
    Generates a PDF with the selected images arranged in a 2x2 grid.
    """
    if not image_paths:
        messagebox.showinfo("Información", "No hay imágenes seleccionadas para generar el PDF.")
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("Archivos PDF", "*.pdf"), ("Todos los archivos", "*.*")],
        title="Guardar PDF como..."
    )

    if not save_path:
        return  # User cancelled

    try:
        c = canvas.Canvas(save_path, pagesize=A4)
        width, height = A4

        # Define margins and quadrant dimensions
        margin = 1 * cm
        quadrant_width = (width - 2 * margin) / 2
        quadrant_height = (height - 2 * margin) / 2

        # Quadrant bottom-left starting positions (from bottom to top)
        positions = [
            (margin, margin),  # Bottom-left
            (margin + quadrant_width, margin),  # Bottom-right
            (margin, margin + quadrant_height),  # Top-left
            (margin + quadrant_width, margin + quadrant_height),  # Top-right
        ]

        # Process images in chunks of 4
        for i in range(0, len(image_paths), 4):
            chunk = image_paths[i:i+4]

            # Add a new page for each new chunk of images
            if i > 0:
                c.showPage()

            # Draw the 4 images on the current page
            for j, img_path in enumerate(chunk):
                pos_x, pos_y = positions[j]
                # Draw image fitting in the quadrant
                c.drawImage(ImageReader(img_path), pos_x, pos_y,
                            width=quadrant_width, height=quadrant_height,
                            preserveAspectRatio=True, anchor='c')

        c.save()
        messagebox.showinfo("Éxito", f"PDF guardado exitosamente en:\n{save_path}")

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al generar el PDF:\n{e}")

# --- UI Setup ---
root = tk.Tk()
root.title("Impresión Maestra - v1.0")
root.geometry("800x600")

# --- Widgets ---
# Frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Load Images Button
load_button = tk.Button(button_frame, text="Cargar Imágenes", command=upload_files)
load_button.pack(side=tk.LEFT, padx=5)

# Generate PDF Button
generate_button = tk.Button(button_frame, text="Generar PDF", command=generate_pdf, state=tk.DISABLED)
generate_button.pack(side=tk.LEFT, padx=5)


# Start the main loop
root.mainloop()
