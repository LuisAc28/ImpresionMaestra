import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageOps

# Global variable to store image paths
image_paths = []

def upload_files():
    """
    Opens a file dialog to select multiple image files (JPG, PNG),
    stores their paths, and enables the PDF generation button.
    """
    global image_paths
    file_paths = filedialog.askopenfilenames(
        title="Seleccionar imágenes",
        filetypes=(("Archivos de imagen", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*"))
    )
    if file_paths:
        image_paths = list(file_paths)
        print(f"{len(image_paths)} archivos seleccionados.")
        generate_button.config(state=tk.NORMAL)
    else:
        image_paths = []
        generate_button.config(state=tk.DISABLED)

def generate_pdf():
    """
    Generates a PDF with the selected images based on the chosen layout and fit mode.
    """
    if not image_paths:
        messagebox.showinfo("Información", "No hay imágenes seleccionadas para generar el PDF.")
        return

    # Get selected options from UI
    layout_choice = layout_var.get()
    fit_mode = fit_mode_var.get()

    # Define layout configurations
    layout_configs = {
        "1 por hoja": (1, 1),
        "2 por hoja": (1, 2), # Changed to 1 row, 2 cols for landscape-like feel
        "4 por hoja (2x2)": (2, 2),
        "6 por hoja (2x3)": (2, 3)
    }
    rows, cols = layout_configs[layout_choice]
    chunk_size = rows * cols

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

        # Define margins and calculate cell dimensions
        margin = 1 * cm
        cell_width = (width - 2 * margin) / cols
        cell_height = (height - 2 * margin) / rows

        # Process images in chunks
        for i in range(0, len(image_paths), chunk_size):
            chunk = image_paths[i:i+chunk_size]

            if i > 0:
                c.showPage()

            # Generate positions for the current page's grid
            positions = []
            for row in range(rows):
                for col in range(cols):
                    # Calculate bottom-left corner of the cell
                    pos_x = margin + col * cell_width
                    pos_y = margin + (rows - 1 - row) * cell_height
                    positions.append((pos_x, pos_y))

            # Draw the images in the current chunk
            for j, img_path in enumerate(chunk):
                pos_x, pos_y = positions[j]

                # Logic for fit modes
                if fit_mode == "Ajustar":
                    # "Fit" mode: preserve aspect ratio, leave borders
                    c.drawImage(ImageReader(img_path), pos_x, pos_y,
                                width=cell_width, height=cell_height,
                                preserveAspectRatio=True, anchor='c')
                elif fit_mode == "Rellenar":
                    # "Fill" mode: crop image to fill the cell
                    img = Image.open(img_path)
                    # Use ImageOps.fit to scale and crop the image
                    cropped_img = ImageOps.fit(img, (int(cell_width), int(cell_height)),
                                               method=Image.Resampling.LANCZOS)
                    c.drawImage(ImageReader(cropped_img), pos_x, pos_y,
                                width=cell_width, height=cell_height)

        c.save()
        messagebox.showinfo("Éxito", f"PDF guardado exitosamente en:\n{save_path}")

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al generar el PDF:\n{e}")


# --- UI Setup ---
root = tk.Tk()
root.title("Impresión Maestra - v1.1")
root.geometry("800x600")

# --- Main container frame ---
main_frame = tk.Frame(root, padx=10, pady=10)
main_frame.pack(fill=tk.BOTH, expand=True)

# --- Options Frame ---
options_frame = ttk.LabelFrame(main_frame, text="Opciones de Diseño", padding=(10, 5))
options_frame.pack(fill=tk.X, pady=5)

# Layout selection
layout_label = ttk.Label(options_frame, text="Diseño por Hoja:")
layout_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
layout_var = tk.StringVar()
layout_options = ["1 por hoja", "2 por hoja", "4 por hoja (2x2)", "6 por hoja (2x3)"]
layout_combo = ttk.Combobox(options_frame, textvariable=layout_var, values=layout_options, state="readonly")
layout_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
layout_combo.set("4 por hoja (2x2)")

# Fit mode selection
fit_mode_label = ttk.Label(options_frame, text="Modo de Ajuste:")
fit_mode_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
fit_mode_var = tk.StringVar()
fit_mode_frame = ttk.Frame(options_frame)
fit_mode_frame.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
fit_radio_fit = ttk.Radiobutton(fit_mode_frame, text="Ajustar (con bordes)", variable=fit_mode_var, value="Ajustar")
fit_radio_fill = ttk.Radiobutton(fit_mode_frame, text="Rellenar (recorte)", variable=fit_mode_var, value="Rellenar")
fit_radio_fit.pack(side=tk.LEFT, expand=True)
fit_radio_fill.pack(side=tk.LEFT, expand=True)
fit_mode_var.set("Ajustar")

options_frame.columnconfigure(1, weight=1)

# --- Action Buttons Frame ---
action_frame = ttk.LabelFrame(main_frame, text="Acciones", padding=(10, 5))
action_frame.pack(fill=tk.X, pady=5)

# Load Images Button
load_button = ttk.Button(action_frame, text="Cargar Imágenes", command=upload_files)
load_button.pack(side=tk.LEFT, padx=5, pady=5)

# Generate PDF Button
generate_button = ttk.Button(action_frame, text="Generar PDF", command=generate_pdf, state=tk.DISABLED)
generate_button.pack(side=tk.LEFT, padx=5, pady=5)

# Start the main loop
root.mainloop()
