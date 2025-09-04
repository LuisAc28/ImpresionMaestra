import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageOps
import rectpack

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

def run_mosaic_layout(image_paths, save_path):
    """
    Generates a PDF using a bin-packing algorithm for a mosaic layout.
    """
    try:
        margin = 1 * cm
        page_width, page_height = A4
        bin_width = page_width - 2 * margin
        bin_height = page_height - 2 * margin

        # 1. Read image dimensions and associate with path
        images_data = []
        for path in image_paths:
            with Image.open(path) as img:
                w, h = img.size
                images_data.append({'width': w, 'height': h, 'path': path})

        # 2. Use rectpack to find optimal positions
        packer = rectpack.newPacker(pack_algo=rectpack.MaxRectsBl, sort_algo=rectpack.SORT_AREA)

        for img in images_data:
            packer.add_rect(img['width'], img['height'], rid=img['path'])

        packer.add_bin(bin_width, bin_height, count=float('inf'))
        packer.pack()

        # 3. Generate the PDF from the packed result
        c = canvas.Canvas(save_path, pagesize=A4)

        for i, abin in enumerate(packer):
            if i > 0:
                c.showPage()

            for rect in abin:
                x = margin + rect.x
                y = margin + rect.y
                c.drawImage(rect.rid, x, y, width=rect.width, height=rect.height)

        c.save()
        messagebox.showinfo("Éxito", f"PDF en modo Mosaico guardado en:\n{save_path}")

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error en el modo Mosaico:\n{e}")

def generate_pdf():
    """
    Generates a PDF with the selected images based on the chosen layout and fit mode.
    """
    if not image_paths:
        messagebox.showinfo("Información", "No hay imágenes seleccionadas para generar el PDF.")
        return

    layout_choice = layout_var.get()

    save_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("Archivos PDF", "*.pdf"), ("Todos los archivos", "*.*")],
        title="Guardar PDF como..."
    )

    if not save_path:
        return

    # --- Main logic branch ---
    if layout_choice == "Mosaico (Ahorro de papel)":
        run_mosaic_layout(image_paths, save_path)
    else:
        # --- Grid-based layout logic ---
        try:
            fit_mode = fit_mode_var.get()
            layout_configs = {
                "1 por hoja": (1, 1), "2 por hoja": (1, 2),
                "4 por hoja (2x2)": (2, 2), "6 por hoja (2x3)": (2, 3)
            }
            rows, cols = layout_configs[layout_choice]
            chunk_size = rows * cols

            c = canvas.Canvas(save_path, pagesize=A4)
            width, height = A4
            margin = 1 * cm
            cell_width = (width - 2 * margin) / cols
            cell_height = (height - 2 * margin) / rows

            for i in range(0, len(image_paths), chunk_size):
                chunk = image_paths[i:i+chunk_size]
                if i > 0:
                    c.showPage()

                positions = []
                for row in range(rows):
                    for col in range(cols):
                        pos_x = margin + col * cell_width
                        pos_y = margin + (rows - 1 - row) * cell_height
                        positions.append((pos_x, pos_y))

                for j, img_path in enumerate(chunk):
                    pos_x, pos_y = positions[j]
                    if fit_mode == "Ajustar":
                        c.drawImage(ImageReader(img_path), pos_x, pos_y, width=cell_width, height=cell_height, preserveAspectRatio=True, anchor='c')
                    elif fit_mode == "Rellenar":
                        img = Image.open(img_path)
                        cropped_img = ImageOps.fit(img, (int(cell_width), int(cell_height)), method=Image.Resampling.LANCZOS)
                        c.drawImage(ImageReader(cropped_img), pos_x, pos_y, width=cell_width, height=cell_height)

            c.save()
            messagebox.showinfo("Éxito", f"PDF guardado exitosamente en:\n{save_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al generar el PDF:\n{e}")


# --- UI Setup ---
root = tk.Tk()
root.title("Impresión Maestra - v1.2") # Version bump
root.geometry("800x600")

main_frame = tk.Frame(root, padx=10, pady=10)
main_frame.pack(fill=tk.BOTH, expand=True)

options_frame = ttk.LabelFrame(main_frame, text="Opciones de Diseño", padding=(10, 5))
options_frame.pack(fill=tk.X, pady=5)

layout_label = ttk.Label(options_frame, text="Diseño por Hoja:")
layout_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
layout_var = tk.StringVar()
layout_options = ["1 por hoja", "2 por hoja", "4 por hoja (2x2)", "6 por hoja (2x3)", "Mosaico (Ahorro de papel)"]
layout_combo = ttk.Combobox(options_frame, textvariable=layout_var, values=layout_options, state="readonly")
layout_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
layout_combo.set("4 por hoja (2x2)")

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

action_frame = ttk.LabelFrame(main_frame, text="Acciones", padding=(10, 5))
action_frame.pack(fill=tk.X, pady=5)

load_button = ttk.Button(action_frame, text="Cargar Imágenes", command=upload_files)
load_button.pack(side=tk.LEFT, padx=5, pady=5)

generate_button = ttk.Button(action_frame, text="Generar PDF", command=generate_pdf, state=tk.DISABLED)
generate_button.pack(side=tk.LEFT, padx=5, pady=5)

root.mainloop()
