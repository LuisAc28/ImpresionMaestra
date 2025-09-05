import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageOps, ImageTk
import rectpack
import os

# Global variables
image_paths = []
thumbnail_references = [] # To prevent garbage collection

def upload_files():
    """
    Opens a file dialog, loads images, and displays thumbnails in the preview panel.
    """
    global image_paths, thumbnail_references

    file_paths_tuple = filedialog.askopenfilenames(
        title="Seleccionar imágenes",
        filetypes=(("Archivos de imagen", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*"))
    )

    # 1. Clear previous state
    for widget in scrollable_frame.winfo_children():
        widget.destroy()
    thumbnail_references.clear()

    if file_paths_tuple:
        image_paths = list(file_paths_tuple)

        # 2. Display new thumbnails in a grid
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path)
                img.thumbnail((100, 100))
                photo_img = ImageTk.PhotoImage(img)

                # 3. Keep a reference to avoid garbage collection
                thumbnail_references.append(photo_img)

                thumb_frame = ttk.Frame(scrollable_frame, padding=5)
                img_label = ttk.Label(thumb_frame, image=photo_img)
                img_label.pack()

                filename = os.path.basename(path)
                name_label = ttk.Label(thumb_frame, text=filename, wraplength=100, justify='center')
                name_label.pack()

                row, col = divmod(i, 4) # 4 columns
                thumb_frame.grid(row=row, column=col, padx=5, pady=5)

            except Exception as e:
                print(f"Error loading thumbnail for {path}: {e}")

        generate_button.config(state=tk.NORMAL)
    else:
        # No files selected, ensure everything is cleared
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

        # 2. Use rectpack to find optimal positions, with rotation enabled
        packer = rectpack.newPacker(pack_algo=rectpack.MaxRectsBl, sort_algo=rectpack.SORT_AREA, rotation=True)

        for img in images_data:
            packer.add_rect(img['width'], img['height'], rid=img['path'])

        packer.add_bin(bin_width, bin_height, count=float('inf'))
        packer.pack()

        # 3. Check for unpacked images
        all_rids = {img['path'] for img in images_data}
        packed_rids = {rect.rid for abin in packer for rect in abin}
        unpacked_rids = all_rids - packed_rids

        if unpacked_rids:
            msg = "Las siguientes imágenes son demasiado grandes para caber en una página y no se incluirán:\n\n"
            for rid in unpacked_rids:
                msg += f"- {os.path.basename(rid)}\n"
            messagebox.showwarning("Imágenes Grandes Omitidas", msg)

        if not any(packer):
            messagebox.showerror("Error", "No se pudo empaquetar ninguna imagen. El PDF no será generado.")
            return

        # 4. Generate the PDF from the packed result
        c = canvas.Canvas(save_path, pagesize=A4)

        for i, abin in enumerate(packer):
            if i > 0:
                c.showPage()

            for rect in abin:
                # Load the original image
                img = Image.open(rect.rid)
                # Rotate it if the packer decided to
                if rect.rotated:
                    img = img.rotate(90, expand=True)

                # Define position and draw the (potentially rotated) image
                x = margin + rect.x
                y = margin + rect.y
                c.drawImage(ImageReader(img), x, y, width=rect.width, height=rect.height)

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
root.title("Impresión Maestra - v1.2")
root.geometry("1024x768") # Increased size for the new layout

# Main container with two panels
main_container = ttk.Frame(root)
main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Left panel for controls
controls_panel = ttk.Frame(main_container, width=320)
controls_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
controls_panel.pack_propagate(False) # Prevent panel from shrinking

# Right panel for preview
preview_panel = ttk.LabelFrame(main_container, text="Previsualización")
preview_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# --- Create Scrollable Canvas for Thumbnails ---
preview_canvas = tk.Canvas(preview_panel, borderwidth=0)
scrollbar = ttk.Scrollbar(preview_panel, orient="vertical", command=preview_canvas.yview)
scrollable_frame = ttk.Frame(preview_canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: preview_canvas.configure(
        scrollregion=preview_canvas.bbox("all")
    )
)

canvas_frame = preview_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
preview_canvas.configure(yscrollcommand=scrollbar.set)

# Add mouse wheel scrolling
def _on_mouse_wheel(event):
    # Cross-platform mouse wheel scrolling
    if event.num == 5 or event.delta < 0:
        preview_canvas.yview_scroll(1, "units")
    elif event.num == 4 or event.delta > 0:
        preview_canvas.yview_scroll(-1, "units")

# Bind mouse wheel to the canvas for scrolling
preview_canvas.bind_all("<MouseWheel>", _on_mouse_wheel)
preview_canvas.bind_all("<Button-4>", _on_mouse_wheel)
preview_canvas.bind_all("<Button-5>", _on_mouse_wheel)

# Pack the canvas and scrollbar
preview_canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")


# --- Populate Controls Panel ---
options_frame = ttk.LabelFrame(controls_panel, text="Opciones de Diseño", padding=(10, 5))
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

action_frame = ttk.LabelFrame(controls_panel, text="Acciones", padding=(10, 5))
action_frame.pack(fill=tk.X, pady=5)

load_button = ttk.Button(action_frame, text="Cargar Imágenes", command=upload_files)
load_button.pack(side=tk.LEFT, padx=5, pady=5)

generate_button = ttk.Button(action_frame, text="Generar PDF", command=generate_pdf, state=tk.DISABLED)
generate_button.pack(side=tk.LEFT, padx=5, pady=5)


root.mainloop()
