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
paper_dims = {} # To store on-screen paper dimensions

def upload_files():
    """
    Opens a file dialog to select image paths and triggers a preview update.
    """
    global image_paths
    file_paths_tuple = filedialog.askopenfilenames(
        title="Seleccionar imágenes",
        filetypes=(("Archivos de imagen", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*"))
    )
    if file_paths_tuple:
        image_paths = list(file_paths_tuple)
        preview_button.config(state=tk.NORMAL)
        generate_button.config(state=tk.NORMAL)
        update_preview()
    else:
        image_paths = []
        preview_button.config(state=tk.DISABLED)
        generate_button.config(state=tk.DISABLED)
        update_preview()

# --- Layout Calculation Logic ---
# (This section is unchanged from the previous step)
def calculate_mosaic_layout(image_paths):
    margin = 1 * cm
    page_width, page_height = A4
    bin_width = page_width - 2 * margin
    bin_height = page_height - 2 * margin
    images_data = [{'width': w, 'height': h, 'path': p}
                   for p in image_paths for w, h in [Image.open(p).size]]
    packer = rectpack.newPacker(pack_algo=rectpack.MaxRectsBl, sort_algo=rectpack.SORT_AREA, rotation=True)
    for img in images_data:
        packer.add_rect(img['width'], img['height'], rid=img['path'])
    packer.add_bin(bin_width, bin_height, count=float('inf'))
    packer.pack()
    all_rids = {img['path'] for img in images_data}
    packed_rids = {rect.rid for abin in packer for rect in abin}
    unpacked_paths = list(all_rids - packed_rids)
    return packer, unpacked_paths

def calculate_grid_layout(image_paths, layout_choice):
    layout_configs = {"1 por hoja": (1, 1), "2 por hoja": (1, 2), "4 por hoja (2x2)": (2, 2), "6 por hoja (2x3)": (2, 3)}
    rows, cols = layout_configs[layout_choice]
    chunk_size = rows * cols
    pages = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]
    return pages

# --- PDF Generation / Preview Logic ---
def update_preview():
    """
    Updates the preview canvas with a visual representation of the layout.
    """
    preview_canvas.delete("layout_item")
    # Store references to PhotoImage objects to prevent garbage collection
    preview_canvas.thumbnail_references = []

    if not image_paths or not paper_dims:
        return

    layout_choice = layout_var.get()

    x0, y0, paper_w_px, paper_h_px = paper_dims['x'], paper_dims['y'], paper_dims['w'], paper_dims['h']
    A4_w_pt, A4_h_pt = A4
    scale = paper_w_px / A4_w_pt
    margin_pt = 1 * cm

    if layout_choice == "Mosaico (Ahorro de papel)":
        try:
            packer, _ = calculate_mosaic_layout(image_paths)
            if any(packer):
                first_page = packer[0]
                for rect in first_page:
                    px = x0 + (margin_pt + rect.x) * scale
                    py = y0 + paper_h_px - (margin_pt + rect.y + rect.height) * scale
                    pw = rect.width * scale
                    ph = rect.height * scale

                    # --- Thumbnail Generation ---
                    img = Image.open(rect.rid)
                    if rect.rotated:
                        img = img.rotate(90, expand=True)

                    # Resize with high-quality downsampling
                    resized_img = img.resize((int(pw), int(ph)), Image.Resampling.LANCZOS)
                    photo_img = ImageTk.PhotoImage(resized_img)

                    # Store reference and draw on canvas
                    preview_canvas.thumbnail_references.append(photo_img)
                    preview_canvas.create_image(px, py, image=photo_img, anchor="nw", tags="layout_item")

        except Exception as e:
            messagebox.showerror("Error de Previsualización", f"No se pudo previsualizar el modo mosaico:\n{e}")
    else:
        pages = calculate_grid_layout(image_paths, layout_choice)
        if not pages: return

        layout_configs = {"1 por hoja": (1, 1), "2 por hoja": (1, 2), "4 por hoja (2x2)": (2, 2), "6 por hoja (2x3)": (2, 3)}
        rows, cols = layout_configs[layout_choice]
        cell_width_pt = (A4_w_pt - 2 * margin_pt) / cols
        cell_height_pt = (A4_h_pt - 2 * margin_pt) / rows

        first_page_paths = pages[0]
        for i, path in enumerate(first_page_paths):
            row, col = divmod(i, cols)

            x_pt = margin_pt + col * cell_width_pt
            y_pt = margin_pt + (rows - 1 - row) * cell_height_pt

            # For grid, we just need the cell dimensions for the placeholder
            pw = cell_width_pt * scale
            ph = cell_height_pt * scale
            px = x0 + x_pt * scale
            py = y0 + paper_h_px - (y_pt + cell_height_pt) * scale

            # --- Thumbnail Generation ---
            try:
                img = Image.open(path)
                # Fit the image within the cell, preserving aspect ratio
                img.thumbnail((int(pw), int(ph)), Image.Resampling.LANCZOS)
                photo_img = ImageTk.PhotoImage(img)

                # Center the thumbnail in the cell
                img_w, img_h = img.size
                px_centered = px + (pw - img_w) / 2
                py_centered = py + (ph - img_h) / 2

                # Store reference and draw on canvas
                preview_canvas.thumbnail_references.append(photo_img)
                preview_canvas.create_image(px_centered, py_centered, image=photo_img, anchor="nw", tags="layout_item")

            except Exception as e:
                # Draw a placeholder if image fails to load
                preview_canvas.create_rectangle(px, py, px + pw, py + ph, outline="red", fill="pink", tags="layout_item")
                preview_canvas.create_text(px + 4, py + 4, text=f"Error:\n{os.path.basename(path)}", anchor="nw", font=("Arial", 7), fill="red", tags="layout_item")

def generate_pdf():
    if not image_paths:
        messagebox.showinfo("Información", "No hay imágenes seleccionadas.")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("Archivos PDF", "*.pdf"), ("Todos los archivos", "*.*")], title="Guardar PDF como...")
    if not save_path:
        return
    try:
        layout_choice = layout_var.get()
        if layout_choice == "Mosaico (Ahorro de papel)":
            packed_pages, unpacked = calculate_mosaic_layout(image_paths)
            if unpacked:
                msg = "Imágenes omitidas por ser demasiado grandes:\n\n" + "\n".join(f"- {os.path.basename(p)}" for p in unpacked)
                messagebox.showwarning("Imágenes Grandes Omitidas", msg)
            if not any(packed_pages):
                messagebox.showerror("Error", "No se pudo empaquetar ninguna imagen.")
                return
            draw_mosaic_pdf(packed_pages, save_path)
        else: # Grid modes
            pages = calculate_grid_layout(image_paths, layout_choice)
            fit_mode = fit_mode_var.get()
            draw_grid_pdf(pages, layout_choice, fit_mode, save_path)
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error inesperado:\n{e}")

def draw_mosaic_pdf(packer, save_path):
    margin = 1 * cm
    c = canvas.Canvas(save_path, pagesize=A4)
    for i, abin in enumerate(packer):
        if i > 0: c.showPage()
        for rect in abin:
            img = Image.open(rect.rid)
            if rect.rotated:
                img = img.rotate(90, expand=True)
            x = margin + rect.x
            y = margin + rect.y
            c.drawImage(ImageReader(img), x, y, width=rect.width, height=rect.height)
    c.save()
    messagebox.showinfo("Éxito", f"PDF en modo Mosaico guardado en:\n{save_path}")

def draw_grid_pdf(pages, layout_choice, fit_mode, save_path):
    layout_configs = {"1 por hoja": (1, 1), "2 por hoja": (1, 2), "4 por hoja (2x2)": (2, 2), "6 por hoja (2x3)": (2, 3)}
    rows, cols = layout_configs[layout_choice]
    c = canvas.Canvas(save_path, pagesize=A4)
    width, height = A4
    margin = 1 * cm
    cell_width = (width - 2 * margin) / cols
    cell_height = (height - 2 * margin) / rows
    for i, page_chunk in enumerate(pages):
        if i > 0: c.showPage()
        positions = [(margin + col * cell_width, margin + (rows - 1 - row) * cell_height) for row in range(rows) for col in range(cols)]
        for j, img_path in enumerate(page_chunk):
            pos_x, pos_y = positions[j]
            if fit_mode == "Ajustar":
                c.drawImage(ImageReader(img_path), pos_x, pos_y, width=cell_width, height=cell_height, preserveAspectRatio=True, anchor='c')
            elif fit_mode == "Rellenar":
                img = Image.open(img_path)
                cropped_img = ImageOps.fit(img, (int(cell_width), int(cell_height)), method=Image.Resampling.LANCZOS)
                c.drawImage(ImageReader(cropped_img), pos_x, pos_y, width=cell_width, height=cell_height)
    c.save()
    messagebox.showinfo("Éxito", f"PDF guardado exitosamente en:\n{save_path}")

# --- UI Setup ---
root = tk.Tk()
root.title("Impresión Maestra - v1.4")
root.geometry("1024x768")

main_container = ttk.Frame(root)
main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

controls_panel = ttk.Frame(main_container, width=320)
controls_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
controls_panel.pack_propagate(False)

preview_panel = ttk.LabelFrame(main_container, text="Previsualización del Diseño (Página 1)")
preview_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

preview_canvas = tk.Canvas(preview_panel, bg="lightgrey")
preview_canvas.pack(side="left", fill="both", expand=True)

def redraw_paper(event):
    global paper_dims
    preview_canvas.delete("all")
    canvas_w, canvas_h = event.width, event.height
    ar = 210 / 297
    paper_w = min(canvas_w * 0.95, (canvas_h * 0.95) * ar)
    paper_h = paper_w / ar
    if paper_h > canvas_h * 0.95:
        paper_h = canvas_h * 0.95
        paper_w = paper_h * ar
    x0 = (canvas_w - paper_w) / 2
    y0 = (canvas_h - paper_h) / 2
    preview_canvas.create_rectangle(x0, y0, x0 + paper_w, y0 + paper_h, fill="white", tags="paper")
    paper_dims = {'x': x0, 'y': y0, 'w': paper_w, 'h': paper_h}
    update_preview()

preview_canvas.bind("<Configure>", redraw_paper)

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

preview_button = ttk.Button(action_frame, text="Actualizar Previsualización", command=update_preview, state=tk.DISABLED)
preview_button.pack(side=tk.LEFT, padx=5, pady=5)

generate_button = ttk.Button(action_frame, text="Generar PDF", command=generate_pdf, state=tk.DISABLED)
generate_button.pack(side=tk.LEFT, padx=5, pady=5)

root.mainloop()
