import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageOps, ImageTk
import rectpack
import os

# Global variables
image_paths = []
paper_dims = {} # To store on-screen paper dimensions
preview_pages = [] # To store the layout data for all pages
current_preview_page_index = 0 # To track the current page in the preview

def get_page_size():
    """
    Returns the page dimensions (A4 or landscape A4) based on the UI selection.
    """
    if orientation_var.get() == "Horizontal":
        return landscape(A4)
    return A4

def _handle_file_selection(replace_current: bool):
    """
    Internal logic for opening file dialog and updating the image list.
    """
    global image_paths
    title = "Seleccionar imágenes para cargar" if replace_current else "Seleccionar imágenes para añadir"
    file_paths_tuple = filedialog.askopenfilenames(
        title=title,
        filetypes=(("Archivos de imagen", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*"))
    )

    if not file_paths_tuple:
        # If replacing, a cancelled dialog means clearing the list.
        if replace_current:
            image_paths = []
    else:
        if replace_current:
            image_paths = list(file_paths_tuple)
        else:
            # Ensure image_paths is a list before extending
            if not isinstance(image_paths, list):
                image_paths = []
            image_paths.extend(list(file_paths_tuple))

    # Update button states based on whether there are images
    if image_paths:
        preview_button.config(state=tk.NORMAL)
        generate_button.config(state=tk.NORMAL)
    else:
        preview_button.config(state=tk.DISABLED)
        generate_button.config(state=tk.DISABLED)

    update_preview()

def upload_files_replace():
    """Action for the 'Cargar Imágenes' button."""
    _handle_file_selection(replace_current=True)

def upload_files_add():
    """Action for the 'Añadir Imágenes' button."""
    _handle_file_selection(replace_current=False)

# --- Layout Calculation Logic ---
def get_grid_dimensions():
    """
    Determines the grid dimensions (rows, cols) based on the layout selection.
    This is the single source of truth for grid sizing.
    """
    layout_choice = layout_var.get()

    if layout_choice == "Personalizado...":
        try:
            # Get values from spinboxes, ensure they are at least 1
            rows = int(rows_var.get())
            cols = int(cols_var.get())
            return (rows if rows > 0 else 1, cols if cols > 0 else 1)
        except (ValueError, tk.TclError):
            return (1, 1)  # Default to 1x1 on error
    else:
        # The master dictionary for all predefined grid layouts
        layout_configs = {
            "1 por hoja": (1, 1),
            "2 por hoja": (1, 2),
            "3 por hoja (1x3)": (1, 3),
            "4 por hoja (2x2)": (2, 2),
            "6 por hoja (2x3)": (2, 3),
            "9 por hoja (3x3)": (3, 3)
        }
        # Return the config, or a default (e.g., 1x1) if not a grid mode
        return layout_configs.get(layout_choice, (1, 1))

def calculate_mosaic_layout(image_paths, pagesize):
    margin = 1 * cm
    page_width, page_height = pagesize
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

def calculate_grid_layout(image_paths):
    rows, cols = get_grid_dimensions()
    chunk_size = rows * cols
    # Handle case where chunk_size might be 0 if user enters invalid custom values
    if chunk_size == 0:
        return []
    pages = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]
    return pages

# --- PDF Generation / Preview Logic ---
def draw_preview_page():
    """
    Draws the currently selected page on the preview canvas.
    """
    preview_canvas.delete("layout_item")
    preview_canvas.thumbnail_references = [] # Reset references for the new page

    if not preview_pages or not paper_dims:
        return

    page_data = preview_pages[current_preview_page_index]
    layout_choice = layout_var.get()
    x0, y0, paper_w_px, paper_h_px = paper_dims['x'], paper_dims['y'], paper_dims['w'], paper_dims['h']
    page_width_pt, page_height_pt = get_page_size()
    scale = paper_w_px / page_width_pt
    margin_pt = 1 * cm

    if layout_choice == "Mosaico (Ahorro de papel)":
        for rect in page_data:
            px = x0 + (margin_pt + rect.x) * scale
            py = y0 + paper_h_px - (margin_pt + rect.y + rect.height) * scale
            pw = rect.width * scale
            ph = rect.height * scale
            try:
                img = Image.open(rect.rid)
                if rect.rotated:
                    img = img.rotate(90, expand=True)
                resized_img = img.resize((int(pw), int(ph)), Image.Resampling.LANCZOS)
                photo_img = ImageTk.PhotoImage(resized_img)
                preview_canvas.thumbnail_references.append(photo_img)
                preview_canvas.create_image(px, py, image=photo_img, anchor="nw", tags="layout_item")
            except Exception as e:
                preview_canvas.create_rectangle(px, py, px + pw, py + ph, outline="red", fill="pink", tags="layout_item")
                preview_canvas.create_text(px + 4, py + 4, text=f"Error:\n{os.path.basename(rect.rid)}", anchor="nw", font=("Arial", 7), fill="red", tags="layout_item")
    else: # Grid layouts
        rows, cols = get_grid_dimensions()
        if rows == 0 or cols == 0: return # Avoid division by zero
        cell_width_pt = (page_width_pt - 2 * margin_pt) / cols
        cell_height_pt = (page_height_pt - 2 * margin_pt) / rows
        for i, path in enumerate(page_data):
            row, col = divmod(i, cols)
            x_pt = margin_pt + col * cell_width_pt
            y_pt = margin_pt + (rows - 1 - row) * cell_height_pt
            pw = cell_width_pt * scale
            ph = cell_height_pt * scale
            px = x0 + x_pt * scale
            py = y0 + paper_h_px - (y_pt + cell_height_pt) * scale
            try:
                img = Image.open(path)
                img.thumbnail((int(pw), int(ph)), Image.Resampling.LANCZOS)
                photo_img = ImageTk.PhotoImage(img)
                img_w, img_h = img.size
                px_centered = px + (pw - img_w) / 2
                py_centered = py + (ph - img_h) / 2
                preview_canvas.thumbnail_references.append(photo_img)
                preview_canvas.create_image(px_centered, py_centered, image=photo_img, anchor="nw", tags="layout_item")
            except Exception as e:
                preview_canvas.create_rectangle(px, py, px + pw, py + ph, outline="red", fill="pink", tags="layout_item")
                preview_canvas.create_text(px + 4, py + 4, text=f"Error:\n{os.path.basename(path)}", anchor="nw", font=("Arial", 7), fill="red", tags="layout_item")

def update_preview():
    """
    Calculates the full layout and redraws the entire preview canvas.
    This is the main function for refreshing the preview pane.
    """
    global preview_pages, current_preview_page_index, paper_dims

    # 0. Save current state
    saved_page_index = current_preview_page_index

    # 1. Recalculate paper dimensions and redraw paper background
    canvas_w = preview_canvas.winfo_width()
    canvas_h = preview_canvas.winfo_height()
    if canvas_w <= 1 or canvas_h <= 1: # Avoids error on first launch
        return

    page_w_pt, page_h_pt = get_page_size()
    ar = page_w_pt / page_h_pt

    paper_w_px = min(canvas_w * 0.95, (canvas_h * 0.95) * ar)
    paper_h_px = paper_w_px / ar
    if paper_h_px > canvas_h * 0.95:
        paper_h_px = canvas_h * 0.95
        paper_w_px = paper_h_px * ar

    x0 = (canvas_w - paper_w_px) / 2
    y0 = (canvas_h - paper_h_px) / 2

    preview_canvas.delete("all")
    preview_canvas.create_rectangle(x0, y0, x0 + paper_w_px, y0 + paper_h_px, fill="white", tags="paper")
    paper_dims = {'x': x0, 'y': y0, 'w': paper_w_px, 'h': paper_h_px}

    # 2. Calculate layout for images
    preview_pages = []
    if not image_paths:
        pass
    else:
        layout_choice = layout_var.get()
        pagesize = (page_w_pt, page_h_pt)
        if layout_choice == "Mosaico (Ahorro de papel)":
            try:
                packer, _ = calculate_mosaic_layout(image_paths, pagesize)
                preview_pages = [bin for bin in packer if bin]
            except Exception as e:
                messagebox.showerror("Error de Cálculo", f"No se pudo calcular el diseño de mosaico:\n{e}")
                preview_pages = []
        else:
            preview_pages = calculate_grid_layout(image_paths)

    # 3. Restore page index
    new_total_pages = len(preview_pages)
    if new_total_pages == 0:
        current_preview_page_index = 0
    elif saved_page_index < new_total_pages:
        current_preview_page_index = saved_page_index
    else:
        current_preview_page_index = new_total_pages - 1

    # 4. Draw the content for the current page and update controls
    draw_preview_page()
    update_pagination_controls()

def show_previous_page():
    """Displays the previous page in the preview."""
    global current_preview_page_index
    if current_preview_page_index > 0:
        current_preview_page_index -= 1
        draw_preview_page()
        update_pagination_controls()

def show_next_page():
    """Displays the next page in the preview."""
    global current_preview_page_index
    if current_preview_page_index < len(preview_pages) - 1:
        current_preview_page_index += 1
        draw_preview_page()
        update_pagination_controls()

def update_pagination_controls():
    """Updates the state of the pagination buttons and page counter label."""
    total_pages = len(preview_pages)

    # Update label
    page_status_label.config(text=f"Página {current_preview_page_index + 1} de {total_pages}" if total_pages > 0 else "Página 0 de 0")

    # Update button states
    prev_page_button.config(state=tk.NORMAL if current_preview_page_index > 0 else tk.DISABLED)
    next_page_button.config(state=tk.NORMAL if current_preview_page_index < total_pages - 1 else tk.DISABLED)

def generate_pdf():
    if not image_paths:
        messagebox.showinfo("Información", "No hay imágenes seleccionadas.")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("Archivos PDF", "*.pdf"), ("Todos los archivos", "*.*")], title="Guardar PDF como...")
    if not save_path:
        return
    try:
        pagesize = get_page_size()
        layout_choice = layout_var.get()
        if layout_choice == "Mosaico (Ahorro de papel)":
            packed_pages, unpacked = calculate_mosaic_layout(image_paths, pagesize)
            if unpacked:
                msg = "Imágenes omitidas por ser demasiado grandes:\n\n" + "\n".join(f"- {os.path.basename(p)}" for p in unpacked)
                messagebox.showwarning("Imágenes Grandes Omitidas", msg)
            if not any(packed_pages):
                messagebox.showerror("Error", "No se pudo empaquetar ninguna imagen.")
                return
            draw_mosaic_pdf(packed_pages, save_path, pagesize)
        else: # Grid modes
            pages = calculate_grid_layout(image_paths)
            fit_mode = fit_mode_var.get()
            draw_grid_pdf(pages, fit_mode, save_path, pagesize)
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error inesperado:\n{e}")

def draw_mosaic_pdf(packer, save_path, pagesize):
    margin = 1 * cm
    c = canvas.Canvas(save_path, pagesize=pagesize)
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

def draw_grid_pdf(pages, fit_mode, save_path, pagesize):
    rows, cols = get_grid_dimensions()
    if rows == 0 or cols == 0:
        messagebox.showerror("Error de Diseño", "Las filas y columnas deben ser mayores que cero.")
        return
    c = canvas.Canvas(save_path, pagesize=pagesize)
    width, height = pagesize
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


def handle_layout_change(event=None):
    """
    Shows or hides the custom layout frame based on the combobox selection
    and triggers a preview update.
    """
    if layout_var.get() == "Personalizado...":
        custom_layout_frame.grid()
    else:
        custom_layout_frame.grid_remove()
    update_preview()


# --- UI Setup ---
root = tk.Tk()
root.title("Impresión Maestra - v1.5")
root.geometry("1024x768")

main_container = ttk.Frame(root)
main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

controls_panel = ttk.Frame(main_container, width=320)
controls_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
controls_panel.pack_propagate(False)

preview_panel = ttk.LabelFrame(main_container, text="Previsualización del Diseño")
preview_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

preview_canvas = tk.Canvas(preview_panel, bg="lightgrey")
preview_canvas.pack(side=tk.TOP, fill="both", expand=True)

# --- Pagination Controls ---
pagination_frame = ttk.Frame(preview_panel)
pagination_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=2)

# Center the frame content by giving weight to empty outer columns
pagination_frame.grid_columnconfigure(0, weight=1)
pagination_frame.grid_columnconfigure(2, weight=1)

prev_page_button = ttk.Button(pagination_frame, text="< Anterior", state=tk.DISABLED, command=show_previous_page)
prev_page_button.grid(row=0, column=0, sticky='e', padx=5, pady=2)

page_status_label = ttk.Label(pagination_frame, text="Página 0 de 0")
page_status_label.grid(row=0, column=1, padx=5, pady=2)

next_page_button = ttk.Button(pagination_frame, text="Siguiente >", state=tk.DISABLED, command=show_next_page)
next_page_button.grid(row=0, column=2, sticky='w', padx=5, pady=2)

def redraw_paper(event):
    """
    Handles canvas resize events by triggering a full preview update.
    """
    update_preview()

preview_canvas.bind("<Configure>", redraw_paper)

options_frame = ttk.LabelFrame(controls_panel, text="Opciones de Diseño", padding=(10, 5))
options_frame.pack(fill=tk.X, pady=5)

# --- Layout Selection ---
layout_label = ttk.Label(options_frame, text="Diseño por Hoja:")
layout_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
layout_var = tk.StringVar()
layout_options = [
    "1 por hoja", "2 por hoja", "3 por hoja (1x3)", "4 por hoja (2x2)",
    "6 por hoja (2x3)", "9 por hoja (3x3)", "Mosaico (Ahorro de papel)", "Personalizado..."
]
layout_combo = ttk.Combobox(options_frame, textvariable=layout_var, values=layout_options, state="readonly")
layout_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
layout_combo.set("4 por hoja (2x2)")
layout_combo.bind("<<ComboboxSelected>>", handle_layout_change)

# --- Custom Layout Frame (Initially Hidden) ---
custom_layout_frame = ttk.Frame(options_frame)
custom_layout_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=0, sticky="ew")

rows_label = ttk.Label(custom_layout_frame, text="Filas:")
rows_label.pack(side=tk.LEFT, padx=(5, 5))
rows_var = tk.StringVar(value="2")
rows_spinbox = tk.Spinbox(custom_layout_frame, from_=1, to=10, width=5, textvariable=rows_var, command=update_preview)
rows_spinbox.pack(side=tk.LEFT, padx=(0, 10))

cols_label = ttk.Label(custom_layout_frame, text="Columnas:")
cols_label.pack(side=tk.LEFT, padx=(5, 5))
cols_var = tk.StringVar(value="2")
cols_spinbox = tk.Spinbox(custom_layout_frame, from_=1, to=10, width=5, textvariable=cols_var, command=update_preview)
cols_spinbox.pack(side=tk.LEFT)
custom_layout_frame.grid_remove() # Hide it by default

# --- Fit Mode ---
fit_mode_label = ttk.Label(options_frame, text="Modo de Ajuste:")
fit_mode_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
fit_mode_var = tk.StringVar(value="Ajustar")
fit_mode_frame = ttk.Frame(options_frame)
fit_mode_frame.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
fit_radio_fit = ttk.Radiobutton(fit_mode_frame, text="Ajustar (con bordes)", variable=fit_mode_var, value="Ajustar", command=update_preview)
fit_radio_fill = ttk.Radiobutton(fit_mode_frame, text="Rellenar (recorte)", variable=fit_mode_var, value="Rellenar", command=update_preview)
fit_radio_fit.pack(side=tk.LEFT, expand=True)
fit_radio_fill.pack(side=tk.LEFT, expand=True)

# --- Orientation ---
orientation_label = ttk.Label(options_frame, text="Orientación:")
orientation_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
orientation_var = tk.StringVar(value="Vertical")
orientation_frame = ttk.Frame(options_frame)
orientation_frame.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
orientation_radio_v = ttk.Radiobutton(orientation_frame, text="Vertical", variable=orientation_var, value="Vertical", command=update_preview)
orientation_radio_h = ttk.Radiobutton(orientation_frame, text="Horizontal", variable=orientation_var, value="Horizontal", command=update_preview)
orientation_radio_v.pack(side=tk.LEFT, expand=True)
orientation_radio_h.pack(side=tk.LEFT, expand=True)

options_frame.columnconfigure(1, weight=1)

action_frame = ttk.LabelFrame(controls_panel, text="Acciones", padding=(10, 5))
action_frame.pack(fill=tk.X, pady=5)

load_button = ttk.Button(action_frame, text="Cargar Imágenes", command=upload_files_replace)
load_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

add_button = ttk.Button(action_frame, text="Añadir Imágenes", command=upload_files_add)
add_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

preview_button = ttk.Button(action_frame, text="Actualizar Previsualización", command=update_preview, state=tk.DISABLED)
preview_button.pack(pady=5, fill=tk.X)

generate_button = ttk.Button(action_frame, text="Generar PDF", command=generate_pdf, state=tk.DISABLED)
generate_button.pack(pady=5, fill=tk.X)

root.mainloop()
