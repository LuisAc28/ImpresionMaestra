import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageOps, ImageTk
import rectpack
import os

# Global variables
image_paths = []
paper_dims = {} # To store on-screen paper dimensions
FIT_MODE_FIT = "fit"
FIT_MODE_FILL = "fill"
preview_pages = [] # To store the layout data for all pages
current_preview_page_index = 0 # To track the current page in the preview

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
    A4_w_pt, A4_h_pt = A4
    scale = paper_w_px / A4_w_pt
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
        layout_configs = {"1 por hoja": (1, 1), "2 por hoja": (1, 2), "4 por hoja (2x2)": (2, 2), "6 por hoja (2x3)": (2, 3)}
        rows, cols = layout_configs[layout_choice]
        cell_width_pt = (A4_w_pt - 2 * margin_pt) / cols
        cell_height_pt = (A4_h_pt - 2 * margin_pt) / rows
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

                # Draw border for 'Ajustar' mode
                if fit_mode_var.get() == FIT_MODE_FIT:
                    try:
                        border_width = int(border_width_var.get())
                        if border_width > 0:
                            border_color = border_color_var.get()
                            preview_canvas.create_rectangle(
                                px_centered, py_centered,
                                px_centered + img_w, py_centered + img_h,
                                outline=border_color, width=border_width, tags="layout_item"
                            )
                    except (ValueError, tk.TclError):
                        pass # Ignore errors from invalid border width during typing
            except Exception as e:
                preview_canvas.create_rectangle(px, py, px + pw, py + ph, outline="red", fill="pink", tags="layout_item")
                preview_canvas.create_text(px + 4, py + 4, text=f"Error:\n{os.path.basename(path)}", anchor="nw", font=("Arial", 7), fill="red", tags="layout_item")

def update_preview():
    """
    Calculates the full layout based on selected images and options,
    then displays the first page of the result.
    """
    global preview_pages, current_preview_page_index

    preview_pages = []
    current_preview_page_index = 0

    if not image_paths or not paper_dims:
        preview_canvas.delete("layout_item")
    else:
        layout_choice = layout_var.get()
        if layout_choice == "Mosaico (Ahorro de papel)":
            try:
                packer, _ = calculate_mosaic_layout(image_paths)
                # A packer is an iterable of bins (pages), convert to a list of non-empty pages
                preview_pages = [bin for bin in packer if bin]
            except Exception as e:
                messagebox.showerror("Error de Cálculo", f"No se pudo calcular el diseño de mosaico:\n{e}")
                preview_pages = []
        else:
            # This function already returns a list of pages (lists of image paths)
            preview_pages = calculate_grid_layout(image_paths, layout_choice)

    # Always draw the current page (even if it's empty) and update controls
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
            border_width = int(border_width_var.get())
            border_color = border_color_var.get()
            draw_grid_pdf(pages, layout_choice, fit_mode, save_path, border_width, border_color)
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

def draw_grid_pdf(pages, layout_choice, fit_mode, save_path, border_width, border_color):
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
            img = Image.open(img_path)

            if fit_mode == FIT_MODE_FIT:
                c.drawImage(ImageReader(img), pos_x, pos_y, width=cell_width, height=cell_height, preserveAspectRatio=True, anchor='c')
                if border_width > 0:
                    img_w, img_h = img.size
                    cell_ar = cell_width / cell_height
                    img_ar = img_w / img_h
                    if img_ar > cell_ar:
                        final_w = cell_width
                        final_h = cell_width / img_ar
                    else:
                        final_h = cell_height
                        final_w = cell_height * img_ar

                    final_x = pos_x + (cell_width - final_w) / 2
                    final_y = pos_y + (cell_height - final_h) / 2

                    c.setStrokeColor(HexColor(border_color))
                    c.setLineWidth(border_width)
                    c.rect(final_x, final_y, final_w, final_h, stroke=1, fill=0)

            elif fit_mode == FIT_MODE_FILL:
                cropped_img = ImageOps.fit(img, (int(cell_width), int(cell_height)), method=Image.Resampling.LANCZOS)
                c.drawImage(ImageReader(cropped_img), pos_x, pos_y, width=cell_width, height=cell_height)
    c.save()
    messagebox.showinfo("Éxito", f"PDF guardado exitosamente en:\n{save_path}")


def handle_fit_mode_change():
    """
    Shows or hides the border options based on the fit mode selection
    and triggers a preview update.
    """
    if fit_mode_var.get() == FIT_MODE_FIT:
        border_options_frame.grid()
    else:
        border_options_frame.grid_remove()
    update_preview()

def choose_border_color():
    """
    Opens a color chooser and updates the border color variable.
    """
    color_code = colorchooser.askcolor(title="Elegir color del borde")
    if color_code and color_code[1]:
        border_color_var.set(color_code[1])
        update_preview()


# --- UI Setup ---
root = tk.Tk()
root.title("Impresión Maestra - v1.6")
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
fit_mode_var = tk.StringVar(value=FIT_MODE_FIT)
fit_mode_frame = ttk.Frame(options_frame)
fit_mode_frame.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
fit_radio_fit = ttk.Radiobutton(fit_mode_frame, text="Ajustar (con bordes)", variable=fit_mode_var, value=FIT_MODE_FIT, command=handle_fit_mode_change)
fit_radio_fill = ttk.Radiobutton(fit_mode_frame, text="Rellenar (recorte)", variable=fit_mode_var, value=FIT_MODE_FILL, command=handle_fit_mode_change)
fit_radio_fit.pack(side=tk.LEFT, expand=True)
fit_radio_fill.pack(side=tk.LEFT, expand=True)

# --- Border Options Frame (Initially Hidden) ---
border_options_frame = ttk.Frame(options_frame)
border_options_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=0, sticky="ew")

border_width_label = ttk.Label(border_options_frame, text="Grosor del Borde:")
border_width_label.pack(side=tk.LEFT, padx=(5, 5))
border_width_var = tk.StringVar(value="1")
border_width_spinbox = tk.Spinbox(border_options_frame, from_=0, to=10, width=5, textvariable=border_width_var, command=update_preview)
border_width_spinbox.pack(side=tk.LEFT, padx=(0, 10))

border_color_var = tk.StringVar(value="#000000")
border_color_button = ttk.Button(border_options_frame, text="Color del Borde", command=choose_border_color)
border_color_button.pack(side=tk.LEFT, padx=(5,5))
border_options_frame.grid_remove() # Hide it by default

options_frame.columnconfigure(1, weight=1)

# Set initial UI state
handle_fit_mode_change()

action_frame = ttk.LabelFrame(controls_panel, text="Acciones", padding=(10, 5))
action_frame.pack(fill=tk.X, pady=5)

load_button = ttk.Button(action_frame, text="Cargar Imágenes", command=upload_files)
load_button.pack(side=tk.LEFT, padx=5, pady=5)

preview_button = ttk.Button(action_frame, text="Actualizar Previsualización", command=update_preview, state=tk.DISABLED)
preview_button.pack(side=tk.LEFT, padx=5, pady=5)

generate_button = ttk.Button(action_frame, text="Generar PDF", command=generate_pdf, state=tk.DISABLED)
generate_button.pack(side=tk.LEFT, padx=5, pady=5)

root.mainloop()
