import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.colors import HexColor
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageOps, ImageTk
import rectpack
import os
import cv2
import numpy as np
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Global variables
loaded_images_data = [] # List of (Image, path) tuples
paper_dims = {} # To store on-screen paper dimensions
orientation_var = None # Will be initialized with UI
face_cascade = None # To cache the loaded Haar Cascade classifier
FIT_MODE_FIT = "fit"
FIT_MODE_FILL = "fill"
FIT_MODE_DEFORM = "deform"

MOSAIC_UNIFY_NONE = "none"
MOSAIC_UNIFY_WIDTH = "width"
MOSAIC_UNIFY_HEIGHT = "height"

preview_pages = [] # To store the layout data for all pages
current_preview_page_index = 0 # To track the current page in the preview

def get_page_size():
    """
    Returns the page dimensions (A4 or landscape A4) based on the UI selection.
    """
    if orientation_var and orientation_var.get() == "Horizontal":
        return landscape(A4)
    return A4

def _handle_file_selection(replace_current: bool):
    """
    Internal logic for opening file dialog, processing images, and updating the image list.
    """
    global loaded_images_data
    title = "Seleccionar imágenes para cargar" if replace_current else "Seleccionar imágenes para añadir"
    file_paths_tuple = filedialog.askopenfilenames(
        title=title,
        filetypes=(("Archivos de imagen", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*"))
    )

    newly_processed_images = []
    if file_paths_tuple:
        for path in file_paths_tuple:
            try:
                with Image.open(path) as img:
                    # Convert to RGBA to handle transparency consistently
                    img = img.convert("RGBA")
                    trimmed_img = trim_whitespace(img)
                    newly_processed_images.append((trimmed_img, path))
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                # Optionally show a message to the user
                messagebox.showwarning("Error de Imagen", f"No se pudo procesar la imagen:\n{os.path.basename(path)}\n\nSerá omitida.")

    if replace_current:
        loaded_images_data = newly_processed_images
    else:
        loaded_images_data.extend(newly_processed_images)

    if loaded_images_data:
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
def load_resources():
    """
    Loads resources like the Haar Cascade classifier at startup.
    """
    global face_cascade
    cascade_path = resource_path('haarcascade_frontalface_default.xml')

    if not os.path.exists(cascade_path):
        messagebox.showwarning(
            "Recurso Opcional No Encontrado",
            f"No se encontró el archivo 'haarcascade_frontalface_default.xml'.\n\n"
            "La función de 'Rellenar' con detección de caras estará desactivada. "
            "Para activarla, coloque el archivo en la misma carpeta que el ejecutable."
        )
        face_cascade = None
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        messagebox.showwarning(
            "Error de Recurso",
            f"No se pudo cargar el clasificador de caras desde:\n{os.path.basename(cascade_path)}\n\n"
            "La función de 'Rellenar' con detección de caras estará desactivada."
        )
        face_cascade = None

def get_face_cascade():
    """Simple accessor for the globally loaded cascade."""
    return face_cascade

def trim_whitespace(image):
    """
    Detects and crops empty borders (white or transparent) from a Pillow image.
    """
    # Convert Pillow Image to NumPy array
    img_np = np.array(image)
    if image.mode == 'RGBA':
        # Use the alpha channel to find the bounding box
        alpha_channel = img_np[:, :, 3]
        coords = np.argwhere(alpha_channel > 0)
    else:
        # For non-alpha images, assume a white background
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # Threshold to find non-white pixels (adjust threshold for off-white)
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        coords = np.argwhere(thresh > 0)

    if coords.size == 0:
        # Image is completely empty, return it as is
        return image

    # Get bounding box from coordinates
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the original Pillow image
    return image.crop((x_min, y_min, x_max + 1, y_max + 1))

def smart_crop(img, target_w, target_h):
    """
    Crops and resizes an image to fill the target dimensions,
    centering on detected faces or the geometric center.
    """
    face_cascade = get_face_cascade()
    if face_cascade is None:
        return ImageOps.fit(img, (target_w, target_h), method=Image.Resampling.LANCZOS)

    # --- Optimization: Detect on a smaller image ---
    original_w, original_h = img.size
    detection_width = 400  # pixels

    if original_w > detection_width:
        scale_ratio = original_w / detection_width
        detection_height = int(original_h / scale_ratio)
        detect_img = img.resize((detection_width, detection_height), Image.Resampling.LANCZOS)
    else:
        scale_ratio = 1.0
        detect_img = img

    img_np = np.array(detect_img.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        # Scale face coordinates back to the original image size
        scaled_faces = [(int(x * scale_ratio), int(y * scale_ratio), int(w * scale_ratio), int(h * scale_ratio)) for x, y, w, h in faces]
        x_coords = [x + w / 2 for x, y, w, h in scaled_faces]
        y_coords = [y + h / 2 for x, y, w, h in scaled_faces]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
    else:
        center_x = original_w / 2
        center_y = original_h / 2

    # 2. Re-implement ImageOps.fit logic with the smart center
    source_w, source_h = img.size
    target_ar = target_w / target_h
    source_ar = source_w / source_h

    if source_ar > target_ar: # Source is wider than target, crop sides
        crop_w = source_h * target_ar
        crop_h = source_h
        left = max(0, center_x - crop_w / 2)
        top = 0
        right = min(source_w, left + crop_w)
        if right == source_w: # Adjust left if we hit the right edge
            left = source_w - crop_w
        bottom = crop_h
    else: # Source is taller than target, crop top/bottom
        crop_w = source_w
        crop_h = source_w / target_ar
        left = 0
        top = max(0, center_y - crop_h / 2)
        right = crop_w
        bottom = min(source_h, top + crop_h)
        if bottom == source_h: # Adjust top if we hit the bottom edge
            top = source_h - crop_h

    box = (left, top, right, bottom)
    cropped_img = img.crop(box)
    return cropped_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

def get_grid_dimensions():
    """
    Determines the grid dimensions (rows, cols) based on the layout selection.
    This is the single source of truth for grid sizing.
    """
    layout_choice = layout_var.get()

    if layout_choice == "Personalizado...":
        try:
            rows = int(rows_var.get())
            cols = int(cols_var.get())
            return (rows if rows > 0 else 1, cols if cols > 0 else 1)
        except (ValueError, tk.TclError):
            return (1, 1)
    else:
        layout_configs = {
            "1 por hoja": (1, 1), "2 por hoja": (1, 2), "3 por hoja (1x3)": (1, 3),
            "4 por hoja (2x2)": (2, 2), "6 por hoja (2x3)": (2, 3), "9 por hoja (3x3)": (3, 3)
        }
        return layout_configs.get(layout_choice, (1, 1))

def handle_layout_change(event=None):
    """
    Shows or hides the custom/mosaic option frames based on the combobox selection
    and triggers a preview update.
    """
    selected_layout = layout_var.get()

    if selected_layout == "Personalizado...":
        custom_layout_frame.grid()
    else:
        custom_layout_frame.grid_remove()

    if selected_layout == "Mosaico (Ahorro de papel)":
        mosaic_options_frame.grid()
        mosaic_unify_frame.grid()
    else:
        mosaic_options_frame.grid_remove()
        mosaic_unify_frame.grid_remove()

    update_preview()

def _run_mosaic_packing(images_data_list, scale_percentage):
    """
    Internal helper to run the mosaic packing algorithm with a given scale.
    Returns the list of pages and a list of unpacked image paths.
    """
    if not images_data_list:
        return [], []

    margin = 1 * cm
    page_width, page_height = get_page_size()
    bin_width = page_width - 2 * margin
    bin_height = page_height - 2 * margin

    unify_mode = mosaic_unify_var.get()
    initial_rects = []

    if unify_mode == MOSAIC_UNIFY_NONE:
        max_dim = 0
        for img, path in images_data_list:
            max_dim = max(max_dim, img.width, img.height)
        target_size = bin_width * (scale_percentage / 100.0)
        scale_factor = target_size / max_dim if max_dim > 0 else 0
        initial_rects = [{'width': img.width * scale_factor, 'height': img.height * scale_factor, 'rid': (img, path)}
                         for img, path in images_data_list]

    elif unify_mode == MOSAIC_UNIFY_WIDTH:
        target_width = bin_width * (scale_percentage / 100.0)
        for img, path in images_data_list:
            original_w, original_h = img.size
            aspect_ratio = original_h / original_w if original_w > 0 else 0
            new_w = target_width
            new_h = new_w * aspect_ratio
            initial_rects.append({'width': new_w, 'height': new_h, 'rid': (img, path)})

    elif unify_mode == MOSAIC_UNIFY_HEIGHT:
        target_height = bin_height * (scale_percentage / 100.0)
        for img, path in images_data_list:
            original_w, original_h = img.size
            aspect_ratio = original_w / original_h if original_h > 0 else 0
            new_h = target_height
            new_w = new_h * aspect_ratio
            initial_rects.append({'width': new_w, 'height': new_h, 'rid': (img, path)})

    final_rects = []
    for r in initial_rects:
        w, h = r['width'], r['height']
        can_fit_as_is = (w <= bin_width and h <= bin_height)
        can_fit_rotated = (h <= bin_width and w <= bin_height)
        if not can_fit_as_is and not can_fit_rotated:
            ratio_as_is = min(bin_width / w if w > 0 else 0, bin_height / h if h > 0 else 0)
            ratio_rotated = min(bin_width / h if h > 0 else 0, bin_height / w if w > 0 else 0)
            downscale_ratio = max(ratio_as_is, ratio_rotated)
            w *= downscale_ratio
            h *= downscale_ratio
        rid_data = {'image': r['rid'][0], 'path': r['rid'][1], 'original_w': w, 'original_h': h}
        final_rects.append({'width': w, 'height': h, 'rid': rid_data})

    packer = rectpack.newPacker(pack_algo=rectpack.MaxRectsBl, sort_algo=rectpack.SORT_AREA, rotation=True)
    for r in final_rects:
        packer.add_rect(r['width'], r['height'], rid=r['rid'])
    packer.add_bin(bin_width, bin_height, count=float('inf'))
    packer.pack()

    list_of_pages = [bin for bin in packer if bin]
    all_rids = {r['rid']['path'] for r in final_rects}
    packed_rids = {rect.rid['path'] for abin in list_of_pages for rect in abin}
    unpacked_paths = list(all_rids - packed_rids)

    return list_of_pages, unpacked_paths

def calculate_mosaic_layout(images_data_list):
    """
    Wrapper that gets the current scale from the UI and runs the packing algorithm.
    """
    scale_percentage = mosaic_scale_var.get()
    return _run_mosaic_packing(images_data_list, scale_percentage)

def auto_fit_mosaic():
    """
    Finds the optimal scale to fit all images on one page and updates the UI.
    """
    if not loaded_images_data:
        messagebox.showinfo("Información", "No hay imágenes cargadas para ajustar.")
        return

    # Search for the best fit by trying scales from largest to smallest
    for p in range(150, 4, -5): # Iterate from 150% down to 5% in steps of 5
        pages, unpacked = _run_mosaic_packing(loaded_images_data, p)
        if len(pages) <= 1 and not unpacked:
            # Found the largest scale that fits on one page
            mosaic_scale_var.set(p)
            update_preview()
            return

    # If no scale worked (e.g., one image is too big even at min scale)
    mosaic_scale_var.set(5) # Default to minimum
    update_preview()
    messagebox.showinfo("Ajuste Automático", "No se pudo encajar todas las imágenes en una sola página. Se ha usado el tamaño mínimo posible.")

def calculate_grid_layout(images_data_list):
    rows, cols = get_grid_dimensions()
    if rows * cols == 0: return []
    chunk_size = rows * cols
    pages = [images_data_list[i:i + chunk_size] for i in range(0, len(images_data_list), chunk_size)]
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
        # For mosaic, the Y-coordinate needs to be inverted from bottom-left (rectpack)
        # to top-left (tkinter), and the reference height might not be the full page
        # on the last page.

        # 1. Find the actual height of the content on this specific page
        page_content_height_pt = 0
        if page_data:
            page_content_height_pt = max(r.y + r.height for r in page_data)

        reference_height_px = page_content_height_pt * scale
        margin_px = margin_pt * scale

        for rect in page_data:
            # 2. Calculate final coordinates using the dynamic content height
            px = x0 + margin_px + (rect.x * scale)

            # The top of the rect in a bottom-up system is (rect.y + rect.height)
            # We subtract this from the total content height to get the top-down position.
            py = y0 + margin_px + reference_height_px - ((rect.y + rect.height) * scale)

            pw = rect.width * scale
            ph = rect.height * scale
            try:
                img = rect.rid['image']
                original_w = rect.rid['original_w']

                if abs(rect.width - original_w) > 1e-6: # Compare floats with tolerance
                    img_to_draw = img.rotate(90, expand=True)
                else:
                    img_to_draw = img

                resized_img = img_to_draw.resize((int(pw), int(ph)), Image.Resampling.LANCZOS)
                photo_img = ImageTk.PhotoImage(resized_img)
                preview_canvas.thumbnail_references.append(photo_img)
                preview_canvas.create_image(px, py, image=photo_img, anchor="nw", tags="layout_item")
            except Exception as e:
                print(f"Error drawing mosaic preview for rect: {rect.rid}. Exception: {e}")
                preview_canvas.create_rectangle(px, py, px + pw, py + ph, outline="red", fill="pink", tags="layout_item")
                preview_canvas.create_text(px + 4, py + 4, text=f"Error:\n{os.path.basename(rect.rid['path'])}", anchor="nw", font=("Arial", 7), fill="red", tags="layout_item")
    else: # Grid layouts
        rows, cols = get_grid_dimensions()
        if rows * cols == 0: return

        # Corrected logic: Calculate based on the on-screen paper dimensions
        margin_px = margin_pt * scale
        drawable_w = paper_w_px - (2 * margin_px)
        drawable_h = paper_h_px - (2 * margin_px)
        cell_w_px = drawable_w / cols
        cell_h_px = drawable_h / rows

        for i, (img, path) in enumerate(page_data):
            row, col = divmod(i, cols)

            # Cell position relative to the paper, not the whole canvas
            px = x0 + margin_px + (col * cell_w_px)
            py = y0 + margin_px + (row * cell_h_px)
            pw = cell_w_px
            ph = cell_h_px

            try:
                img_copy = img.copy()
                if fit_mode_var.get() == FIT_MODE_FIT:
                    img_copy.thumbnail((int(pw), int(ph)), Image.Resampling.LANCZOS)
                    img_w, img_h = img_copy.size
                    px_centered = px + (pw - img_w) / 2
                    py_centered = py + (ph - img_h) / 2
                    photo_img = ImageTk.PhotoImage(img_copy)
                elif fit_mode_var.get() == FIT_MODE_DEFORM:
                    img_copy = img_copy.resize((int(pw), int(ph)), Image.Resampling.LANCZOS)
                    px_centered = px
                    py_centered = py
                    photo_img = ImageTk.PhotoImage(img_copy)
                else: # FIT_MODE_FILL
                    img_copy = smart_crop(img_copy, int(pw), int(ph))
                    px_centered = px
                    py_centered = py
                    photo_img = ImageTk.PhotoImage(img_copy)

                preview_canvas.thumbnail_references.append(photo_img)
                preview_canvas.create_image(px_centered, py_centered, image=photo_img, anchor="nw", tags="layout_item")

                if fit_mode_var.get() == FIT_MODE_FIT:
                    try:
                        border_width = int(border_width_var.get())
                        if border_width > 0:
                            border_color = border_color_var.get()
                            # Draw border around the calculated cell, not the paper
                            preview_canvas.create_rectangle(
                                px, py, px + pw, py + ph,
                                outline=border_color, width=border_width, tags="layout_item"
                            )
                    except (ValueError, tk.TclError):
                        pass
            except Exception as e:
                # Fallback to drawing a simple error box in the cell
                preview_canvas.create_rectangle(px, py, px + pw, py + ph, outline="red", fill="pink", tags="layout_item")
                preview_canvas.create_text(px + 4, py + 4, text=f"Error:\n{os.path.basename(path)}", anchor="nw", font=("Arial", 7), fill="red", tags="layout_item")

def update_preview():
    """
    Calculates the full layout and redraws the entire preview canvas.
    This is the main function for refreshing the preview pane.
    """
    # Pre-emptive check for face cascade if fill mode is selected
    if fit_mode_var.get() == FIT_MODE_FILL and get_face_cascade() is None:
        return # Abort if the required file is missing

    global preview_pages, current_preview_page_index, paper_dims

    # 0. Save current state
    saved_page_index = current_preview_page_index

    # 1. Recalculate paper dimensions and redraw paper background
    canvas_w = preview_canvas.winfo_width()
    canvas_h = preview_canvas.winfo_height()
    if canvas_w <= 1 or canvas_h <= 1:
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
    if loaded_images_data:
        layout_choice = layout_var.get()
        if layout_choice == "Mosaico (Ahorro de papel)":
            try:
                # The function now returns a list of pages directly
                preview_pages, unpacked_paths = calculate_mosaic_layout(loaded_images_data)
                if unpacked_paths:
                    msg = ("Las siguientes imágenes son demasiado grandes para caber en la página "
                           "con el tamaño de mosaico seleccionado y serán omitidas:\n\n" +
                           "\n".join(f"- {os.path.basename(p)}" for p in unpacked_paths))
                    messagebox.showwarning("Imágenes Grandes Omitidas", msg)
            except Exception as e:
                messagebox.showerror("Error de Cálculo", f"No se pudo calcular el diseño de mosaico:\n{e}")
                preview_pages = []
        else:
            preview_pages = calculate_grid_layout(loaded_images_data)

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
    update_thumbnails_panel()

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

def update_thumbnails_panel():
    """Clears and repopulates the thumbnail list in the left panel."""
    # Clear existing thumbnails
    for widget in thumbnails_inner_frame.winfo_children():
        widget.destroy()

    # Keep a reference to the PhotoImage objects to prevent garbage collection
    thumb_canvas.thumb_references = []

    for i, (img, path) in enumerate(loaded_images_data):
        thumb_frame = ttk.Frame(thumbnails_inner_frame, padding=5)
        thumb_frame.pack(fill=tk.X, expand=True)

        img_copy = img.copy()
        img_copy.thumbnail((80, 80), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(img_copy)
        thumb_canvas.thumb_references.append(photo_img)

        thumb_label = ttk.Label(thumb_frame, image=photo_img)
        thumb_label.pack(side=tk.LEFT, padx=(0, 5))

        info_label = ttk.Label(thumb_frame, text=f"{i+1}. {os.path.basename(path)}", wraplength=100, justify=tk.LEFT)
        info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

def update_pagination_controls():
    """Updates the state of the pagination buttons and page counter label."""
    total_pages = len(preview_pages)

    # Update label
    page_status_label.config(text=f"Página {current_preview_page_index + 1} de {total_pages}" if total_pages > 0 else "Página 0 de 0")

    # Update button states
    prev_page_button.config(state=tk.NORMAL if current_preview_page_index > 0 else tk.DISABLED)
    next_page_button.config(state=tk.NORMAL if current_preview_page_index < total_pages - 1 else tk.DISABLED)

def generate_pdf():
    if not loaded_images_data:
        messagebox.showinfo("Información", "No hay imágenes seleccionadas.")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("Archivos PDF", "*.pdf"), ("Todos los archivos", "*.*")], title="Guardar PDF como...")
    if not save_path:
        return
    try:
        layout_choice = layout_var.get()
        if layout_choice == "Mosaico (Ahorro de papel)":
            list_of_pages, unpacked_paths = calculate_mosaic_layout(loaded_images_data)
            if unpacked_paths:
                msg = "Imágenes omitidas por ser demasiado grandes:\n\n" + "\n".join(f"- {os.path.basename(p)}" for p in unpacked_paths)
                messagebox.showwarning("Imágenes Grandes Omitidas", msg)
            if not list_of_pages:
                messagebox.showerror("Error", "No se pudo empaquetar ninguna imagen.")
                return
            draw_mosaic_pdf(list_of_pages, save_path)
        else: # Grid modes
            fit_mode = fit_mode_var.get()
            if fit_mode == FIT_MODE_FILL and get_face_cascade() is None:
                return # Abort if the required file is missing
            pages = calculate_grid_layout(loaded_images_data)
            border_width = int(border_width_var.get())
            border_color = border_color_var.get()
            draw_grid_pdf(pages, fit_mode, save_path, border_width, border_color)
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error inesperado:\n{e}")

def draw_mosaic_pdf(pages, save_path):
    margin = 1 * cm
    c = canvas.Canvas(save_path, pagesize=get_page_size())
    for i, page in enumerate(pages):
        if i > 0: c.showPage()
        for rect in page:
            img = rect.rid['image']
            original_w = rect.rid['original_w']

            # Infer rotation
            if rect.width == original_w:
                img_to_draw = img
            else:
                img_to_draw = img.rotate(90, expand=True)

            page_height = get_page_size()[1]
            x = margin + rect.x
            # Invert y-coordinate for PDF's bottom-left origin
            y = page_height - (margin + rect.y + rect.height)
            c.drawImage(ImageReader(img_to_draw), x, y, width=rect.width, height=rect.height)
    c.save()
    messagebox.showinfo("Éxito", f"PDF en modo Mosaico guardado en:\n{save_path}")

def draw_grid_pdf(pages, fit_mode, save_path, border_width, border_color):
    rows, cols = get_grid_dimensions()
    if rows * cols == 0:
        messagebox.showerror("Error de Diseño", "Las filas y columnas deben ser mayores que cero.")
        return
    pagesize = get_page_size()
    c = canvas.Canvas(save_path, pagesize=pagesize)
    width, height = pagesize
    margin = 1 * cm
    cell_width = (width - 2 * margin) / cols
    cell_height = (height - 2 * margin) / rows
    for i, page_chunk in enumerate(pages):
        if i > 0: c.showPage()
        positions = [(margin + col * cell_width, margin + (rows - 1 - row) * cell_height) for row in range(rows) for col in range(cols)]
        for j, (img, path) in enumerate(page_chunk):
            pos_x, pos_y = positions[j]

            if fit_mode == FIT_MODE_FIT:
                # Draw border around the cell first
                if border_width > 0:
                    c.setStrokeColor(HexColor(border_color))
                    c.setLineWidth(border_width)
                    c.rect(pos_x, pos_y, cell_width, cell_height, stroke=1, fill=0)

                # Manually calculate the centered position for the image
                img_w, img_h = img.width, img.height
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

                # Then draw the image on top, centered
                c.drawImage(ImageReader(img), final_x, final_y, width=final_w, height=final_h)

            elif fit_mode == FIT_MODE_FILL:
                cropped_img = smart_crop(img, int(cell_width), int(cell_height))
                c.drawImage(ImageReader(cropped_img), pos_x, pos_y, width=cell_width, height=cell_height)
            elif fit_mode == FIT_MODE_DEFORM:
                deformed_img = img.resize((int(cell_width), int(cell_height)), Image.Resampling.LANCZOS)
                c.drawImage(ImageReader(deformed_img), pos_x, pos_y, width=cell_width, height=cell_height)
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
root.title("Impresión Maestra - v2.2")
root.geometry("1024x768")

# Main layout with 3 resizable panels
main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Left Panel: Thumbnails
thumbnails_panel = ttk.LabelFrame(main_paned_window, text="Imágenes Cargadas")
main_paned_window.add(thumbnails_panel, weight=0)

# Center Panel: Controls
controls_panel = ttk.Frame(main_paned_window)
main_paned_window.add(controls_panel, weight=0)

# Right Panel: Preview
preview_panel = ttk.LabelFrame(main_paned_window, text="Previsualización del Diseño")
main_paned_window.add(preview_panel, weight=1)

# --- Children of the panels ---
# Children of controls_panel (no change, they are already linked to this variable)
controls_panel.pack_propagate(False) # Keep this for the controls panel

# Children of preview_panel (no change, they are already linked to this variable)

# --- Thumbnail Panel UI ---
thumb_canvas = tk.Canvas(thumbnails_panel)
thumb_scrollbar = ttk.Scrollbar(thumbnails_panel, orient="vertical", command=thumb_canvas.yview)
thumbnails_inner_frame = ttk.Frame(thumb_canvas)

thumbnails_inner_frame.bind(
    "<Configure>",
    lambda e: thumb_canvas.configure(
        scrollregion=thumb_canvas.bbox("all")
    )
)

thumb_canvas.create_window((0, 0), window=thumbnails_inner_frame, anchor="nw")
thumb_canvas.configure(yscrollcommand=thumb_scrollbar.set)

thumb_canvas.pack(side="left", fill="both", expand=True)
thumb_scrollbar.pack(side="right", fill="y")

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
    page_w_pt, page_h_pt = get_page_size()
    ar = page_w_pt / page_h_pt
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

# --- Layout Selection (row 0) ---
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

# --- Custom Layout Frame (row 1, initially hidden) ---
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
custom_layout_frame.grid_remove()

# --- Mosaic Options Frame (row 2, initially hidden) ---
mosaic_options_frame = ttk.Frame(options_frame)
mosaic_options_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=0, sticky="ew")
mosaic_scale_label = ttk.Label(mosaic_options_frame, text="Tamaño de Mosaico:")
mosaic_scale_label.pack(side=tk.LEFT, padx=(5, 5))
mosaic_scale_var = tk.DoubleVar(value=100)
mosaic_scale_slider = ttk.Scale(mosaic_options_frame, from_=5, to=150, variable=mosaic_scale_var, orient=tk.HORIZONTAL, command=lambda e: update_preview())
mosaic_scale_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
auto_fit_button = ttk.Button(mosaic_options_frame, text="Ajuste Automático", command=auto_fit_mosaic)
auto_fit_button.pack(side=tk.LEFT, padx=(5, 0))
mosaic_options_frame.grid_remove()

# --- Mosaic Unify Frame (row 3, initially hidden) ---
mosaic_unify_frame = ttk.LabelFrame(options_frame, text="Unificar Dimensiones de Mosaico")
mosaic_unify_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
mosaic_unify_var = tk.StringVar(value=MOSAIC_UNIFY_NONE)
unify_none_radio = ttk.Radiobutton(mosaic_unify_frame, text="Ninguno", variable=mosaic_unify_var, value=MOSAIC_UNIFY_NONE, command=update_preview)
unify_width_radio = ttk.Radiobutton(mosaic_unify_frame, text="Unificar Anchos", variable=mosaic_unify_var, value=MOSAIC_UNIFY_WIDTH, command=update_preview)
unify_height_radio = ttk.Radiobutton(mosaic_unify_frame, text="Unificar Alturas", variable=mosaic_unify_var, value=MOSAIC_UNIFY_HEIGHT, command=update_preview)
unify_none_radio.pack(side=tk.LEFT, expand=True, padx=5)
unify_width_radio.pack(side=tk.LEFT, expand=True, padx=5)
unify_height_radio.pack(side=tk.LEFT, expand=True, padx=5)
mosaic_unify_frame.grid_remove()

# --- Fit Mode (row 4) ---
fit_mode_label = ttk.Label(options_frame, text="Modo de Ajuste:")
fit_mode_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
fit_mode_var = tk.StringVar(value=FIT_MODE_FIT)
fit_mode_frame = ttk.Frame(options_frame)
fit_mode_frame.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
fit_radio_fit = ttk.Radiobutton(fit_mode_frame, text="Ajustar", variable=fit_mode_var, value=FIT_MODE_FIT, command=handle_fit_mode_change)
fit_radio_fill = ttk.Radiobutton(fit_mode_frame, text="Rellenar", variable=fit_mode_var, value=FIT_MODE_FILL, command=handle_fit_mode_change)
fit_radio_deform = ttk.Radiobutton(fit_mode_frame, text="Deformar", variable=fit_mode_var, value=FIT_MODE_DEFORM, command=handle_fit_mode_change)
fit_radio_fit.pack(side=tk.LEFT, expand=True)
fit_radio_fill.pack(side=tk.LEFT, expand=True)
fit_radio_deform.pack(side=tk.LEFT, expand=True)

# --- Border Options Frame (row 5, initially hidden) ---
border_options_frame = ttk.Frame(options_frame)
border_options_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=0, sticky="ew")
border_width_label = ttk.Label(border_options_frame, text="Grosor del Borde:")
border_width_label.pack(side=tk.LEFT, padx=(5, 5))
border_width_var = tk.StringVar(value="1")
border_width_spinbox = tk.Spinbox(border_options_frame, from_=0, to=10, width=5, textvariable=border_width_var, command=update_preview)
border_width_spinbox.pack(side=tk.LEFT, padx=(0, 10))
border_color_var = tk.StringVar(value="#000000")
border_color_button = ttk.Button(border_options_frame, text="Color del Borde", command=choose_border_color)
border_color_button.pack(side=tk.LEFT, padx=(5,5))
border_options_frame.grid_remove()

# --- Orientation (row 6) ---
orientation_label = ttk.Label(options_frame, text="Orientación:")
orientation_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")
orientation_var = tk.StringVar(value="Vertical")
orientation_frame = ttk.Frame(options_frame)
orientation_frame.grid(row=6, column=1, padx=5, pady=5, sticky="ew")
orientation_radio_v = ttk.Radiobutton(orientation_frame, text="Vertical", variable=orientation_var, value="Vertical", command=update_preview)
orientation_radio_h = ttk.Radiobutton(orientation_frame, text="Horizontal", variable=orientation_var, value="Horizontal", command=update_preview)
orientation_radio_v.pack(side=tk.LEFT, expand=True)
orientation_radio_h.pack(side=tk.LEFT, expand=True)

options_frame.columnconfigure(1, weight=1)

# Set initial UI state
handle_fit_mode_change()
handle_layout_change()

# Load resources at startup
load_resources()

# Set initial sash positions for the PanedWindow
root.update_idletasks()
main_paned_window.sashpos(0, 120)
main_paned_window.sashpos(1, 400)

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
