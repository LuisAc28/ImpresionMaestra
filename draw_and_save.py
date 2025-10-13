from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
import os

# These constants were moved from the main file or are specific to this module.
FIT_MODE_FIT = "fit"
FIT_MODE_FILL = "fill"
FIT_MODE_DEFORM = "deform"

def draw_and_save_pages_as_images(pages, save_path, layout_choice, page_size_pt, grid_dims, fit_mode, smart_crop_func, crop_resize_func):
    """
    Creates an A4 canvas in memory, draws the layout for each page onto it,
    and saves each page as a separate PNG file.
    This function is now decoupled and receives all necessary context as arguments.
    """
    DPI = 300
    # Use the provided page size to calculate pixel dimensions
    A4_WIDTH_PX = int((page_size_pt[0] / 72) * DPI)
    A4_HEIGHT_PX = int((page_size_pt[1] / 72) * DPI)
    POINTS_PER_INCH = 72
    SCALE = DPI / POINTS_PER_INCH

    # Handle file path for multiple pages
    base_path, ext = os.path.splitext(save_path)
    if not ext or ext.lower() != ".png": ext = ".png"

    total_pages = len(pages)

    for i, page_data in enumerate(pages):
        # Create a blank white canvas for the page
        canvas_img = Image.new('RGBA', (A4_WIDTH_PX, A4_HEIGHT_PX), (255, 255, 255, 255))

        page_width_pt, page_height_pt = page_size_pt
        margin_pt = 1 * cm
        margin_px = int(margin_pt * SCALE)

        if layout_choice == "Mosaico (Ahorro de papel)":
            for rect in page_data:
                img = rect.rid['image']
                original_w = rect.rid['original_w']

                # Handle rotation
                if abs(rect.width - original_w) > 1e-6:
                    img_to_draw = img.rotate(90, expand=True)
                else:
                    img_to_draw = img

                # Calculate dimensions and position in pixels
                px_w = int(rect.width * SCALE)
                px_h = int(rect.height * SCALE)

                # Resize image to target pixel dimensions
                resized_img = img_to_draw.resize((px_w, px_h), Image.Resampling.LANCZOS)

                # Calculate paste position, inverting Y-axis from bottom-left to top-left
                paste_x = margin_px + int(rect.x * SCALE)
                paste_y = A4_HEIGHT_PX - (margin_px + int(rect.y * SCALE) + px_h)

                # Paste onto canvas, using the image's alpha channel as a mask for transparency
                canvas_img.paste(resized_img, (paste_x, paste_y), resized_img)

        else: # Grid layouts
            rows, cols = grid_dims

            drawable_width_pt = page_width_pt - 2 * margin_pt
            drawable_height_pt = page_height_pt - 2 * margin_pt

            cell_width_pt = drawable_width_pt / cols
            cell_height_pt = drawable_height_pt / rows

            cell_width_px = int(cell_width_pt * SCALE)
            cell_height_px = int(cell_height_pt * SCALE)

            for j, item in enumerate(page_data):
                img = item['image']
                rotation = item['rotation']
                row, col = divmod(j, cols)

                if rotation != 0:
                    img = img.rotate(rotation, expand=True)

                # Determine the top-left corner of the cell in pixels
                cell_paste_x = margin_px + (col * cell_width_px)
                cell_paste_y = margin_px + (row * cell_height_px)

                if fit_mode == FIT_MODE_FIT:
                    img_copy = img.copy()
                    img_copy.thumbnail((cell_width_px, cell_height_px), Image.Resampling.LANCZOS)
                    img_w, img_h = img_copy.size
                    # Center the image within the cell
                    paste_x = cell_paste_x + (cell_width_px - img_w) // 2
                    paste_y = cell_paste_y + (cell_height_px - img_h) // 2
                    canvas_img.paste(img_copy, (paste_x, paste_y), img_copy)

                elif fit_mode == FIT_MODE_FILL:
                    if item.get('custom_crop_box'):
                        box = item['custom_crop_box']
                    else:
                        # Use the passed-in smart crop function
                        box = smart_crop_func(img, cell_width_px / cell_height_px)

                    # Use the passed-in crop and resize function
                    cropped_img = crop_resize_func(img, box, cell_width_px, cell_height_px)
                    canvas_img.paste(cropped_img, (cell_paste_x, cell_paste_y), cropped_img)

                elif fit_mode == FIT_MODE_DEFORM:
                    deformed_img = img.resize((cell_width_px, cell_height_px), Image.Resampling.LANCZOS)
                    canvas_img.paste(deformed_img, (cell_paste_x, cell_paste_y), deformed_img)

        # Determine the final save path for this page
        if total_pages > 1:
            final_save_path = f"{base_path}_{i+1}{ext}"
        else:
            # If the user typed "name.png", don't make it "name.png.png"
            final_save_path = save_path if save_path.lower().endswith('.png') else f"{base_path}{ext}"

        canvas_img.save(final_save_path, 'PNG')
