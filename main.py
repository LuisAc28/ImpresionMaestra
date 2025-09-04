import tkinter as tk
from tkinter import filedialog


def upload_files():
    """
    Opens a file dialog to select multiple image files (JPG, PNG)
    and prints their paths.
    """
    file_paths = filedialog.askopenfilenames(
        title="Seleccionar imágenes",
        filetypes=(
            ("Archivos de imagen", "*.jpg *.jpeg *.png"),
            ("Todos los archivos", "*.*")
        )
    )
    if file_paths:
        print("Archivos seleccionados:")
        for path in file_paths:
            print(path)


# Create the main window
root = tk.Tk()
root.title("Impresión Maestra - v1.0")
root.geometry("800x600")

# Add a button
load_button = tk.Button(root, text="Cargar Imágenes", command=upload_files)
load_button.pack()

# Start the main loop
root.mainloop()
