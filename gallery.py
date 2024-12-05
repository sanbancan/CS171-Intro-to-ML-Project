import os 
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
from page import Page

class GalleryPage(Page):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.images = []
        title_label = tk.Label(self, text="Gallery", 
                               font=("Tw Cen MT", 40), 
                               fg='#3a7ebf', bg='#cfcfcf')
        title_label.pack(pady=(20, 0))
        
        self.image_frame = tk.Frame(self)
        self.image_frame.pack(pady=(10, 10))

        self.description_label = tk.Label(self, text="Images uploaded for analysis will be stored here.", 
                                           font=("Tw Cen MT", 15), bg='#cfcfcf', anchor='center')
        self.description_label.pack(pady=(10, 10))

    def add_image(self, image_path):
        self.images.append(image_path)
        self.update_gallery()

    def load_images_from_folder(self, folder_path):
        self.images.clear()
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')): 
                self.add_image(os.path.join(folder_path, filename))

    def update_gallery(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        for image_path in self.images:
            try:
                img = Image.open(image_path)
                img = img.resize((100, 100), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                img_label = tk.Label(self.image_frame, image=img_tk)
                img_label.image = img_tk
                img_label.pack(side='left', padx=5, pady=5)
                img_label.bind("<Button-1>", lambda event, path=image_path: self.open_details_page(path))
            except Exception as e:
                print(f"Error displaying image {image_path}: {e}")

    def open_details_page(self, image_path):
        self.master.open_details_page(image_path)

