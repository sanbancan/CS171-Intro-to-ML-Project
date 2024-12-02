import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
from page import Page

from PIL import Image, ImageTk
from page import Page

class DetailsPage(Page):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.master = master
        self.image_path = None
        self.result = None

        self.image_label = tk.Label(self)
        self.image_label.pack(side='left', padx=(20, 10), pady=20)

        self.details_frame = tk.Frame(self)
        self.details_frame.pack(side='right', padx=(10, 20), pady=20)

        self.result_label = tk.Label(self.details_frame, text="", font=("Tw Cen MT", 20))
        self.result_label.pack(pady=(10, 10))

    def set_details(self, image_path, result):
        self.image_path = image_path
        self.result = result
        self.display_image()
        self.display_result()

    def display_image(self):
        img = Image.open(self.image_path)
        img = img.resize((300, 300), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

    def display_result(self):
        self.result_label.config(text=f"Prediction: {self.result}")
