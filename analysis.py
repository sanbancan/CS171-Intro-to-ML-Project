import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import os
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from PIL import ImageTk
from page import Page
import threading
from model import predict_image

class AnalysisPage(Page):
    def __init__(self, gallery, master, model, details_page, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.gallery = gallery
        self.details_page = details_page
        self.uploaded_image = None
        self.model = model

        title_label = tk.Label(self, text="Choose a photo to upload for analysis",
                               font=("Tw Cen MT", 40),
                               fg='#3a7ebf', bg='#cfcfcf')
        title_label.pack(pady=(20, 0))

        self.upload_button = ctk.CTkButton(self, text="Upload Photo", command=self.upload_photo)
        self.upload_button.pack(pady=(20, 10))

        self.submit_button = ctk.CTkButton(self, text="Submit", command=self.submit_photo)
        self.submit_button.pack(pady=(10, 20))

        self.image_label = tk.Label(self, bg='#cfcfcf')
        self.image_label.pack(pady=(10, 20))

        self.images_folder = "images"
        os.makedirs(self.images_folder, exist_ok=True)

    def upload_photo(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.uploaded_image = file_path
            self.display_image(file_path)

    def display_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((400, 400), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

    def submit_photo(self):
        if self.uploaded_image:
            threading.Thread(target=self._process_and_predict, daemon=True).start()
        else:
            print("No image uploaded.")

    def _process_and_predict(self):
        try:
            saved_image_path = os.path.join(self.images_folder, os.path.basename(self.uploaded_image))
            img = Image.open(self.uploaded_image)
            img.save(saved_image_path)
            processed_image = self.process_image(saved_image_path)
            self.model.load_weights('modified_model.weights.h5')
            result = predict_image(self.model, processed_image)
            self.master.after(0, lambda: self._update_gui_after_prediction(saved_image_path, result))
        except Exception as e:
            print(f"Error during prediction: {e}")

    def _update_gui_after_prediction(self, saved_image_path, result):
        print(f"Prediction: {result}")
        self.gallery.add_image(saved_image_path)
        self.details_page.set_details(saved_image_path, result)
        self.master.open_details_page(saved_image_path, result)

    def process_image(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (75, 75))
        img_normalized = img.astype('float32') / 255.0
        img_reshaped = np.reshape(img_normalized, (1, 75, 75, 1))
        return img_reshaped


