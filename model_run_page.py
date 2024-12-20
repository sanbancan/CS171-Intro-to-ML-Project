import customtkinter as ctk
import tkinter as tk
from datetime import datetime
import threading
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
from model import load_data, train_model, create_model, evaluate_model

class ModelRunPage(ctk.CTkFrame):
    def __init__(self, master, model_creator, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.model_creator = model_creator
        self.start_time = None
        self.elapsed_time = 0
        self.training_running = False

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1, uniform="equal")  
        self.grid_columnconfigure(1, weight=2, uniform="equal") 

        self.terminal_frame = ctk.CTkFrame(self)
        self.terminal_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=10, pady=10)

        self.terminal = ScrolledText(self.terminal_frame, wrap=tk.WORD, height=10)
        self.terminal.grid(row=0, column=0, sticky="nsew")
        self.terminal.insert(tk.END, "Running model...\n")

        self.timer_label = ctk.CTkLabel(self.terminal_frame, text="Time Elapsed: 00:00", font=("Arial", 14))
        self.timer_label.grid(row=1, column=0, pady=10)
        
        self.chart_label = ctk.CTkLabel(self, text="", width=600, height=300)
        self.chart_label.grid(row=0, column=1, padx=5, pady=10)

        self.confusion_label = ctk.CTkLabel(self, text="", width=600, height=300)
        self.confusion_label.grid(row=1, column=1, padx=5, pady=10)

        self.run_model_button = ctk.CTkButton(self, text="Run Model", command=self.run_model)
        self.run_model_button.grid(row=2, column=1, pady=20)

    def run_model(self):
        self.start_time = datetime.now()
        self.elapsed_time = 0
        self.training_running = True
        self.terminal.delete(1.0, tk.END)
        self.timer_label.configure(text="Time Elapsed: 00:00")
        self.run_model_button.configure(state="disabled")
        self.update_timer()

        training_thread = threading.Thread(target=self.train_model)
        training_thread.start()

    def update_timer(self):
        if self.training_running:
            self.elapsed_time = (datetime.now() - self.start_time).seconds
            minutes, seconds = divmod(self.elapsed_time, 60)
            self.timer_label.configure(text=f"Time Elapsed: {minutes:02}:{seconds:02}")
            self.after(1000, self.update_timer)

    def train_model(self):
        self.log("Loading and preprocessing data...")
        train_file = r'train/train.json'
        test_file = r'test/test.json'
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(train_file, test_file)
        self.log("Data loaded. Training the model...")

        self.model = create_model()
        history = train_model(X_train, y_train, X_val, y_val)
        self.log("Model training completed. Weights saved to modified_model.weights.h5")

        train_loss_avg = np.mean(history.history['loss'])
        val_loss_avg = np.mean(history.history['val_loss'])
        train_acc_avg = np.mean(history.history['accuracy'])
        val_acc_avg = np.mean(history.history['val_accuracy'])

        self.log(f"Avg Training Loss: {train_loss_avg:.4f}")
        self.log(f"Avg Validation Loss: {val_loss_avg:.4f}")
        self.log(f"Avg Training Accuracy: {train_acc_avg:.4f}")
        self.log(f"Avg Validation Accuracy: {val_acc_avg:.4f}")

        self.model.load_weights('modified_model.weights.h5')
        test_loss, test_accuracy = evaluate_model(self.model, X_test, y_test)

        self.log(f"Test Accuracy: {test_accuracy:.4f}")
        self.training_running = False

        self.save_training_plot(history, test_loss, test_accuracy, X_val, y_val)
        self.master.after(0, self.display_training_results)

    def log(self, message):
        self.terminal.insert(tk.END, message + "\n")
        self.terminal.yview(tk.END)

    def save_training_plot(self, history, test_loss, test_accuracy, X_val, y_val):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_results_with_test.png')
        plt.close()

        # Confusion Matrix Plot
        y_pred = (self.model.predict(X_val) > 0.5).astype(int).flatten()
        cm = confusion_matrix(y_val, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()

    def display_training_results(self):
        try:
            image_1 = Image.open('training_results_with_test.png')
            ctk_image_1 = ctk.CTkImage(image_1, size=(600, 300))
            self.chart_label.configure(image=ctk_image_1)
            self.chart_label.image = ctk_image_1

            image_2 = Image.open('confusion_matrix.png')
            ctk_image_2 = ctk.CTkImage(image_2, size=(600, 300))
            self.confusion_label.configure(image=ctk_image_2)
            self.confusion_label.image = ctk_image_2

            self.run_model_button.configure(state="normal")

        except Exception as e:
            self.log(f"Error displaying images: {e}")










