import customtkinter as ctk
import tkinter as tk
from datetime import datetime
import threading
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import numpy as np

import matplotlib.pyplot as plt
from model import load_data, train_model

class ModelRunPage(ctk.CTkFrame):
    def __init__(self, master, model_creator, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.model_creator = model_creator
        self.start_time = None
        self.elapsed_time = 0
        self.training_running = False

        self.run_model_button = ctk.CTkButton(self, text="Run Model", command=self.run_model)
        self.run_model_button.pack(pady=20)

        self.terminal_frame = ctk.CTkFrame(self)
        self.terminal_frame.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)

        self.terminal = ScrolledText(self.terminal_frame, wrap=tk.WORD, height=10)
        self.terminal.pack(fill="both", expand=True)
        self.terminal.insert(tk.END, "Running model...\n")

        # Timer display
        self.timer_label = ctk.CTkLabel(self.terminal_frame, text="Time Elapsed: 00:00", font=("Arial", 14))
        self.timer_label.pack(side="bottom", pady=10)

        # Label for displaying the training charts
        self.chart_label = ctk.CTkLabel(self.terminal_frame)
        self.chart_label.pack(side="bottom", pady=10)

    def run_model(self):
        self.start_time = datetime.now()
        self.elapsed_time = 0
        self.training_running = True
        self.terminal.delete(1.0, tk.END)
        self.timer_label.configure(text="Time Elapsed: 00:00")
        self.run_model_button.pack_forget()
        self.run_model_button.configure(state="disabled")
        self.update_timer()

        # Start training in a separate thread
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
        train_file = r'dataset/train.json/data/processed/train.json'
        test_file = r'dataset/test.json/data/processed/test.json'
        X_train, y_train, X_val, y_val, _ = load_data(train_file, test_file)
        self.log("Data loaded. Training the model...")

        model = self.model_creator()
        history = train_model(X_train, y_train, X_val, y_val)
        self.log("Model training complete. Weights saved.")

        # Calculate averages
        train_loss_avg = np.mean(history.history['loss'])
        val_loss_avg = np.mean(history.history['val_loss'])
        train_acc_avg = np.mean(history.history['accuracy'])
        val_acc_avg = np.mean(history.history['val_accuracy'])

        # Log the averages
        self.log(f"Avg Training Loss: {train_loss_avg:.4f}")
        self.log(f"Avg Validation Loss: {val_loss_avg:.4f}")
        self.log(f"Avg Training Accuracy: {train_acc_avg:.4f}")
        self.log(f"Avg Validation Accuracy: {val_acc_avg:.4f}")
        self.model = model
        self.training_running = False

        # Save plot and update GUI in the main thread
        self.save_training_plot(history)
        self.master.after(0, self.display_training_results)

    def log(self, message):
        self.terminal.insert(tk.END, message + "\n")
        self.terminal.yview(tk.END)

    def save_training_plot(self, history):
        plt.figure(figsize=(12, 6))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        self.plot_path = 'training_results.png'
        plt.savefig(self.plot_path)
        plt.close()

    def display_training_results(self):
        # Convert saved plot to CTkImage and update the label
        image = Image.open(self.plot_path)
        ctk_image = ctk.CTkImage(image, size=(600, 300))  # Adjust size as needed
        self.chart_label.configure(image=ctk_image)
        self.chart_label.image = ctk_image  # Keep a reference to avoid garbage collection

        self.run_model_button.configure(state="normal")









