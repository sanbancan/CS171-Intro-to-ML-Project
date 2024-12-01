import customtkinter as ctk

class Page(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
    
    def show(self):
        self.lift()
