import customtkinter as ctk
from analysis import AnalysisPage
from gallery import GalleryPage
from home import HomePage
from details import DetailsPage
from model import create_model
from model_run_page import ModelRunPage

ctk.set_appearance_mode("Light")

class View(ctk.CTkFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gallery = GalleryPage(self)
        self.details_page = DetailsPage(self)
        self.home = HomePage(self)
        self.model_run = ModelRunPage(self, model_creator=create_model)
        self.analysis = AnalysisPage(gallery=self.gallery, master=self, model=create_model(), details_page=self.details_page)

        self.sidebar = ctk.CTkFrame(self, width=200)
        self.container = ctk.CTkFrame(self)

        self.sidebar.pack(side="left", fill="y", expand=False)
        self.container.pack(side="right", fill="both", expand=True)

        self.home.place(in_=self.container, x=0, y=0, relwidth=1, relheight=1)
        self.model_run.place(in_=self.container, x=0, y=0, relwidth=1, relheight=1)
        self.analysis.place(in_=self.container, x=0, y=0, relwidth=1, relheight=1)
        self.details_page.place(in_=self.container, x=0, y=0, relwidth=1, relheight=1)
        self.gallery.place(in_=self.container, x=0, y=0, relwidth=1, relheight=1)

        home_button = ctk.CTkButton(self.sidebar, text="Home Page", command=self.show_home)
        analysis_button = ctk.CTkButton(self.sidebar, text="Analysis", command=self.show_analysis)
        gallery_button = ctk.CTkButton(self.sidebar, text="Gallery", command=self.show_gallery)
        model_button = ctk.CTkButton(self.sidebar, text="Model", command=self.show_model_run)

        home_button.pack(pady=10, padx=10, fill='x')
        analysis_button.pack(pady=10, padx=10, fill='x')
        gallery_button.pack(pady=10, padx=10, fill='x')
        model_button.pack(pady=10, padx=10, fill='x') 

        self.show_home()

    def show_frame(self, frame):
        """Switch to a specific frame."""
        frame.tkraise()

    def show_home(self):
        self.show_frame(self.home)

    def show_analysis(self):
        self.show_frame(self.analysis)

    def show_gallery(self):
        self.show_frame(self.gallery)

    def show_model_run(self):
        self.show_frame(self.model_run)

    def open_details_page(self, image_path, result):
        self.details_page.set_details(image_path, result)
        self.show_frame(self.details_page)

if __name__ == "__main__":
    root = ctk.CTk()
    root.title("Iceberg Detection and Analysis")
    root.geometry("1280x820")
    main = View(root)
    main.pack(side="top", fill="both", expand=True)
    root.mainloop()
