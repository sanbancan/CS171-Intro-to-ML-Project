import customtkinter as ctk
import tkinter as tk
from page import Page
import webbrowser

class HomePage(Page):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        def callback(url):
            webbrowser.open_new_tab(url)
        
        title_label = tk.Label(self, text="Iceberg Detection and Analysis", 
                               font=("Tw Cen MT", 40), 
                               bg='#cfcfcf')
        title_label.pack(pady=(20, 0))
        
        description_label = tk.Label(self, text=("Data regarding icebergs can be very crucial to monitoring climate change as well\n"
                                                  "as sea levels. Using SAR imaging combined with despeckling techniques can provide\n"
                                                  "ways to examine these icebergs more closely, and obtain the necessary information\n"
                                                  "required to make predictions on the effects of climate change as well as the effects\n"
                                                  "of the icebergs themselves on the ecosystem."), 
                                     font=("Tw Cen MT", 15), bg='#cfcfcf', anchor='center')
        description_label.pack(pady=(10, 10), ipady=20)
        
        instruction_label = tk.Label(self, text="Instructions",
                                      font=("Tw Cen MT", 30), 
                                      fg='#3a7ebf', bg='#cfcfcf')
        instruction_label.pack(pady=(20, 10), ipady=20)
        
        instruction_label2 = tk.Label(self, text="To start, first navigate to the Model page to train the model. Modifications can be made to improve results.\n"
                                      "Then, in the Analysis page, the trained model will predict if uploaded images are icebergs or non-icebergs.\n"
                                      "The Gallery page will store the image and the model's prediction.",
                                       font=("Tw Cen MT", 15), bg='#cfcfcf')
        instruction_label2.pack(pady=(10, 10))

        
        repo_label = tk.Label(self, text="Link to the Github repository.", 
                                  font=("Tw Cen MT", 15), bg='#cfcfcf',
                                  cursor="hand2")
        repo_label.pack(side='bottom', pady=(20, 10), ipady=20)
        repo_label.bind("<Button-1>", lambda e:
            callback("https://github.com/levictoria0117/cs171-final-project"))

