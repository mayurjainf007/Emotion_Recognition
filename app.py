import tkinter as tk
from tkinter import messagebox
import subprocess

def run_script(script_name):
    try:
        # subprocess.run is used here for simplicity. Adjust the paths to your actual scripts.
        subprocess.run(['python', script_name], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while running the script: {e}")

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Script Runner GUI")
        self.geometry("300x200")

        # Button to run Emotion Classification
        self.btn_run1 = tk.Button(self, text="Train Model", command=lambda: run_script('model_generation.py'))
        self.btn_run1.pack(pady=20)

        # Button to run Emotion Classification Live
        self.btn_run2 = tk.Button(self, text="Emotion Classification", command=lambda: run_script('emotion_classification.py'))
        self.btn_run2.pack(pady=20)

        # Button to run Confusion Matrix
        self.btn_run3 = tk.Button(self, text="Model Summary", command=lambda: run_script('model_summary.py'))
        self.btn_run3.pack(pady=20)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
