import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model_path = r'D:\project\A.I. in healthcare\tb_detector.keras'

model = load_model(model_path)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    return "TB! please consult a doctor" if prediction[0] > 0.8 else "Normal! you are safe"

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

        
        result = predict_image(file_path)
        result_label.config(text=f"Prediction: {result}")


app = tk.Tk()
app.title("Tuberculosis Detection")
app.attributes('-fullscreen', True)  

bg_image = Image.open('1.png')
bg_image = bg_image.resize((app.winfo_screenwidth(), app.winfo_screenheight()), Image.LANCZOS)  # Updated line
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(app, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

panel = tk.Label(app)
panel.pack(side="top", padx=10, pady=10)

upload_button = tk.Button(app, text="Upload X-ray Image", command=upload_image, height=2, width=20)
upload_button.pack(side="top", pady=20)

result_label = tk.Label(app, text="", font=("Helvetica", 16), bg='white')
result_label.pack(side="top", pady=20)

exit_button = tk.Button(app, text="Exit", command=app.quit, height=2, width=20)
exit_button.pack(side="bottom", pady=20)

app.mainloop()
