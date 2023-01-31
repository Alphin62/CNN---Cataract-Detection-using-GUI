# import required libraires
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from keras.models import load_model


root = tk.Tk()
root.title("Cataract Detection")

# Load the model
model = load_model("CNN_trained_model.h5")


# Function for browse the image & predict the result
def browse_image():
  
    file_path = filedialog.askopenfilename()
    if file_path:
      
        # Read the image using OpenCV
        image = cv2.imread(file_path)

        # Preprocess the image
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)

        # Predict whether it's a cataract or not
        predictions = model.predict(image)
        if predictions[0][0] > 0.5:
            output_label.config(text="Cataract")
        else:
            output_label.config(text="Not Cataract")
        
        # Display the selected image
        image = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        image_label = tk.Label(root, image=image)
        image_label.image = image
        image_label.pack()
        

# Create buttons 
browse_button = tk.Button(text="Browse", command=lambda: browse_image())
browse_button.pack()

output_label = tk.Label(root)
output_label.pack()

root.mainloop()
