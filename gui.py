import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import cv2
import tensorflow as tf
from scipy.ndimage import center_of_mass

model = tf.keras.models.load_model("model.keras")

root = tk.Tk()
root.title("Digit Recognizer")

width = 280
height = 280

canvas_image = Image.new("L", (width, height), color=255)
draw = ImageDraw.Draw(canvas_image)

def draw_on_canvas(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
    draw.ellipse([x - r, y - r, x + r, y + r], fill='black')

def clear_canvas():
    canvas.delete("all")
    global canvas_image, draw
    canvas_image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas_image)

def save_image():
    canvas_image.save("digit.png")
    print("Image saved as digit.png")

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to read image from", path)
        return None

    img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    img = cv2.bitwise_not(img)
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    margin = 4
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + 2 * margin, img.shape[1] - x)
    h = min(h + 2 * margin, img.shape[0] - y)
    img = img[y:y+h, x:x+w]

    img = cv2.resize(img, (18, 18), interpolation=cv2.INTER_AREA)
    padded = np.pad(img, ((5,5), (5,5)), mode='constant', constant_values=0)

    cy, cx = center_of_mass(padded)
    shiftx = np.round(14 - cx).astype(int)
    shifty = np.round(14 - cy).astype(int)
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    padded = cv2.warpAffine(padded, M, (28, 28))

    padded = padded / 255.0
    padded = padded.reshape(1, 28, 28, 1)

    return padded

def predict_digit(path='digit.png'):
    img = preprocess_image(path)
    if img is None:
        return "Error"
    prediction = model.predict(img)
    return np.argmax(prediction)

canvas = tk.Canvas(root, width=width, height=height, bg="white")
canvas.pack()
canvas.bind("<B1-Motion>", draw_on_canvas)

def predict():
    save_image()
    result = predict_digit()
    if result != "Error":
        result_label.config(text=f"Predicted Digit: {result}")
    else:
        result_label.config(text="Prediction Failed")

result_label = tk.Label(root, text="Draw a digit and click Predict", font=("Helvetica", 16))
result_label.pack(pady=10)

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()

clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

root.mainloop()
