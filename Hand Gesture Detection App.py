import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

model = YOLO('best.pt')

def detect_gesture(image):
    results = model(image)
    annotated = results[0].plot()
    return Image.fromarray(annotated[..., ::-1])

demo = gr.Interface(
    fn=detect_gesture,
    inputs=gr.Image(type="pil", label="Upload hand gesture image"),
    outputs=gr.Image(type="pil", label="Detected gesture"),
    title="Hand Gesture Detection",
    description="Upload an image to detect hand gestures using YOLOv8"
)

demo.launch()