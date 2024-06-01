import cv2
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import os
from playsound import playsound

# Ensure PyTorch is installed
try:
    import torch
    print("PyTorch version:", torch._version_)
except ImportError as e:
    print("PyTorch is not installed. Please install it using 'pip install torch'")
    exit()

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Process the image and generate a caption
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_length=150, num_beams=5, early_stopping=True) 
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(caption)

    # Display the frame with the caption
    # cv2.putText(frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.imshow('BLIP Real-Time Captioning', frame)

    print(caption)

    # Text-to-speech using gTTS
    tts = gTTS(text=caption, lang='en')
    tts.save("caption.mp3")
    playsound("caption.mp3")
    os.remove("caption.mp3")

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()