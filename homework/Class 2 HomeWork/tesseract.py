from PIL import Image
import pytesseract 
import os
import logging

image = Image.open('test.png')

#perform OCR on the image
text = pytesseract.image_to_string(image)

file_name = "image_to_text.txt"
with open(file_name, "w", encoding="utf-8") as f:
    f.write(text)

logging.info(text)