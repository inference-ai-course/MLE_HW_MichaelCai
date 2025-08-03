from PIL import Image
import pytesseract 
import logging

image = Image.open('spanish.png')

#perform OCR on the image
text = pytesseract.image_to_string(image,lang="span+eng")

file_name = "image_to_text_spanish.txt"
with open(file_name, "w", encoding="utf-8") as f:
    f.write(text)

logging.info(text)