# Option 1: Using Nougat (specialized for mathematical documents)
from transformers import NougatProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load Nougat model (better for mathematical formulas)
processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

# Load image
image = Image.open('math_formula.png').convert('RGB')

# Process image
pixel_values = processor(image, return_tensors="pt").pixel_values
outputs = model.generate(
    pixel_values,
    min_length=1,
    max_length=4096,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
)

# Decode output
sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print("Nougat Output:")
print(sequence)

print("\n" + "="*50)

# Option 2: Fallback to TrOCR with better settings
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
# trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

# pixel_values_trocr = trocr_processor(image, return_tensors='pt').pixel_values
# generated_ids_trocr = trocr_model.generate(pixel_values_trocr, max_length=1000)
# trocr_text = trocr_processor.batch_decode(generated_ids_trocr, skip_special_tokens=True)[0]

# print("TrOCR Output:")
# print(trocr_text)