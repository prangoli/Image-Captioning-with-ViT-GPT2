import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import streamlit as st
import os
from zipfile import ZipFile
import io

# Set the page configuration
st.set_page_config(page_title="Enhanced Image Captioning", layout="wide")

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, feature_extractor, tokenizer

model, feature_extractor, tokenizer = load_model()

# Function to generate image caption
def generate_caption(image, max_length, num_beams, temperature, top_k, top_p):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    
    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values, 
            max_length=max_length, 
            num_beams=num_beams, 
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_dict_in_generate=True
        ).sequences
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return caption

# Streamlit app
st.title("Enhanced Image Captioning")
st.write("Upload images to get captions generated by the model.")

# Sidebar for parameter adjustment and model selection
st.sidebar.title("Settings")
max_length = st.sidebar.slider("Max Caption Length", min_value=10, max_value=50, value=16, step=1)
num_beams = st.sidebar.slider("Number of Beams", min_value=1, max_value=10, value=4, step=1)
temperature = st.sidebar.slider("Temperature", min_value=0.7, max_value=1.5, value=1.0, step=0.1)
top_k = st.sidebar.slider("Top-k", min_value=0, max_value=100, value=50, step=1)
top_p = st.sidebar.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.1)

st.sidebar.markdown("""
### About
This app uses the ViT-GPT2 model for image captioning. Adjust the parameters to see how the captions change.
- **Max Caption Length**: Maximum length of the generated caption.
- **Number of Beams**: Number of beams for beam search. More beams result in better quality but are slower.
- **Temperature**: Sampling temperature. Higher values result in more diverse outputs.
- **Top-k**: The number of highest probability vocabulary tokens to keep for top-k-filtering.
- **Top-p**: The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.
""")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    captions = []
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("Generating caption...")
            
            with st.spinner('Processing...'):
                caption = generate_caption(
                    image, max_length, num_beams, temperature, top_k, top_p
                )
                captions.append((uploaded_file.name, image, caption))
            
            st.write(f"**Generated Caption:** {caption}")
        except Exception as e:
            st.error(f"Error processing the image {uploaded_file.name}: {e}")

    # Provide option to download the captions and images
    if captions:
        with io.BytesIO() as buffer:
            with ZipFile(buffer, "w") as zip_file:
                for img_name, img, cap in captions:
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    zip_file.writestr(f"{os.path.splitext(img_name)[0]}.jpg", img_byte_arr.getvalue())
                    zip_file.writestr(f"{os.path.splitext(img_name)[0]}.txt", cap)
            buffer.seek(0)
            st.download_button("Download Captions and Images", buffer, "captions_and_images.zip", "application/zip")
else:
    st.info("Please upload image files (jpg, jpeg, png).")


