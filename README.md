# Image-Captioning-with-ViT-GPT2
App for generating descriptions based on images


# Enhanced Image Captioning

## Overview

Enhanced Image Captioning is a web application that leverages advanced deep learning models to generate descriptive captions for images. The application is built using Streamlit, PyTorch, and the Hugging Face Transformers library. It integrates a Vision-Encoder-Decoder architecture, specifically utilizing the ViT-GPT2 model, which combines the Vision Transformer (ViT) as the encoder and GPT-2 as the decoder.


## Examples \



<img width="1858" alt="skyman" src="https://github.com/user-attachments/assets/3c430035-808b-4ab8-be31-4ae1650a46f9"> \



<img width="1856" alt="kiddog" src="https://github.com/user-attachments/assets/6d49f211-0002-47e9-b149-c5776dbfe35b">



## How It Works

### Vision-Encoder-Decoder Model

The model used in this application is based on a Vision-Encoder-Decoder architecture:

1. **Vision Transformer (ViT) as Encoder:**
   - The Vision Transformer (ViT) is a model that applies Transformer architecture directly to image patches. Unlike traditional Convolutional Neural Networks (CNNs), ViT divides the image into patches, treats each patch as a sequence, and processes it using self-attention mechanisms.
   - The ViT model encodes the visual information of the input image into a fixed-length representation. This encoded representation captures essential features of the image, such as objects, textures, and spatial relationships.

2. **GPT-2 as Decoder:**
   - GPT-2 (Generative Pre-trained Transformer 2) is a state-of-the-art language model that generates human-like text based on the given input. It has been pre-trained on a large corpus of text data, making it capable of generating coherent and contextually relevant text.
   - In this application, GPT-2 takes the encoded image representation from ViT as input and generates a descriptive caption. The decoder uses beam search and other advanced techniques to optimize the generated captions.

### Image Processing Pipeline

1. **Image Input:**
   - Users upload images in supported formats (JPG, JPEG, PNG) through the web interface. These images are processed and converted into the appropriate format for the model.

2. **Feature Extraction:**
   - The ViT model processes the input image and extracts key features. These features are then fed into the GPT-2 model.

3. **Caption Generation:**
   - The GPT-2 model generates text based on the features provided by the ViT model. The output is a descriptive caption that reflects the content of the image.

4. **Parameter Adjustment:**
   - Users can fine-tune the caption generation process by adjusting parameters such as maximum caption length, number of beams, temperature, top-k, and top-p. These parameters control aspects like diversity, length, and the precision of the generated captions.

## Libraries Used

### PyTorch

- **PyTorch** is a deep learning framework that provides flexible and efficient tools for building and training neural networks. It is widely used in research and industry due to its dynamic computation graph and strong support for GPU acceleration.

### Hugging Face Transformers

- **Hugging Face Transformers** is a library that provides pre-trained models and tools for natural language processing (NLP) and computer vision tasks. It offers a wide range of models, including GPT-2, BERT, ViT, and others, that can be easily fine-tuned for specific tasks.

### Streamlit

- **Streamlit** is an open-source app framework designed for creating and sharing data science applications. It allows developers to build interactive web applications quickly and easily, using Python scripts. In this project, Streamlit is used to create the user interface, handle image uploads, and display generated captions.

## Understanding the Models

### Vision Transformer (ViT)

- The **Vision Transformer** model was introduced as an alternative to traditional CNNs for image processing tasks. By treating image patches as tokens and applying the Transformer architecture, ViT can capture long-range dependencies and global context more effectively. This model is particularly effective for tasks that require a holistic understanding of the image.

### GPT-2

- **GPT-2** is part of the Transformer family of models and is designed for generating natural language text. It uses a multi-layer architecture with self-attention mechanisms to understand and generate text sequences. GPT-2's ability to generate text that is contextually relevant and coherent makes it ideal for tasks like image captioning, where the generated text needs to accurately describe the visual content.

## Conclusion

The Enhanced Image Captioning project demonstrates the power of combining advanced deep learning models to tackle complex tasks such as generating descriptive captions for images. By integrating the Vision Transformer with GPT-2, the application is capable of producing high-quality captions that are both accurate and contextually appropriate. The use of PyTorch and Hugging Face Transformers libraries enables efficient model deployment, while Streamlit provides a seamless interface for users to interact with the application.
