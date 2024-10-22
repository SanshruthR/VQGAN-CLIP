# VQGAN-CLIP: Vector Quantized Generative Adversarial Networks with CLIP
![Python](https://img.shields.io/badge/Python-3.7%2B-4B8BBE?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![OpenAI](https://img.shields.io/badge/OpenAI-powered-412991?style=for-the-badge&logo=openai)
![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?style=for-the-badge&logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-9400D3?style=for-the-badge)


https://github.com/user-attachments/assets/4397cd03-013e-4d61-9f4b-e453176ef1b6



This project implements VQGAN (Vector Quantized GAN) combined with CLIP (Contrastive Language–Image Pretraining) to generate images based on text prompts.
## Features
- VQGAN architecture for high-quality image generation.
- Integration with OpenAI's CLIP for text-based guidance.
- Flexible and modular code.
- Video creation for image morphing across iterations.

## Architecture

### VQGAN (Vector Quantized GAN)
The core of VQGAN is its ability to generate high-quality images while using a **vector quantization (VQ)** process to limit the model's expressive capacity.

**Key Components:**
1. **Codebook (Z):** 
   - Encodes continuous latent vectors into discrete codebook entries
   - Maps input vectors to the closest codebook entry
   - Helps maintain discrete representation while training

2. **Quantization Loss:**
   - Minimizes the difference between encoder output and quantized vectors
   - Uses stop-gradient operator to prevent backpropagation through codebook
   - Balances reconstruction quality with codebook usage

3. **GAN Loss:**
   - Uses adversarial training with a discriminator
   - Real data compared against generated images
   - Helps produce realistic and high-quality outputs

### CLIP (Contrastive Language-Image Pretraining)
CLIP, developed by OpenAI, uses a transformer-based architecture to link text and images in a shared latent space. In this project, CLIP guides the VQGAN by scoring the similarity between generated images and the provided text prompt.

**Image-Text Alignment:**
- Encodes both images and text into a shared feature space
- Maximizes similarity between matching image-text pairs
- Uses dot product to measure alignment between encodings

### Training Objective
The combined objective of VQGAN-CLIP is to optimize both the reconstruction and adversarial losses from VQGAN, along with the CLIP-based text-image alignment. The final loss combines:
- Vector quantization loss
- GAN adversarial loss
- CLIP similarity score

These components are weighted to balance their contributions to the final output, allowing for high-quality image generation guided by text descriptions.

   
## Usage
1. Open the provided Google Colab notebook: [Demonstration](https://colab.research.google.com/drive/1ivRYvTaX90PRghQIqAdOyEawkY0YLefa?authuser=0)
2. Upload the model and follow the instructions for generating images based on your text prompts.
3. Customize the text prompts, and the model will generate images based on your input.
## Acknowledgements
- [VQGAN Paper: "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841)
- [CLIP Paper: "Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/abs/2103.00020)
- [OpenAI](https://openai.com/)
