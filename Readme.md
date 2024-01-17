# Anime vs Animal Image Classifier

This project uses a pre-trained ResNet-18 model to classify images as either anime or animal. The model has been fine-tuned on a custom dataset of anime and animal images. Specially Pokemon images.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- Gradio
- PIL

You can install the required packages using pip:


pip install torch torchvision gradio pillow


# Installing

Clone the repository to your local machine:

git clone https://github.com/felipefe20/Image-classification-pytorch.git
cd Image-classification-pytorch


Running the Application
You can run the application using the following command:

python app.py

This will start a Gradio interface in your web browser where you can upload an image and get a prediction.



## Project Structure
- app.py: This is the main application file. It defines the image transformations, the prediction function, and the Gradio interface.
- notebook_anime_vs_animal.ipynb: This Jupyter notebook contains the code for training and evaluating the model.
