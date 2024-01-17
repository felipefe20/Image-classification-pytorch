import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models

# Define the transformations
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define prediction function
def predict(input_image: Image.Image):
    # Load the pre-trained model
    model_conv = models.resnet18(pretrained=True)
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = torch.nn.Linear(num_ftrs, 2)  # Replace the last fully connected layer with a new one
    model_conv.load_state_dict(torch.load("final_model_conv.pt"))
    model_conv.eval()

    # Transform the input image
    image = data_transforms(input_image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Make predictions
    with torch.no_grad():
        prediction = model_conv(image)

    probabilities = torch.nn.functional.softmax(prediction, dim=1).numpy()[0]

    # Assuming your model returns logits, you might want to apply sigmoid
    class_names= ["animal", "anime"]
    label = class_names[probabilities.argmax()]

    return label


# Define Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    live=True  # Set to False if you want to use it without the Gradio live interface
)

iface.launch(share=True)

