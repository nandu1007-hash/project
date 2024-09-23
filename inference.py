import torch
import torchvision.transforms as transforms
from PIL import Image
from model import Mymodel  # Import your model from the appropriate file
import os
import argparse
from tqdm import tqdm

# Define the transformation for the input image
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class Args:
    def __init__(self):
        self.distribution = "beta"  # Example argument, adjust as needed

args = Args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Mymodel(args=args, classes=4).to(DEVICE)
model.load_state_dict(torch.load("/kaggle/input/nandumodel/mymodel.pth", map_location=DEVICE, weights_only=True))
model.eval()

# Function to perform inference
def infer(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        try:
            logits, std = model(image)
            # print(f"logits shape: {logits.shape}")  # Debugging statement
            # Average the logits across the spatial dimensions
            logits = logits.mean(dim=[2, 3])
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        except (RuntimeError, AssertionError) as e:
            return None, None

    return predicted_class, probabilities.cpu().numpy()

# Function to calculate accuracy with progress bar
def calculate_accuracy(folder_path, correct_class):
    total_images = 0
    correct_predictions = 0

    image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png"))]

    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(folder_path, filename)
        predicted_class, _ = infer(image_path)
        if predicted_class is not None:
            if predicted_class == correct_class:
                correct_predictions += 1
            total_images += 1

    accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    return accuracy

# Main function to parse arguments and run the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for image classification")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images")
    parser.add_argument("correct_class", type=int, help="Correct class for the images in the folder")
    args = parser.parse_args()

    accuracy = calculate_accuracy(args.folder_path, args.correct_class)
    print(f"Accuracy: {accuracy:.2f}%")