import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dataloader import CustomImageDataset
from model import Mymodel  # Import your model from the appropriate file
from torch.utils.data import DataLoader
from tqdm import tqdm

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = CustomImageDataset(
    root="/kaggle/input/brain-tumor-mri-dataset", split="train", transform=transform
)
test_dataset = CustomImageDataset(
    root="/kaggle/input/brain-tumor-mri-dataset", split="test", transform=transform
)

# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)


class Args:
    def __init__(self):
        self.distribution = "beta"  # Example argument, adjust as needed


args = Args()

# Initialize model, assuming the model outputs both predictions and uncertainties
model = Mymodel(args, classes=4).to(DEVICE)  # For binary classification, classes=1

criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Uncertainty-based loss
def bayesian_uncertainty_loss(logits, labels, std, device):
    bce = criterion(logits, labels)

    uncertainty_penalty = torch.mean(
        0.5 * torch.exp(-std) * (logits - labels) ** 2 + 0.5 * std
    )

    return bce + uncertainty_penalty


# Training loop
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device).float()

        optimizer.zero_grad()

        # Model outputs both logits (predictions) and standard deviation (uncertainty)
        logits, std = model(inputs)

        # Calculate the combined loss: BCE + uncertainty loss
        loss = bayesian_uncertainty_loss(logits, labels, std, device)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += labels.size(0)
        _, predicted = logits.max(1)
        correct += (predicted == labels.unsqueeze(1)).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).float()

            logits, std = model(inputs)

            loss = bayesian_uncertainty_loss(logits, labels.unsqueeze(1), std, device)

            running_loss += loss.item()
            predicted = (torch.sigmoid(logits) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


for epoch in tqdm(range(NUM_EPOCHS)):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, DEVICE)
    test_loss, test_acc = evaluate(model, test_loader, DEVICE)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print()

print("Training complete!")

# Save the trained model weights
torch.save(model.state_dict(), "mymodel.pth")
