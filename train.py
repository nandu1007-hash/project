import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from dataloader import CustomImageDataset
from model import Mymodel  # Import your model from the appropriate file
from torch.utils.data import DataLoader
from tqdm import tqdm

BATCH_SIZE = 1
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
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
model = Mymodel(args, classes=4).to(DEVICE)  # For 4-class classification
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Uncertainty-based loss
def bayesian_uncertainty_loss(logits, labels, std, device):
    # Debug print statements
    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels unique values: {torch.unique(labels)}")

    # Ensure labels are the correct shape and type
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)
    labels = labels.squeeze().long()

    ce = criterion(logits, labels)
    uncertainty_penalty = torch.mean(
        0.5
        * torch.exp(-std)
        * torch.sum(
            (F.one_hot(labels, num_classes=logits.size(1)) - F.softmax(logits, dim=1))
            ** 2,
            dim=1,
        )
        + 0.5 * std
    )
    return ce + uncertainty_penalty


# Training loop
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # Model outputs both logits (predictions) and standard deviation (uncertainty)
        logits, std = model(inputs)
        # Calculate the combined loss: CE + uncertainty loss
        loss = bayesian_uncertainty_loss(logits, labels, std, device)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total += labels.size(0)
        _, predicted = logits.max(1)
        correct += (predicted == labels.squeeze()).sum().item()
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
            inputs, labels = inputs.to(device), labels.to(device)
            logits, std = model(inputs)
            loss = bayesian_uncertainty_loss(logits, labels, std, device)
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()
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
