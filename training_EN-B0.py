import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import time

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
batch_size = 16
learning_rate = 1e-6
epochs = 100

# Load weights
weights = EfficientNet_B0_Weights.DEFAULT

# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transform = weights.transforms()

# Dataset
full_dataset = datasets.ImageFolder(root="dataset")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

train_dataset = datasets.ImageFolder(root="dataset", transform=train_transform)
val_dataset = datasets.ImageFolder(root="dataset", transform=val_transform)

train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size)

# Model
model = efficientnet_b0(weights=weights)

# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze last layers
for param in model.features[-2:].parameters():
    param.requires_grad = True

# Replace classifier
num_classes = len(full_dataset.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model = model.to(device)

# Loss & optimizer (L2 Regularization added)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Tracking
train_acc_list, val_acc_list = [], []
train_loss_list, val_loss_list = [], []

# Training loop
for epoch in range(epochs):

    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    # Store
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print("-" * 50)

print("Training Complete!")

# Runtime end
start_time = time.time()
end_time = time.time()
print(f"\nTotal Training Time: {end_time - start_time:.2f} seconds")

# Fitting Probability Function
def fitting_probability(train_acc_list, val_acc_list):

    train_acc = train_acc_list[-1]
    val_acc = val_acc_list[-1]

    gap = abs(train_acc - val_acc)

    overfit_prob = min(1, gap/train_acc)
    underfit_prob = max(0, (80 - train_acc)/80)
    goodfit_prob = 1 - max(overfit_prob, underfit_prob)

    print("\nModel Fit Analysis")
    print("-------------------")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Gap: {gap:.2f}%\n")

    print(f"Overfitting Probability: {overfit_prob:.2f}")
    print(f"Underfitting Probability: {underfit_prob:.2f}")
    print(f"Good Fit Probability: {goodfit_prob:.2f}")

    # Plot
    plt.figure()
    labels = ["Overfit", "Underfit", "Good Fit"]
    values = [overfit_prob, underfit_prob, goodfit_prob]
    plt.bar(labels, values)
    plt.title("Model Fit Probability")
    plt.ylim(0, 1)
    plt.show()

    if overfit_prob > 0.6:
        print("Model is likely OVERFITTING")
    elif underfit_prob > 0.6:
        print("Model is likely UNDERFITTING")
    else:
        print("Model is reasonably well fitted")

# Call function
fitting_probability(train_acc_list, val_acc_list)

# Accuracy Plot
plt.figure()
plt.plot(range(1, epochs+1), train_acc_list, label="Train Accuracy")
plt.plot(range(1, epochs+1), val_acc_list, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.legend()
plt.show()

# Loss Plot
plt.figure()
plt.plot(range(1, epochs+1), train_loss_list, label="Train Loss")
plt.plot(range(1, epochs+1), val_loss_list, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.show()

torch.save(model.state_dict(), "efficientnet_b0.pth")