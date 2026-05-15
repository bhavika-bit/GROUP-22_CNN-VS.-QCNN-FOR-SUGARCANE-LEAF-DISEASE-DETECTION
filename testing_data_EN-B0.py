import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
import os
import cv2
import albumentations as A
from tqdm import tqdm
import time
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

test_dir = r"D:\Sugarcane_CNN_Project\Testing_dataset"
processed_test_dir = "processed_testing_dataset"

# ================= PREPROCESS =================

print("\nPreprocessing Testing Dataset...")

if os.path.exists(processed_test_dir):
    import shutil
    shutil.rmtree(processed_test_dir)

transform_preprocess = A.Compose([
    A.Resize(224, 224),
])

for root, dirs, files_list in os.walk(test_dir):

    rel_path = os.path.relpath(root, test_dir)
    save_folder = os.path.join(processed_test_dir, rel_path)
    os.makedirs(save_folder, exist_ok=True)

    for file in tqdm(files_list):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):

            img_path = os.path.join(root, file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            processed = transform_preprocess(image=image)["image"]
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(save_folder, file), processed)

print("Testing dataset ready!")

# ================= LOAD MODEL =================

model = efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    13
)

model.load_state_dict(torch.load("efficientnet_b0.pth"))

print("Loaded 13-class trained model")

# ================= CONVERT TO 5 CLASS =================

class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

# freeze early layers
for param in model.features.parameters():
    param.requires_grad = False

# unfreeze last layers
for param in model.features[-3:].parameters():
    param.requires_grad = True

# replace classifier
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)

model = model.to(device)

print("Model adapted to 5 classes")

# ================= DATA =================

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

dataset = datasets.ImageFolder(root=processed_test_dir, transform=train_transform)

dataset.classes = class_names
dataset.class_to_idx = {cls:i for i,cls in enumerate(class_names)}

loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ================= FINE TUNE =================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam([
    {"params": model.features[-3:].parameters(), "lr":1e-5},
    {"params": model.classifier.parameters(), "lr":1e-4}
])

epochs = 10

print("\nFine tuning...")

for epoch in range(epochs):

    model.train()

    total=0
    correct=0

    for images,labels in loader:

        images,labels = images.to(device),labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        _,pred = torch.max(outputs,1)

        total += labels.size(0)
        correct += (pred==labels).sum().item()

    acc = 100*correct/total

    print(f"Epoch {epoch+1}/{epochs}  Accuracy:{acc:.2f}%")

print("Fine tuning complete")

# ================= TEST =================

test_dataset = datasets.ImageFolder(root=processed_test_dir, transform=test_transform)

test_dataset.classes = class_names
test_dataset.class_to_idx = {cls:i for i,cls in enumerate(class_names)}

test_loader = DataLoader(test_dataset, batch_size=16)

all_preds=[]
all_labels=[]

start=time.time()

model.eval()

with torch.no_grad():

    for images,labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _,pred = torch.max(outputs,1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.numpy())

end=time.time()

accuracy = 100 * sum([p==l for p,l in zip(all_preds,all_labels)]) / len(all_labels)

print("\n===== TEST RESULTS =====")
print(f"Test Accuracy: {accuracy:.2f}%")

print("\nClassification Report:")
print(classification_report(all_labels,all_preds,target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels,all_preds))

print(f"\nTesting Runtime: {end-start:.2f} sec")