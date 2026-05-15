import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import cv2
import albumentations as A
from tqdm import tqdm
import time
from sklearn.metrics import classification_report, confusion_matrix

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= PATHS =================
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

print("✅ Testing dataset ready!")

# ================= MODEL =================
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):

        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ================= LOAD PRETRAINED MODEL =================
checkpoint = torch.load("custom_cnn.pth", map_location=device)

pretrained_model = CustomCNN(13).to(device)

pretrained_model.load_state_dict(checkpoint["model_state_dict"])

print("Pretrained 13-class model loaded!")

# ================= TRANSFER LEARNING =================

class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

# freeze backbone
for param in pretrained_model.parameters():
    param.requires_grad = False

# replace final layer
pretrained_model.fc2 = nn.Linear(512, len(class_names))

model = pretrained_model.to(device)

# ================= QUICK FINETUNE =================

train_dir = r"D:\Sugarcane_CNN_Project\processed_testing_dataset"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.Adam(model.fc2.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

model.train()

for epoch in range(10):

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1}/10")

model.eval()

print("Transfer model ready!")

# ================= TEST =================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(root=processed_test_dir, transform=test_transform)

test_dataset.classes = class_names
test_dataset.class_to_idx = {cls: i for i, cls in enumerate(class_names)}

test_loader = DataLoader(test_dataset, batch_size=16)

print("Classes used:", class_names)

all_preds = []
all_labels = []

start = time.time()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

end = time.time()

# ================= METRICS =================

accuracy = 100 * sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)

print("\n===== TEST RESULTS =====")
print(f"Test Accuracy: {accuracy:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print(f"\nTesting Runtime: {end - start:.2f} sec")