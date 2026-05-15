PHASE1_EPOCHS     = 20      # head only
PHASE2_EPOCHS     = 30       # unfreeze last 3 blocks
EARLY_STOP_PATIENCE = 7       # stop if val loss doesn't improve for 7 epochs


LR_HEAD           = 1e-3
LR_FINETUNE       = 1e-5
WEIGHT_DECAY      = 1e-4      # L2 regularization


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]



#  STEP 1 — AUTO-DETECT CLASSES FROM DATASET FOLDER

print(f"\n[1] Scanning dataset folder: '{DATASET_DIR}'")


CLASS_NAMES = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])
NUM_CLASSES = len(CLASS_NAMES)


print(f"    Classes found ({NUM_CLASSES}): {CLASS_NAMES}")



#  STEP 2 — MOBILENETV2 WITH DROPOUT HEAD

print(f"\n[2] Loading MobileNetV2 pretrained on ImageNet...")


model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)


# Freeze all feature layers
for param in model.features.parameters():
    param.requires_grad = False


# Replace classifier with Dropout + Linear (helps prevent overfitting)
in_features = model.classifier[1].in_features  # 1280
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),               # added dropout before head
    nn.Linear(in_features, NUM_CLASSES)
)


model = model.to(device)


total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    Total params    : {total_params:,}")
print(f"    Trainable (P1)  : {trainable_params:,}  (head only)")



#  STEP 3 — DATASET WITH BALANCED SAMPLING

print(f"\n[3] Loading dataset ({SAMPLES_PER_CLASS} images/class × {NUM_CLASSES} classes = {SAMPLES_PER_CLASS*NUM_CLASSES} total)")


base_dataset = datasets.ImageFolder(root=DATASET_DIR)
# Align class ordering with sorted CLASS_NAMES
base_dataset.class_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}
base_dataset.classes      = CLASS_NAMES


# # Balanced random sampling
# class_indices = defaultdict(list)
# for idx, label in enumerate(base_dataset.targets):
#     class_indices[label].append(idx)


# selected_indices = []
# for label in sorted(class_indices.keys()):
#     available = class_indices[label]
#     n = min(SAMPLES_PER_CLASS, len(available))
#     selected_indices.extend(random.sample(available, n))


# Balanced random sampling
class_indices = defaultdict(list)
for idx, label in enumerate(base_dataset.targets):
    class_indices[label].append(idx)


# Use every image available (no sampling cap)
selected_indices = []
for label in sorted(class_indices.keys()):
    selected_indices.extend(class_indices[label])


dist = Counter([base_dataset.targets[i] for i in selected_indices])
print(f"    Class distribution: { {CLASS_NAMES[k]: v for k, v in sorted(dist.items())} }")
print(f"    Total images used : {len(selected_indices)}")


dist = Counter([base_dataset.targets[i] for i in selected_indices])
print(f"    Class distribution: { {CLASS_NAMES[k]: v for k, v in sorted(dist.items())} }")


# Aggressive augmentation to fight overfitting on small data
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),                         # random crop vs center
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # randomly erase small patch
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


train_dataset_full = datasets.ImageFolder(root=DATASET_DIR, transform=train_transform)
val_dataset_full   = datasets.ImageFolder(root=DATASET_DIR, transform=val_transform)


for ds in [train_dataset_full, val_dataset_full]:
    ds.class_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}
    ds.classes      = CLASS_NAMES


# # 80/20 split
# train_size = int(0.8 * len(selected_indices))
# val_size   = len(selected_indices) - train_size
# train_idx, val_idx = random_split(selected_indices, [train_size, val_size])


# train_loader = DataLoader(Subset(train_dataset_full, train_idx.indices),
#                           batch_size=BATCH_SIZE, shuffle=True)
# val_loader   = DataLoader(Subset(val_dataset_full, val_idx.indices),
#                           batch_size=BATCH_SIZE)


# print(f"    Train: {train_size}  |  Val: {val_size}")


# Per-class 80/20 stratified split
train_indices_final = []
val_indices_final   = []


for label in sorted(class_indices.keys()):
    cls_samples = [i for i in selected_indices if base_dataset.targets[i] == label]
    random.shuffle(cls_samples)
    split_at = int(0.8 * len(cls_samples))
    train_indices_final.extend(cls_samples[:split_at])
    val_indices_final.extend(cls_samples[split_at:])


train_size = len(train_indices_final)
val_size   = len(val_indices_final)


train_loader = DataLoader(Subset(train_dataset_full, train_indices_final),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(Subset(val_dataset_full, val_indices_final),
                          batch_size=BATCH_SIZE)


print(f"    Train: {train_size}  |  Val: {val_size}")
# print(f"    Per class → Train: {int(train_size/NUM_CLASSES)}  Val: {int(val_size/NUM_CLASSES)}")
print(f"    Train: {train_size}  |  Val: {val_size}  (80/20 per-class stratified)")





#  STEP 4 — HELPERS

criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(loader)   # per-batch average (cleaner for plotting)
    acc      = 100.0 * correct / total
    return avg_loss, acc




class EarlyStopping:
    """Stops training when val loss doesn't improve for `patience` epochs."""
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.counter    = 0
        self.triggered  = False


    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered





#  STEP 5 — PHASE 1: TRAIN HEAD ONLY

total_epochs_run = 0
train_acc_list, val_acc_list   = [], []
train_loss_list, val_loss_list = [], []
best_val_acc = 0.0


print(f"\n{'='*60}")
print(f"  PHASE 1 — Head only  (max {PHASE1_EPOCHS} epochs, early stop patience={EARLY_STOP_PATIENCE})")
print(f"{'='*60}")


optimizer1  = optim.Adam(model.classifier.parameters(),
                         lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
scheduler1  = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min',
                                                    factor=0.5, patience=3, verbose=True)
early_stop1 = EarlyStopping(patience=EARLY_STOP_PATIENCE)


start_time = time.time()


for epoch in range(PHASE1_EPOCHS):
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer1, train=True)
    vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, train=False)


    train_acc_list.append(tr_acc);   val_acc_list.append(vl_acc)
    train_loss_list.append(tr_loss); val_loss_list.append(vl_loss)
    total_epochs_run += 1


    # Save best model
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save({"model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "val_acc": best_val_acc}, SAVE_BEST_PATH)
        tag = "  ← best saved"
    else:
        tag = ""


    overfit_gap = tr_acc - vl_acc
    print(f"Epoch {epoch+1:02d}/{PHASE1_EPOCHS}  "
          f"TrainLoss: {tr_loss:.4f}  TrainAcc: {tr_acc:.2f}%  "
          f"ValLoss: {vl_loss:.4f}  ValAcc: {vl_acc:.2f}%  "
          f"Gap: {overfit_gap:+.2f}%{tag}")


    scheduler1.step(vl_loss)
    if early_stop1.step(vl_loss):
        print(f"  [Early Stop] Val loss didn't improve for {EARLY_STOP_PATIENCE} epochs. Stopping Phase 1.")
        break


phase1_end = total_epochs_run   # mark boundary for plot



#  STEP 6 — PHASE 2: UNFREEZE LAST 3 INVERTEDRESIDUAL BLOCKS

print(f"\n{'='*60}")
print(f"  PHASE 2 — Unfreeze last 3 blocks  (max {PHASE2_EPOCHS} epochs)")
print(f"{'='*60}")


for block in model.features[-3:]:
    for param in block.parameters():
        param.requires_grad = True


trainable2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params (Phase 2): {trainable2:,} / {total_params:,}")


optimizer2 = optim.Adam([
    {"params": model.features[-3:].parameters(), "lr": LR_FINETUNE, "weight_decay": WEIGHT_DECAY},
    {"params": model.classifier.parameters(),    "lr": LR_FINETUNE, "weight_decay": WEIGHT_DECAY},
])
scheduler2  = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min',
                                                    factor=0.5, patience=3, verbose=True)
early_stop2 = EarlyStopping(patience=EARLY_STOP_PATIENCE)


for epoch in range(PHASE2_EPOCHS):
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer2, train=True)
    vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, train=False)


    train_acc_list.append(tr_acc);   val_acc_list.append(vl_acc)
    train_loss_list.append(tr_loss); val_loss_list.append(vl_loss)
    total_epochs_run += 1


    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save({"model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "val_acc": best_val_acc}, SAVE_BEST_PATH)
        tag = "  ← best saved"
    else:
        tag = ""


    overfit_gap = tr_acc - vl_acc
    print(f"Epoch {epoch+1+phase1_end:02d}/{phase1_end+PHASE2_EPOCHS}  "
          f"TrainLoss: {tr_loss:.4f}  TrainAcc: {tr_acc:.2f}%  "
          f"ValLoss: {vl_loss:.4f}  ValAcc: {vl_acc:.2f}%  "
          f"Gap: {overfit_gap:+.2f}%{tag}")


    scheduler2.step(vl_loss)
    if early_stop2.step(vl_loss):
        print(f"  [Early Stop] Val loss didn't improve for {EARLY_STOP_PATIENCE} epochs. Stopping Phase 2.")
        break


end_time = time.time()
print(f"\nTotal Training Time: {end_time - start_time:.2f} seconds")
print(f"Best Val Accuracy  : {best_val_acc:.2f}%  (saved to {SAVE_BEST_PATH})")



#  STEP 7 — FINAL EVALUATION (using best saved checkpoint)

print(f"\n{'='*60}")
print(f"  FINAL EVALUATION — loading best checkpoint")
print(f"{'='*60}")


best_ckpt = torch.load(SAVE_BEST_PATH, map_location=device, weights_only=False)
model.load_state_dict(best_ckpt["model_state_dict"])
model.eval()


all_preds, all_labels_eval = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels_eval.extend(labels.numpy())


final_acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels_eval)) / len(all_labels_eval)
print(f"Final Val Accuracy : {final_acc:.2f}%\n")
print("Classification Report:")
print(classification_report(all_labels_eval, all_preds, target_names=CLASS_NAMES))


cm = confusion_matrix(all_labels_eval, all_preds)
print("Confusion Matrix:")
print(cm)



#  STEP 8 — OVERFITTING ANALYSIS

print(f"\n{'='*60}")
print(f"  OVERFITTING ANALYSIS")
print(f"{'='*60}")


final_train_acc = train_acc_list[-1]
final_val_acc   = val_acc_list[-1]
gap             = final_train_acc - final_val_acc


overfit_prob = round(min(1.0, max(0.0, gap / max(final_train_acc, 1e-6))), 2)
underfit_prob = round(max(0.0, (70 - final_train_acc) / 70), 2)
goodfit_prob  = round(max(0.0, 1 - max(overfit_prob, underfit_prob)), 2)


print(f"  Final Train Acc   : {final_train_acc:.2f}%")
print(f"  Final Val Acc     : {final_val_acc:.2f}%")
print(f"  Gap (Train-Val)   : {gap:+.2f}%")
print(f"  Overfit Prob      : {overfit_prob:.2f}")
print(f"  Underfit Prob     : {underfit_prob:.2f}")
print(f"  Good Fit Prob     : {goodfit_prob:.2f}")


if overfit_prob > 0.6:
    verdict = "OVERFITTING — consider more dropout or data augmentation"
elif underfit_prob > 0.6:
    verdict = "UNDERFITTING — train longer or unfreeze more layers"
else:
    verdict = "GOOD FIT"
print(f"  Verdict           : {verdict}")



#  STEP 9 — PLOTS (4 subplots in one figure)

epochs_range = range(1, total_epochs_run + 1)
p2_start     = phase1_end + 0.5   # vertical line position for Phase 2


fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("MobileNetV2 Transfer Learning — Sugarcane Disease Classification\n"
             f"({NUM_CLASSES} Classes  |  All Data  |  Best Val Acc: {best_val_acc:.2f}%)",
             fontsize=13, fontweight='bold')


# --- Plot 1: Accuracy ---
ax = axes[0, 0]
ax.plot(epochs_range, train_acc_list, label="Train Accuracy", color="steelblue",  linewidth=2)
ax.plot(epochs_range, val_acc_list,   label="Val Accuracy",   color="darkorange", linewidth=2)
ax.axvline(p2_start, color='gray', linestyle='--', linewidth=1.2, label="Phase 2 Start")
ax.set_title("Accuracy vs Epochs")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))
ax.legend(); ax.grid(alpha=0.3)


# --- Plot 2: Loss ---
ax = axes[0, 1]
ax.plot(epochs_range, train_loss_list, label="Train Loss", color="steelblue",  linewidth=2)
ax.plot(epochs_range, val_loss_list,   label="Val Loss",   color="darkorange", linewidth=2)
ax.axvline(p2_start, color='gray', linestyle='--', linewidth=1.2, label="Phase 2 Start")
ax.set_title("Loss vs Epochs")
ax.set_xlabel("Epoch"); ax.set_ylabel("Average Loss")
ax.legend(); ax.grid(alpha=0.3)


# --- Plot 3: Overfit Gap ---
ax = axes[1, 0]
gap_list = [t - v for t, v in zip(train_acc_list, val_acc_list)]
colors   = ["red" if g > 10 else "orange" if g > 5 else "green" for g in gap_list]
ax.bar(epochs_range, gap_list, color=colors)
ax.axvline(p2_start, color='gray', linestyle='--', linewidth=1.2, label="Phase 2 Start")
ax.axhline(10, color='red',    linestyle=':', linewidth=1.2, label="Overfit threshold (10%)")
ax.axhline(5,  color='orange', linestyle=':', linewidth=1.2, label="Warning threshold (5%)")
ax.axhline(0,  color='black',  linewidth=0.8)
ax.set_title("Train−Val Accuracy Gap (Overfitting Monitor)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Gap (%)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)


# --- Plot 4: Fit Probability Bar ---
ax = axes[1, 1]
fit_labels = ["Overfitting", "Underfitting", "Good Fit"]
fit_values = [overfit_prob, underfit_prob, goodfit_prob]
bar_colors = ["tomato", "royalblue", "mediumseagreen"]
bars = ax.bar(fit_labels, fit_values, color=bar_colors, width=0.4)
ax.set_ylim(0, 1.15)
ax.set_title("Model Fit Probability (Final Epoch)")
ax.set_ylabel("Probability")
for bar, val in zip(bars, fit_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.03,
            f"{val:.2f}", ha='center', fontweight='bold')
ax.grid(axis='y', alpha=0.3)


plt.tight_layout()
plt.savefig("mobilenetv2_training_report.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: mobilenetv2_training_report.png")



#  STEP 10 — SAVE FINAL MODEL

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names":      CLASS_NAMES,
    "num_classes":      NUM_CLASSES,
    "final_val_acc":    final_acc,
    "best_val_acc":     best_val_acc,
    "architecture":     "MobileNetV2_TransferLearning"
}, SAVE_FINAL_PATH)


print(f"\nFinal model saved : {SAVE_FINAL_PATH}")
print(f"Best checkpoint   : {SAVE_BEST_PATH}")
print(f"Classes trained   : {NUM_CLASSES}  →  {CLASS_NAMES}")

