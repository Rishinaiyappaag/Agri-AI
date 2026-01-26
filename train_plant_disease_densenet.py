import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            print(f"⚠️ Skipping corrupted image: {self.samples[index][0]}")
            new_index = (index + 1) % len(self.samples)
            return super().__getitem__(new_index)

# =========================
# MAIN GUARD (IMPORTANT FOR WINDOWS)
# =========================
if __name__ == "__main__":

    # =========================
    # CONFIG
    # =========================
    DATASET_DIR = r"C:\MCA FINAL PROJECT\Agri AI\plant_disease_converted"
    MODEL_SAVE_PATH = "plant_disease_densenet.pth"

    BATCH_SIZE = 4
    EPOCHS = 2
    LEARNING_RATE = 0.0001

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # =========================
    # TRANSFORMS
    # =========================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # =========================
    # DATASET
    # =========================
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "validation")

    train_dataset = SafeImageFolder(train_dir, transform=train_transform)
    val_dataset = SafeImageFolder(val_dir, transform=val_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Total classes:", num_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # =========================
    # MODEL
    # =========================
    model = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )

    model.classifier = nn.Linear(
        model.classifier.in_features,
        num_classes
    )

    model = model.to(DEVICE)

    # =========================
    # TRAINING SETUP
    # =========================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 📊 HISTORY (FOR GRAPHS)
    train_losses = []
    val_accuracies = []

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        print(f"\n🔁 Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"✅ Training Loss: {avg_loss:.4f}")

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        print(f"📊 Validation Accuracy: {accuracy:.2f}%")

    # =========================
    # SAVE MODEL
    # =========================
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names
        },
        MODEL_SAVE_PATH
    )

    print("\n🎉 TRAINING COMPLETE")
    print("✅ Model saved as:", MODEL_SAVE_PATH)

    # =========================
    # 📈 PLOTS
    # =========================
    epochs_range = range(1, EPOCHS + 1)

    # 🔵 LOSS GRAPH
    plt.figure()
    plt.plot(epochs_range, train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss plant disease")
    plt.grid(True)
    plt.savefig("training_loss plant disease.png")
    plt.show()

    # 🟢 ACCURACY GRAPH
    plt.figure()
    plt.plot(epochs_range, val_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy plant disease")
    plt.grid(True)
    plt.savefig("validation_accuracy plant disease.png")
    plt.show()
