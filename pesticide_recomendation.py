# pesticide_recomendation.py
# ============================================================
# PEST CLASSIFICATION using MobileNetV2 Transfer Learning
# ============================================================
# Previous model: Small 3-layer CNN at 64x64 → predicted beetle
# for everything because it couldn't learn pest features.
#
# New model: MobileNetV2 (pretrained on ImageNet) fine-tuned
# for pest classification at 128x128. Much stronger feature
# extraction = actually learns what each pest looks like.
# ============================================================

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import json
import os

# ╔════════════════════════════════════════════════════════════╗
# ║                    CONFIG                                  ║
# ╚════════════════════════════════════════════════════════════╝

TRAIN_DIR = "Data/train"
TEST_DIR  = "Data/test"
IMG_SIZE  = 128          # Bigger than 64 = better features
BATCH_SIZE = 32
EPOCHS = 30              # With early stopping, won't always run all 30
MODEL_SAVE_PATH = "Trained_model.h5"

# ╔════════════════════════════════════════════════════════════╗
# ║                    DATA GENERATORS                         ║
# ╚════════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("PEST MODEL TRAINING — MobileNetV2 Transfer Learning")
print("="*60)

# Strong augmentation to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_set = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

NUM_CLASSES = training_set.num_classes
CLASS_NAMES = list(training_set.class_indices.keys())

print(f"\n📊 Dataset Info:")
print(f"  Training samples: {training_set.samples}")
print(f"  Test samples:     {test_set.samples}")
print(f"  Number of classes: {NUM_CLASSES}")
print(f"  Classes: {CLASS_NAMES}")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")

# ╔════════════════════════════════════════════════════════════╗
# ║                    MODEL — MobileNetV2                     ║
# ╚════════════════════════════════════════════════════════════╝

# Load MobileNetV2 pretrained on ImageNet (without top classification layer)
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers (use pretrained features as-is)
base_model.trainable = False

# Add custom classification head
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

classifier = Model(inputs, outputs)

classifier.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n📐 Model Architecture:")
print(f"  Base: MobileNetV2 (frozen, pretrained on ImageNet)")
print(f"  Head: GAP → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense({NUM_CLASSES})")
print(f"  Total params: {classifier.count_params():,}")
trainable = sum([w.numpy().size for w in classifier.trainable_weights])
print(f"  Trainable params: {trainable:,}")

# ╔════════════════════════════════════════════════════════════╗
# ║                    CALLBACKS                               ║
# ╚════════════════════════════════════════════════════════════╝

callbacks = [
    # Stop training if validation accuracy doesn't improve for 5 epochs
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when validation loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    # Save best model during training
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ╔════════════════════════════════════════════════════════════╗
# ║                    PHASE 1: TRAIN HEAD ONLY                ║
# ╚════════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("PHASE 1: Training classification head (base frozen)")
print("="*60 + "\n")

history1 = classifier.fit(
    training_set,
    epochs=15,
    validation_data=test_set,
    callbacks=callbacks
)

# ╔════════════════════════════════════════════════════════════╗
# ║                    PHASE 2: FINE-TUNE TOP LAYERS           ║
# ╚════════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("PHASE 2: Fine-tuning top layers of MobileNetV2")
print("="*60 + "\n")

# Unfreeze the last 30 layers of MobileNetV2 for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
classifier.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_trainable = sum([w.numpy().size for w in classifier.trainable_weights])
print(f"  Trainable params after unfreezing: {fine_tune_trainable:,}")

history2 = classifier.fit(
    training_set,
    epochs=EPOCHS,
    initial_epoch=len(history1.history['accuracy']),
    validation_data=test_set,
    callbacks=callbacks
)

# ╔════════════════════════════════════════════════════════════╗
# ║                    EVALUATE                                ║
# ╚════════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("EVALUATION")
print("="*60)

loss, accuracy = classifier.evaluate(test_set, verbose=0)
print(f"\n  ✅ Test Accuracy: {accuracy*100:.2f}%")
print(f"  📉 Test Loss: {loss:.4f}")

# ╔════════════════════════════════════════════════════════════╗
# ║                    SAVE MODEL + CLASS NAMES                ║
# ╚════════════════════════════════════════════════════════════╝

# Save model (best was already saved by ModelCheckpoint)
classifier.save(MODEL_SAVE_PATH)
print(f"\n  ✅ Model saved as: {MODEL_SAVE_PATH}")

# Save class names mapping (important for prediction)
class_info = {
    "class_names": CLASS_NAMES,
    "class_indices": training_set.class_indices,
    "img_size": IMG_SIZE,
    "num_classes": NUM_CLASSES
}

with open("pest_class_names.json", "w") as f:
    json.dump(class_info, f, indent=2)
print(f"  ✅ Class names saved as: pest_class_names.json")

# ╔════════════════════════════════════════════════════════════╗
# ║                    PLOTS                                   ║
# ╚════════════════════════════════════════════════════════════╝

# Combine histories
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss_hist = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

epochs_range = range(1, len(acc) + 1)

# Accuracy plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.axvline(x=len(history1.history['accuracy']), color='gray',
            linestyle='--', label='Fine-tune starts')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Pest Model — Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss_hist, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.axvline(x=len(history1.history['loss']), color='gray',
            linestyle='--', label='Fine-tune starts')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Pest Model — Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("pest_training_plots.png", dpi=150)
plt.show()

print("\n🎉 TRAINING COMPLETE!")
print(f"   Final Test Accuracy: {accuracy*100:.2f}%")
print(f"   Model: {MODEL_SAVE_PATH}")
print(f"   Run your app.py now — the pest model will load automatically.")