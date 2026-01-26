import os
import shutil

SOURCE_DIR = r"C:\MCA FINAL PROJECT\Agri AI\plant disease"
TARGET_DIR = r"C:\MCA FINAL PROJECT\Agri AI\plant_disease_converted"

splits = ["train", "validation", "test"]

print("🔍 SOURCE DIR EXISTS:", os.path.exists(SOURCE_DIR))
print("📂 SOURCE DIR:", SOURCE_DIR)

for split in splits:
    src_split = os.path.join(SOURCE_DIR, split)
    tgt_split = os.path.join(TARGET_DIR, split)

    print(f"\n➡️ Processing split: {split}")
    print("   Source exists:", os.path.exists(src_split))

    if not os.path.exists(src_split):
        print("   ❌ Skipping (folder not found)")
        continue

    os.makedirs(tgt_split, exist_ok=True)

    for plant in os.listdir(src_split):
        plant_path = os.path.join(src_split, plant)

        if not os.path.isdir(plant_path):
            continue

        print(f"   🌱 Plant: {plant}")

        for disease in os.listdir(plant_path):
            disease_path = os.path.join(plant_path, disease)

            if not os.path.isdir(disease_path):
                continue

            label = f"{plant}___{disease}"
            label_dir = os.path.join(tgt_split, label)
            os.makedirs(label_dir, exist_ok=True)

            images = os.listdir(disease_path)
            print(f"      🦠 Disease: {disease} ({len(images)} images)")

            for img in images:
                src_img = os.path.join(disease_path, img)
                dst_img = os.path.join(label_dir, img)

                if not os.path.exists(dst_img):
                    shutil.copy(src_img, dst_img)

print("\n✅ DATASET CONVERSION COMPLETE")
