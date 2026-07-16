from PIL import Image
import os
import string

# =========================================================
# CONFIG
# =========================================================

# Ukuran image hasil resize
BASE_WIDTH = 100

# Maksimum dataset
TRAINING_LIMIT = 1000
VALIDATION_LIMIT = 200

# Semua class:
# 0-9 + A-Z
CLASSES = [str(i) for i in range(10)] + list(string.ascii_uppercase)

# =========================================================
# RESIZE FUNCTION
# =========================================================

def resize_image(image_path):

    try:

        img = Image.open(image_path)

        # Convert ke grayscale
        img = img.convert("L")

        # Hitung aspect ratio
        w_percent = BASE_WIDTH / float(img.size[0])

        h_size = int(float(img.size[1]) * float(w_percent))

        # Resize
        img = img.resize((BASE_WIDTH, h_size), Image.LANCZOS)

        # Save overwrite
        img.save(image_path)

        return True

    except Exception as e:

        print(f"[ERROR] {image_path}")
        print(e)

        return False

# =========================================================
# PROCESS DATASET
# =========================================================

def process_dataset(mode, limit):

    total_success = 0
    total_failed = 0

    print("\n" + "=" * 60)
    print(f"PROCESSING {mode.upper()} DATASET")
    print("=" * 60)

    for class_name in CLASSES:

        folder_path = f"Dataset/{mode}/{class_name}"

        # Skip jika folder tidak ada
        if not os.path.exists(folder_path):

            print(f"[SKIP] Folder tidak ditemukan: {folder_path}")
            continue

        print(f"\n[CLASS] {class_name}")

        success = 0
        failed = 0

        # Ambil semua file png
        files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith(".png")
        ])

        # Batasi jumlah file
        files = files[:limit]

        for file_name in files:

            image_path = os.path.join(folder_path, file_name)

            status = resize_image(image_path)

            if status:
                success += 1
            else:
                failed += 1

        total_success += success
        total_failed += failed

        print(f"   Success : {success}")
        print(f"   Failed  : {failed}")

    print("\n" + "=" * 60)
    print(f"{mode.upper()} FINISHED")
    print("=" * 60)
    print(f"TOTAL SUCCESS : {total_success}")
    print(f"TOTAL FAILED  : {total_failed}")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # TRAINING
    process_dataset(
        mode="Training",
        limit=TRAINING_LIMIT
    )

    # VALIDATION
    process_dataset(
        mode="Validation",
        limit=VALIDATION_LIMIT
    )

    print("\nALL PROCESS COMPLETED!")