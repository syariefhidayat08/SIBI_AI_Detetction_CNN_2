import os
import cv2
import time
import string
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Dense, Dropout,
    Flatten, BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.utils import to_categorical

# =========================================================
# CONFIG
# =========================================================

IMG_WIDTH = 100
IMG_HEIGHT = 89
CHANNEL = 1

DATASET_PATH = "Dataset/Training"

CLASSES = [str(i) for i in range(10)] + list(string.ascii_uppercase)

# =========================================================
# LOAD DATASET
# =========================================================

images = []
labels = []
valid_classes = []

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

for class_index, class_name in enumerate(CLASSES):

    folder_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.exists(folder_path):
        continue

    files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

    if len(files) < 2:
        continue

    valid_classes.append(class_name)

    for file_name in files:

        image_path = os.path.join(folder_path, file_name)

        image = cv2.imread(image_path)
        if image is None:
            continue

        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = image / 255.0
        image = image.reshape(IMG_HEIGHT, IMG_WIDTH, CHANNEL)

        images.append(image)
        labels.append(len(valid_classes) - 1)

images = np.array(images, dtype=np.float32)
labels = np.array(labels)

NUM_CLASSES = len(valid_classes)

labels = to_categorical(labels, num_classes=NUM_CLASSES)

images, labels = shuffle(images, labels, random_state=42)

# =========================================================
# SPLIT DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    images, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Training :", len(X_train))
print("Testing  :", len(X_test))
print("Classes  :", NUM_CLASSES)

# =========================================================
# MODEL CNN
# =========================================================

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', padding='same',
                 input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNEL)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.30))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================================================
# CALLBACK
# =========================================================

os.makedirs("TrainedModelCNN", exist_ok=True)

checkpoint = ModelCheckpoint(
    "TrainedModelCNN/best_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1
)

# =========================================================
# TRAINING
# =========================================================

start_train = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

training_time = time.time() - start_train

# =========================================================
# SAVE MODEL
# =========================================================

model.save("TrainedModelCNN/FINAL_MODEL.keras")

# =========================================================
# EVALUATION
# =========================================================

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

precision = precision_score(y_true, y_pred_classes, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred_classes, average='weighted', zero_division=0)

cm = confusion_matrix(y_true, y_pred_classes)

# =========================================================
# INFERENCE SPEED
# =========================================================

sample = X_test[0:1]

start = time.time()
for _ in range(100):
    model.predict(sample, verbose=0)
end = time.time()

avg_infer = (end - start) / 100
fps = 1 / avg_infer

# =========================================================
# PRINT RESULT
# =========================================================

print("\nRESULT")
print("="*50)
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("FPS      :", fps)

# =========================================================
# SAVE TEXT RESULT
# =========================================================

with open("TrainedModelCNN/results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"FPS: {fps}\n")
    f.write(f"Training Time: {training_time}\n")

# =========================================================
# GRAPH 1 - ACCURACY LOSS
# =========================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["Train","Val"])

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["Train","Val"])

plt.savefig("TrainedModelCNN/accuracy_loss.png")
plt.show()

# =========================================================
# GRAPH 2 - CONFUSION MATRIX
# =========================================================

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=valid_classes,
            yticklabels=valid_classes)

plt.title("Confusion Matrix")
plt.savefig("TrainedModelCNN/confusion_matrix.png")
plt.show()

# =========================================================
# GRAPH 3 - CLASS ACCURACY
# =========================================================

class_acc = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(12,5))
plt.bar(valid_classes, class_acc)
plt.title("Class Accuracy")
plt.xticks(rotation=90)
plt.savefig("TrainedModelCNN/class_accuracy.png")
plt.show()