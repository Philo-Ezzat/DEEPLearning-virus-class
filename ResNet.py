import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Define a residual block
def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    shortcut = x

    x = Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    x = BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x

# Build ResNet-like model from scratch
def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

    # Add residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 512, stride=2)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model

# Model configuration
input_shape = (224, 224, 3)
num_classes = 22  # Replace '3' with the number of classes

model = build_resnet(input_shape, num_classes)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Data preparation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the trained model
model.save('virus_classification_resnet_scratch.h5')

# Evaluation Metrics
# Get true labels and predictions
val_labels = validation_generator.classes
val_predictions = model.predict(validation_generator)
val_pred_classes = np.argmax(val_predictions, axis=1)

# Confusion Matrix
cm = tf.math.confusion_matrix(labels=val_labels, predictions=val_pred_classes).numpy()
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(len(validation_generator.class_indices)), validation_generator.class_indices.keys(), rotation=45)
plt.yticks(range(len(validation_generator.class_indices)), validation_generator.class_indices.keys())
plt.show()

# Calculate metrics manually
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
precision = np.diag(cm) / np.sum(cm, axis=0)
recall = np.diag(cm) / np.sum(cm, axis=1)
f1_scores = 2 * (precision * recall) / (precision + recall)

# Average metrics
precision_avg = np.nanmean(precision)
recall_avg = np.nanmean(recall)
f1_avg = np.nanmean(f1_scores)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision_avg:.2f}")
print(f"Recall: {recall_avg:.2f}")
print(f"F1 Score: {f1_avg:.2f}")

# ROC Curve and AUC
val_labels_onehot = tf.keras.utils.to_categorical(val_labels, num_classes=len(validation_generator.class_indices))
fpr, tpr, roc_auc = {}, {}, {}

for i in range(len(validation_generator.class_indices)):
    fpr[i], tpr[i], _ = tf.metrics.roc_curve(val_labels_onehot[:, i], val_predictions[:, i])
    roc_auc[i] = tf.metrics.auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']

for i, color in zip(range(len(validation_generator.class_indices)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
