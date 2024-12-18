import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

# Load the DenseNet121 model without the top layer
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(22, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data preparation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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

# Unfreeze the base model for fine-tuning
for layer in base_model.layers:
    layer.trainable = True

# Recompile the model for fine-tuning with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fine-tune the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the trained model
model.save('virus_classification_densenet.h5')

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
