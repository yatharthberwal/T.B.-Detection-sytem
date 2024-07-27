import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Load the saved model
model = load_model('tb_detector.keras')

# Define paths to your dataset for validation
val_dir = r'D:\project\A.I. in healthcare\dataset\validation'

# Data preprocessing and augmentation for validation data
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Important to keep the order for confusion matrix
)

# Generate predictions
predictions = model.predict(val_generator)

# Convert predictions to class labels
y_pred = (predictions > 0.8).astype(int)

# Get true classes
y_true = val_generator.classes

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)


# Define labels for the matrix
labels = ['Negative', 'Positive']

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

