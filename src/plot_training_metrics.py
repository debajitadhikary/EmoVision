import matplotlib.pyplot as plt
import json
import numpy as np

# Load training history from JSON file
with open('history.json', 'r') as f:
    history = json.load(f)

# Get the total number of epochs from the length of accuracy history
total_epochs = len(history['accuracy'])
print(f"Total epochs: {total_epochs}")

# Generate epoch numbers to be used for x-axis
epochs = np.arange(1, total_epochs + 1)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, history['accuracy'], label='Train Accuracy')
plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(epochs)  # Ensure all epochs are shown on the x-axis
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(epochs, history['loss'], label='Train Loss')
plt.plot(epochs, history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)  # Ensure all epochs are shown on the x-axis
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
