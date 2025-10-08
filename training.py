import matplotlib.pyplot as plt

# -----------------------------
# Epoch numbers
# -----------------------------
epochs = list(range(31, 51))  # 31 to 50

# -----------------------------
# Copy the values from your logs
# -----------------------------
train_losses = [0.1235, 0.0362, 0.0684, 0.0593, 0.0342,
                0.1675, 0.1850, 0.1655, 0.1454, 0.0951,
                0.0467, 0.0369, 0.1141, 0.2091, 0.0813,
                0.0375, 0.0580, 0.0694, 0.0619, 0.0892]

val_losses = [0.2257, 0.2422, 0.1304, 0.1197, 0.2488,
              0.2855, 0.3616, 0.2146, 0.1791, 0.2388,
              0.1842, 0.2207, 0.3037, 0.1747, 0.1724,
              0.2392, 0.1633, 0.2493, 0.1638, 0.2226]

train_accuracies = [95.70, 98.96, 97.92, 97.63, 98.96,
                    93.77, 96.88, 94.21, 95.40, 98.22,
                    97.92, 98.96, 98.66, 92.43, 97.48,
                    98.52, 98.37, 97.63, 98.96, 96.29]

val_accuracies = [92.35, 93.53, 95.88, 95.29, 92.94,
                  90.59, 87.06, 94.12, 92.94, 92.35,
                  91.76, 91.76, 86.47, 92.94, 92.94,
                  93.53, 91.76, 91.18, 94.12, 93.53]

# -----------------------------
# Plot Loss and Accuracy
# -----------------------------
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'o-', label='Train Loss')
plt.plot(epochs, val_losses, 'o-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss (Epochs 31-50)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'o-', label='Train Accuracy')
plt.plot(epochs, val_accuracies, 'o-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Train vs Validation Accuracy (Epochs 31-50)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
