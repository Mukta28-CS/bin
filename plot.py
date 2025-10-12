import matplotlib.pyplot as plt
import torch

# -----------------------------
# Load your saved training history
# -----------------------------
history_file = "checkpoints/training_history_epoch_50.pt"
history = torch.load(history_file)

train_losses = history['train_losses']
val_losses = history['val_losses']
train_accuracies = history['train_accuracies']
val_accuracies = history['val_accuracies']

# -----------------------------
# Determine available range
# -----------------------------
total_epochs = len(train_losses)     # how many epochs of data are available
start_epoch = 50                     # we resumed after epoch 50
epochs = list(range(1, total_epochs + 1))  # 1-based full epoch list

# Slice only epochs after 50 if you resumed training
if total_epochs > start_epoch:
    epochs = epochs[start_epoch:]
    train_losses = train_losses[start_epoch:]
    val_losses = val_losses[start_epoch:]
    train_accuracies = train_accuracies[start_epoch:]
    val_accuracies = val_accuracies[start_epoch:]

# -----------------------------
# Step points every 10 epochs
# -----------------------------
step = 10
epochs_step = epochs[::step]
train_losses_step = train_losses[::step]
val_losses_step = val_losses[::step]
train_acc_step = train_accuracies[::step]
val_acc_step = val_accuracies[::step]

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', color='blue', alpha=0.6)
plt.plot(epochs, val_losses, label='Validation Loss', color='red', alpha=0.6)
plt.scatter(epochs_step, train_losses_step, color='blue')
plt.scatter(epochs_step, val_losses_step, color='red')
plt.xlabel(f'Epochs ({start_epoch+1}–{start_epoch+len(epochs)})')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy', color='green', alpha=0.6)
plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange', alpha=0.6)
plt.scatter(epochs_step, train_acc_step, color='green')
plt.scatter(epochs_step, val_acc_step, color='orange')
plt.xlabel(f'Epochs ({start_epoch+1}–{start_epoch+len(epochs)})')
plt.ylabel('Accuracy (%)')
plt.title('Train vs Validation Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("training_results_50_onwards.png", dpi=300)
plt.show()

print("📊 Graph generated and saved as training_results_50_onwards.png")
