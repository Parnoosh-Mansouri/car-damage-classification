import matplotlib.pyplot as plt


# Fine-tuning training history
epochs = [1, 2, 3, 4, 5]

train_loss = [
    0.3839,
    0.2391,
    0.1955,
    0.1595,
    0.1366
]

val_loss = [
    0.2687,
    0.2450,
    0.2359,
    0.2367,
    0.2510
]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Fine-tuning Loss")
plt.legend()
plt.grid(True)

plt.savefig("results/loss_curve.png", dpi=300)
plt.close()


# Model comparison
models = [
    "Baseline",
    "Augmentation",
    "Fine-tuning"
]

f1_scores = [
    0.67,
    0.76,
    0.80
]

plt.figure(figsize=(8, 5))
plt.bar(models, f1_scores)

plt.xlabel("Model")
plt.ylabel("Macro F1")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.grid(axis="y")

plt.savefig(
    "results/model_comparison.png",
    dpi=300
)

plt.close()

print("Graphs saved successfully!")