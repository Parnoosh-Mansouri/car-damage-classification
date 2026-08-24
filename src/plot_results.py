import matplotlib.pyplot as plt


# Fine-tuning loss history
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

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Train Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Fine-tuning Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Final performance on the unseen test set
plt.figure(figsize=(6, 5))

plt.bar(
    ["Final Fine-tuned Model"],
    [0.76]
)

plt.ylabel("Macro F1")
plt.title("Final Test Performance")
plt.ylim(0, 1)
plt.grid(axis="y")

plt.savefig(
    "results/final_test_f1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Graphs saved successfully!")