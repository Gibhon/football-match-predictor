import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import src.dataset as data_set
import src.model as model
import src.train as train
import torch
from torch import nn

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent

    train_features_np, train_labels_np, val_features_np, val_labels_np = (
        data_set.read_data()
    )

    train_dataset = data_set.FootballDataset(train_features_np, train_labels_np)
    val_dataset = data_set.FootballDataset(
        val_features_np, val_labels_np, train_dataset.mean, train_dataset.std
    )

    train_data_loader = data_set.load_data(train_dataset, True, num_workers=4)
    val_data_loader = data_set.load_data(val_dataset, False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    football_model = model.FootballPredictor(n_inputs=20).to(device)

    n_epoch = 10
    weights = torch.tensor([1.2157, 0.7619, 1.1561]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        football_model.parameters(), weight_decay=1e-4, lr=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model_path = base_dir / "model.pth"

    min_loss_epoch, min_loss, history = train.train(
        model=football_model,
        n_epoch=n_epoch,
        data_loader=train_data_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        filepath=model_path,
        val_data_loader=val_data_loader,
    )

    epochs = [i + 1 for i in range(n_epoch)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    ax1 = axs[0]
    ax2 = axs[1]

    ax1.plot(epochs, history["train_loss"], color="crimson", label="Train")
    ax1.plot(
        epochs,
        history["val_loss"],
        color="darkorange",
        label="Validation",
        linestyle="--",
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss VS Validation Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    ax2.plot(epochs, history["train_acc"], color="lightblue", label="Train")
    ax2.plot(
        epochs, history["val_acc"], color="black", label="Validation", linestyle="--"
    )
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy VS Validation Accuracy")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("model_performance.png", dpi=300, bbox_inches="tight")
    plt.show()
