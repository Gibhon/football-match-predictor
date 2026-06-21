import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
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

    weights = torch.tensor([1.2157, 0.7619, 1.1561]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        football_model.parameters(), weight_decay=1e-4, lr=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model_path = base_dir / "model.pth"

    min_loss_epoch, min_loss, history = train.train(
        model=football_model,
        n_epoch=10,
        data_loader=train_data_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        filepath=model_path,
    )
    val_loss, val_acc = train.val(football_model, val_data_loader, loss_fn, device)

    print(f"Min Loss : {min_loss}")
    print(f"Min Loss Epoch : {min_loss_epoch}")
    print(history)
    print(
        "\n__________________________________________________________________________"
    )
    print(f"Val Loss : {val_loss}")
    print(f"Val Acc : {val_acc}")
