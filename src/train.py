import torch


def train(model, n_epoch, data_loader, optimizer, loss_fn, scheduler, device, filepath):
    n_accurate = 0
    n_total = 0
    history = {"Accuracy": [], "Loss": []}
    for epoch in range(n_epoch):
        model.train()
        loss = 0
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(x_batch)
            running_loss = loss_fn(predictions, y_batch)

            loss.backward()
            optimizer.step()

            loss += loss.item() * x_batch.size(0)
            n_total += predictions.size(0)
            n_accurate += (predictions == y_batch).sum().item()
        scheduler.step()

    torch.save(model.state_dict(), filepath)


def val(model, data_loader, loss_fn, device):
    total_loss = 0
    n_total = 0
    n_accurate = 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction = model(x_batch)
            loss = loss_fn(prediction, y_batch)

            total_loss += loss.item()
            n_total += prediction.size(0)

    print(f"Loss : {total_loss / len(data_loader)}")
    print(f"Accuracy: {(n_accurate / n_total) * 100}%")


def test_model(model, test_set_loader, device, filepath):
    model.load_state_dict(torch.load(filepath))
    n_total = 0
    n_accurate = 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_set_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction = model(x_batch)

            n_total += prediction.size(0)
            _, prediction_label = torch.max(prediction, 1)
            n_accurate += (prediction_label == y_batch).sum().item()

    print(f"Accuracy: {n_accurate / n_total}")
