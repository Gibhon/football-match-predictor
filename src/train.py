import torch


def train(model, n_epoch, data_loader, loss_fn, optimizer, scheduler, device, filepath):
    history = {"loss": [], "Accuracy": []}
    min_loss_epoch = 0
    min_loss = 100
    for epoch in range(n_epoch):
        total_loss = 0
        n_accurate = 0
        n_total = 0
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction_dim3 = model.forward(x_batch)
            prediction_final = torch.argmax(prediction_dim3, dim=1)
            loss = loss_fn(prediction_dim3, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_accurate += (prediction_final == y_batch).sum().item()
            n_total += prediction_final.size(0)
        scheduler.step()
        epoch_loss = total_loss / len(data_loader)
        epoch_acc = (n_accurate / n_total) * 100
        history["loss"].append(epoch_loss)
        history["Accuracy"].append(epoch_acc)
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            min_loss_epoch = epoch
    torch.save(model.state_dict(), filepath)
    return (min_loss_epoch, min_loss, history)


def val(model, data_loader, loss_fn, device):
    total_loss = 0
    n_total = 0
    n_accurate = 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction_dim3 = model(x_batch)
            prediction = torch.argmax(prediction_dim3, dim=1)
            loss = loss_fn(prediction_dim3, y_batch)

            total_loss += loss.item()
            n_total += prediction.size(0)
            n_accurate += (prediction == y_batch).sum().item()

    return (total_loss / len(data_loader), (n_accurate / n_total) * 100)


def test(model, data_loader, device, filepath):
    model.load_state_dict(torch.load(filepath))
    n_total = 0
    n_accurate = 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction_dim3 = model(x_batch)
            prediction = torch.argmax(prediction_dim3, dim=1)
            n_accurate += (prediction == y_batch).sum().item()

    return (n_accurate / n_total) * 100
