import torch

def train_clean_model(model, criterion, optimizer, dataloader, device, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            inputs, labels = batch['img'].to(device), batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return model

def single_epoch(model, set, criterion, optimizer, dataloader, device):
    # Set is train/val. Will change the name to something more meaningful.

    if set == "train":
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            inputs, labels = batch['img'].to(device), batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)
    if set == "val":
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch['img'].to(device), batch['labels'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
        return running_loss / len(dataloader)
 
def test_model(model, dataloader, criterion, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch['img'].to(device), batch['labels'].to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")





