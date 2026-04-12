from model_def import ResNet, ResidualBlock
from preprocess import preprocess
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train():
    x_tensor, y_tensor, mapping = preprocess()

    # train test split
    x_train, x_, y_train, y_ = train_test_split(
        x_tensor,
        y_tensor,
        test_size=0.15,
        random_state=90,
        stratify=y_tensor,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_,
        y_,
        test_size=0.33,
        random_state=90,
        stratify=y_,
    )

    # put it to train val test loader
    traindataset = TensorDataset(x_train, y_train)
    trainloader = DataLoader(traindataset, batch_size=4, shuffle=True)

    valdataset = TensorDataset(x_val, y_val)
    valloader = DataLoader(valdataset, batch_size=4, shuffle=False)

    testdataset = TensorDataset(x_test, y_test)
    testloader = DataLoader(testdataset, batch_size=4, shuffle=False)

    max_val_acc = -1
    model = ResNet()

    # training
    model = model.to("cpu")
    validation_accuracies = []
    training_loss = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 0.0001)
    device = "cpu"
    print("Starting Training...")

    epo = 20
    for epoch in range(epo):
        model.train()
        running_loss = 0.0

        for x, y in trainloader:
            x, y = x.to("cpu"), y.to("cpu")

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(trainloader)
        training_loss.append(avg_train_loss)

        print(f"Epoch [{epoch+1}/{epo}], Loss: {avg_train_loss:.4f}")

        # Evaluate model on validation data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in valloader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        val_accuracy = 100 * correct / total
        max_val_acc = max(val_accuracy, max_val_acc)
        if val_accuracy == max_val_acc:
            print("Saving at accuracy", val_accuracy)
            torch.save(model.state_dict(), "model.pth")
        print(f"Val Accuracy: {val_accuracy:.2f}%\n")
        validation_accuracies.append(val_accuracy)
    # final evaluation on test data
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    test_accuracy = 100 * correct / total
    print("Test Acc", test_accuracy)
    torch.save(model.state_dict(), "model.pth")
