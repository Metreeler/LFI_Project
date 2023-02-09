import torch.nn as nn
import torch


class MyNeuralNetwork(nn.Module):
    def __init__(self, name_model):
        super(MyNeuralNetwork, self).__init__()
        self.name_model = name_model
        if name_model == "model_1":
            self.layer1 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=784, out_features=2048),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.layer2 = nn.Sequential(
                nn.Linear(in_features=2048, out_features=2048),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.layer3 = nn.Sequential(
                nn.Linear(in_features=2048, out_features=26),
                nn.Softmax(1)
            )
        elif self.name_model == "model_2":
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )

            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )

            self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )

            self.layer4 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=1152, out_features=512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=512, out_features=26)
            )
        elif self.name_model == "model_3":
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding="same"),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )

            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )

            self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )

            self.layer4 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=2048, out_features=1024),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(in_features=1024, out_features=1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=1024, out_features=26),
                nn.Softmax(1)
            )
        elif self.name_model == "model_4":
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=0, stride=2),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )

            self.layer2 = nn.Sequential(
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"),
                nn.BatchNorm2d(64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )

            self.layer3 = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"),
                nn.BatchNorm2d(128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )

            self.layer4 = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same"),
                nn.BatchNorm2d(256),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding="same")
            )

            self.layer5 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=256, out_features=256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=256, out_features=26)
            )
        else:
            pass

    def forward(self, x):
        if self.name_model == "model_1":
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.name_model == "model_2":
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        elif self.name_model == "model_3":
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        if self.name_model == "model_4":
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
        else:
            pass
        return x


def training(data_loader, validation_data_loader, epochs, optimizer, model, criterion):
    train_summary = ""
    train_loss_history = []
    train_acc_history = []
    validation_loss_history = []
    validation_acc_history = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_summary += "====================\nEpoch n°" + str(epoch) + "\n"
        print("====================")
        print("Epoch n°" + str(epoch))
        running_loss = 0.0
        running_corrects = 0

        for i, (inputs, labels) in enumerate(data_loader):
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, prediction = torch.max(outputs, 1)

            # backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # print statistics
            if i % 1 == 0:
                print(f'Training Batch: {i:4} of {len(data_loader)}')

            running_corrects += torch.sum(prediction == labels.data)

        epoch_train_loss = running_loss / len(data_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(data_loader.dataset)
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)

        train_summary += f"----------\nTraining Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}\n"
        print('-' * 10)
        print(f'Training Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}\n')

        running_loss = 0.0
        running_corrects = 0

        for i, (inputs, labels) in enumerate(validation_data_loader):

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, prediction = torch.max(outputs, 1)

            # print statistics
            running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(prediction == labels.data)

        epoch_validation_loss = running_loss / len(data_loader.dataset)
        epoch_validation_acc = running_corrects.double() / len(data_loader.dataset)
        validation_loss_history.append(epoch_validation_loss)
        validation_acc_history.append(epoch_validation_acc)

        train_summary += f"----------\nValidation Loss: {epoch_validation_loss:.4f} Acc: {epoch_validation_acc:.4f}\n"
        print('-' * 10)
        print(f'Validation Loss: {epoch_validation_loss:.4f} Acc: {epoch_validation_acc:.4f}\n')

    return train_loss_history, train_acc_history, validation_loss_history, validation_acc_history, model, train_summary


def testing(data_loader, model, criterion):
    test_summary = ""
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, prediction = torch.max(outputs, 1)

            # print statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(prediction == labels.data)

        test_loss = running_loss / len(data_loader.dataset)
        test_acc = running_corrects.double() / len(data_loader.dataset)
    test_summary += "-"*30 + f"\nTesting Loss: {test_loss:.4f} Acc: {test_acc:.4f}\n"
    print('-' * 30)
    print(f'Testing Loss: {test_loss:.4f} Acc: {test_acc:.4f}\n')

    return test_loss, test_acc, test_summary

