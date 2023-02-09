import random
import pyglet
import pyglet.window.key

from torch import tensor
from my_neural_network import MyNeuralNetwork
from my_neural_network import training
from my_neural_network import testing
from torch import nn
from torch import optim
import torch
import torch.utils.data as data
from my_data_set_loader import MyDataSet
from my_data_set_loader import adapt_letter_to_image
import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot(train_history, validation_history, metric, num_epochs):
    plt.title(f"Validation/Train {metric} vs. Number of Training Epochs")
    plt.xlabel(f"Training Epochs")
    plt.ylabel(f"Validation/Train {metric}")
    plt.plot(range(1, num_epochs + 1), train_history, label="Train")
    plt.plot(range(1, num_epochs + 1), validation_history, label="Validation")
    if 'accuracy' in metric:
        plt.ylim((0, 1.))
    else:
        plt.ylim((0, 4.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.savefig(f"save/{metric}.png")
    plt.show()


if __name__ == "__main__":
    mode = ""
    while mode != "train" and mode != "test":
        mode = input("Mode train or test : ")
    name_model = "model_4"
    if mode == "train":
        summary = "Summary of the model :\n"
        train_ratio = 0.6
        validation_ratio = 0.2
        test_ratio = 0.2
        data_set = MyDataSet(file_path="./data/A_Z Handwritten Data.csv", name_model=name_model)
        print("data set loaded")
        train_length = int(len(data_set) * train_ratio)
        validation_length = int(len(data_set) * validation_ratio)
        test_length = len(data_set) - train_length - validation_length

        data_train, data_validation, data_test = data.random_split(data_set, [train_length, validation_length, test_length])

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        batch_size = 0.1
        num_epochs = 20
        learning_rate = 0.001
        momentum = 0.9

        summary += "Train size : " + str(train_length) + "\n"
        summary += "Validation size : " + str(validation_length) + "\n"
        summary += "Test size : " + str(test_length) + "\n"
        summary += "Batch size : " + str(batch_size) + "\n"
        summary += "Number of epochs : " + str(num_epochs) + "\n"
        summary += "Learning Rate : " + str(learning_rate) + "\n"
        summary += "Momentum : " + str(momentum) + "\n"

        train_data_loader = data.DataLoader(data_train, batch_size=int(batch_size*train_length), shuffle=True)
        validation_data_loader = data.DataLoader(data_train, batch_size=int(batch_size*validation_length), shuffle=True)
        test_data_loader = data.DataLoader(data_test, batch_size=test_length, shuffle=True)

        model = MyNeuralNetwork(name_model).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = nn.CrossEntropyLoss()
        print("data fully transformed, begin training")
        training_loss_hist, training_acc_hist, validation_loss_hist, validation_acc_hist, model, train_summary = \
            training(train_data_loader, validation_data_loader, num_epochs, optimizer, model, criterion)

        train_acc_history = [h.cpu().numpy() for h in training_acc_hist]
        train_loss_history = [np.array(h) for h in training_loss_hist]

        test_loss, test_acc, test_summary = testing(test_data_loader, model, criterion)
        summary += train_summary + test_summary
        with open("save/summary_" + name_model + ".txt", "w") as f:
            f.write(summary)
        f.close()
        torch.save(model.state_dict(), "save/" + name_model + ".save")
        plot(train_acc_history, validation_acc_hist, 'accuracy_' + name_model, num_epochs)
        plot(train_loss_history, validation_loss_hist, 'loss_' + name_model, num_epochs)
    elif mode == "test":
        model = MyNeuralNetwork(name_model)
        model.load_state_dict(torch.load("save/" + name_model + ".save"))
        model.eval()
        classes = {
            0: "a",
            1: "b",
            2: "c",
            3: "d",
            4: "e",
            5: "f",
            6: "g",
            7: "h",
            8: "i",
            9: "j",
            10: "k",
            11: "l",
            12: "m",
            13: "n",
            14: "o",
            15: "p",
            16: "q",
            17: "r",
            18: "s",
            19: "t",
            20: "u",
            21: "v",
            22: "w",
            23: "x",
            24: "y",
            25: "z",
        }
        path = "images/"
        path += input("Enter name of the image : ")
        image = cv2.imread(path)
        letters = []
        spaces = []
        pic = pyglet.image.load(path)
        width, height = pic.width, pic.height

        title = "Cropping the image"

        window = pyglet.window.Window(width, height, title)

        mouse_x, mouse_y = 0, 0
        x1, y1, x2, y2 = -1, -1, -1, -1

        @window.event
        def on_mouse_motion(x, y, dx, dy):
            global mouse_x
            global mouse_y
            mouse_x = x
            mouse_y = y

        # on draw event
        @window.event
        def on_draw():
            global x1
            global y1
            global x2
            global y2

            global number_of_part
            # clearing the window
            window.clear()

            # drawing the label on the window
            pic.blit(0, 0)

            line = pyglet.shapes.Line(0, mouse_y, width, mouse_y, 1, color=[255, 255, 255])
            line.draw()
            line = pyglet.shapes.Line(mouse_x, 0, mouse_x, height, 1, color=[255, 255, 255])
            line.draw()

            for coord in letters:
                sx1, sy1, sx2, sy2 = coord
                line = pyglet.shapes.Line(sx1, sy1, sx1, sy2, 1, color=[255, 0, 0])
                line.draw()
                line = pyglet.shapes.Line(sx1, sy1, sx2, sy1, 1, color=[255, 0, 0])
                line.draw()
                line = pyglet.shapes.Line(sx2, sy2, sx1, sy2, 1, color=[255, 0, 0])
                line.draw()
                line = pyglet.shapes.Line(sx2, sy2, sx2, sy1, 1, color=[255, 0, 0])
                line.draw()

            if (x2, y2) != (-1, -1):

                letters.append((x1, y1, x2, y2))

                number_of_part += 1
                x1, y1, x2, y2 = -1, -1, -1, -1

        @window.event
        def on_key_press(symbol, modifiers):
            global letters
            global spaces
            if symbol == pyglet.window.key.BACKSPACE:
                letters = letters[:-1]
            elif symbol == pyglet.window.key.SPACE:
                spaces.append(len(letters) - 1)

        # mouse press event
        @window.event
        def on_mouse_press(x, y, button, modifiers):
            global x1
            global y1
            global x2
            global y2
            if (x1, y1) == (-1, -1):
                x1 = x
                y1 = y
            elif (x2, y2) == (-1, -1):
                x2 = x
                y2 = y

        number_of_part = 0

        # start running the application
        print("Select the letters, if you want to delete a selection, press BACKSPACE "
              "and if you want to add a space, press the SPACE bar")
        print("close the window when finished")
        pyglet.app.run()
        output = ""
        for i in range(len(letters)):
            (dx1, dy1, dx2, dy2) = letters[i]
            dx1, dx2 = min(dx1, dx2), max(dx1, dx2)
            dy1, dy2 = height - dy1, height - dy2
            dy1, dy2 = min(dy1, dy2), max(dy1, dy2)
            img = image[dy1:dy2, dx1:dx2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            th, threshed = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
            if name_model == "model_1" or name_model == "model_2" or name_model == "model_4":
                img_resized = adapt_letter_to_image(threshed, 28, 1)
                img_resized = np.asarray(img_resized).reshape((-1, 1, 28, 28)).astype("float32")
                img_resized = tensor(img_resized)
            else:
                img_resized = adapt_letter_to_image(threshed, 32, 3)
                img_resized = np.asarray(img_resized).reshape((-1, 3, 32, 32)).astype("float32")
                img_resized = tensor(img_resized)
            outputs = model(img_resized)
            _, prediction = torch.max(outputs, 1)
            output += classes[int(prediction[0])]
            if i in spaces:
                output += " "
        print(output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
