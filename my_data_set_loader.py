from torch.utils.data import Dataset
from torch import tensor
import csv
import numpy as np
import cv2


def adapt_letter_to_image(image, size, pixel_size):
    row_img, col_img = image.shape

    # crop image
    x, y, w, h = -1, -1, -1, -1
    for i in range(row_img):
        for j in range(col_img):
            if image[i][j] != 0:
                if x == -1 or x > i:
                    x = i
                if y == -1 or y > j:
                    y = j
                if i >= w:
                    w = i
                if j >= h:
                    h = j

    res = image[x:(w + 1), y:(h + 1)]

    # resize image so that it enters a size*size square
    row, col = res.shape

    output = np.zeros((size, size))
    b_row, b_col = output.shape

    if col >= row:
        if col != size:
            res = cv2.resize(res, (size, int((size / col) * row)), interpolation=cv2.INTER_LINEAR)
        row, col = res.shape
        x = int((b_row - row) / 2)
        y = 0
    else:
        if row != size:
            res = cv2.resize(res, (int((size / row) * col), size), interpolation=cv2.INTER_LINEAR)
        row, col = res.shape
        x = 0
        y = int((b_col - col) / 2)
    output[x:(x + row), y:(y + col)] = res
    if pixel_size > 1:
        out_dim = []
        for i in range(pixel_size):
            out_dim.append(output)
        output = np.dstack(out_dim)

    return output


def create_vectors(file_path, name_model):
    width = 28
    height = 28
    labels = []
    image_data = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        for line in csv_reader:
            image = np.zeros((width, height))
            labels.append(int(line[0]))
            for i in range(0, height):
                for j in range(0, width):
                    image[i][j] = line[1 + i * height + j]
            if name_model == "model_1" or name_model == "model_2" or name_model == "model_4":
                image = adapt_letter_to_image(image, 28, 1)
            elif name_model == "model_3":
                image = adapt_letter_to_image(image, 32, 3)
            image_data.append(image)
    csv_file.close()
    return labels, image_data


def load_file(file_path, name_model):
    labels, data = create_vectors(file_path, name_model)
    if name_model == "model_1" or name_model == "model_2" or name_model == "model_4":
        data = np.asarray(data).reshape((-1, 1, 28, 28)).astype("float32")
    elif name_model == "model_3":
        data = np.asarray(data).reshape((-1, 3, 32, 32)).astype("float32")
    return labels, data


class MyDataSet(Dataset):
    def __init__(self, file_path, name_model, transform=None):
        self.labels, self.data = load_file(file_path, name_model)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        label = tensor(label)
        image = self.data[index]
        image = tensor(image)
        return image, label
