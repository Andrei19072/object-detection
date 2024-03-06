import pickle
import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
<<<<<<< HEAD
from PIL import Image
=======
import json
import cv2
from tqdm import tqdm
import imagesize
>>>>>>> dd99ae10b36f47b4c71402f59bf6c8de4e01e557

torch.set_default_dtype(torch.float64)
np.set_printoptions(threshold=np.inf)

IMAGE_WIDTH = 448
IMAGE_HEIGHT = 448
S = 30
B = 2
CONFIDENCE_THRESHHOLD = 0.5

class Model(nn.Module):
    def __init__(
        self,
        x,
    ):
        super(Model, self).__init__()
        data, _ = self._preprocessor(x)

        self.x = x
        self.input_size = data.shape[1]
        self.output_size = 1

        self.nb_epoch = 1000
        self.learning_rate = 0.01
        self.batch_size = 16

        self.model = nn.Sequential(  # TODO
            nn.Linear(self.input_size, 128),
            nn.Sigmoid(),
            nn.Linear(128, 32),
            nn.Sigmoid(),
            nn.Linear(32, self.output_size),
            nn.Sigmoid(),
        )

        self.loss_function = nn.BCELoss()  # TODO

        self.optimiser = optim.Adam(self.model.parameters(), self.learning_rate)

    def _preprocessor(self, x, y=None):

<<<<<<< HEAD
        target_size = (128, 128)  # Replace with the required dimensions
        num_channels = 3  # Replace with the required number of channels

        x_processed = []
        for img in x:
            img = Image.open(img).resize(target_size)
            
            if num_channels == 1:  # Add a channel dimension if grayscale
                img = img.convert('L')
            else:
                img = img.convert('RGB')
            img_array = np.asarray(img) / 255.0  # Normalize pixel values

        # If the images are supposed to be greyscale (single-channel), ensure that they have the correct shape
        if num_channels == 1:
        # Add an extra dimension to align with 'channels_last' format expected by some frameworks
            x_processed = x_processed[..., np.newaxis]
        # Convert the list of processed images to a numpy array
        x_processed = np.array(x_processed)
        x_processed.append(img_array)

        # Return the processed images and labels 
        return x_processed, y
=======
        # preprocess x
        ...

        if y:
            mapping = {}
            for i in tqdm(range(len(y))):
                datum = y[i]
                id = datum["ID"]
                mapping[id] = np.zeros((S, S, 5))
                (image_width, image_height) = imagesize.get(f"data/Images/{id}.jpg")
                for box in datum["gtboxes"]:
                    if box["tag"] == "person" and not box["extra"].get("ignore"):
                        [x, y, w, h] = box["vbox"]
                        x, y, w, h = round(x/image_width * IMAGE_WIDTH), round(y/image_height * IMAGE_HEIGHT), round(w/image_width* IMAGE_WIDTH), round(h/image_height * IMAGE_HEIGHT)
                        s_x = int(x // (IMAGE_WIDTH / S))
                        s_y = int(y // (IMAGE_HEIGHT / S))
                        mapping[id][s_x][s_y][0] = round(x % (IMAGE_WIDTH / S))
                        mapping[id][s_x][s_y][1] = round(y % (IMAGE_WIDTH / S))
                        mapping[id][s_x][s_y][2] = w
                        mapping[id][s_x][s_y][3] = h
                        mapping[id][s_x][s_y][4] = 1

        return x, y
>>>>>>> dd99ae10b36f47b4c71402f59bf6c8de4e01e557

    def forward(self, x):
        return self.model(x)

    def fit(self, x, y, x_val=None, y_val=None):
        x, y = self._preprocessor(x, y)
        x_val, y_val = self._preprocessor(x_val, y_val)

        train_losses = []
        val_losses = []

        print("Training...")
        for epoch in range(self.nb_epoch):
            indices = np.random.choice(x.shape[0], self.batch_size)
            x_batch = torch.from_numpy(x[indices]).double()
            y_batch = torch.from_numpy(y[indices]).double()
            y_pred = self.model(x_batch)

            train_loss = self.loss_function(y_pred, y_batch)
            self.optimiser.zero_grad()
            train_loss.backward()
            self.optimiser.step()

            if epoch % 100 == 0:
                with torch.no_grad():
                    indices = np.random.choice(x_val.shape[0], self.batch_size)
                    x_val_batch = torch.from_numpy(x_val[indices])
                    y_val_batch = torch.from_numpy(y_val[indices])
                    y_val_pred = self.model(x_val_batch)
                    val_loss = self.loss_function(y_val_pred, y_val_batch)

                    train_losses.append(train_loss.item())
                    val_losses.append(val_loss.item())

                    print(f"Training loss: {train_loss}")
                    print(f"Validation loss: {val_loss}")

        print("\nFinished Training...")
        return train_losses, val_losses

    def predict(self, x):
        x, _ = self._preprocessor(x)
        with torch.no_grad():
            predictions = self.model(torch.from_numpy(x).double()).detach()
        people = self.get_people_in_labels(predictions.numpy())
        return people

    def score(self, x, y):
        _, y = self._preprocessor(x, y)

        predictions = self.predict(x)
        labels = self.get_people_in_labels(y.numpy())
        total_error = 0
        for i in range(len(labels)):
            total_error += abs(labels[i] - predictions[i]) / labels[i]

        score = total_error / len(labels)
        return score
    
    def get_people_in_labels(self, labels):
        people_arr = np.zeros((labels.shape[0],))
        for i, label in enumerate(labels):
            people = 0
            for i in range(S):
                for j in range(S):
                    if label[i][j][4] > CONFIDENCE_THRESHHOLD:
                        people += 1
            people_arr[i] = people
        return people_arr
            


def save_model(trained_model):
    with open("model.pickle", "wb") as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in model.pickle\n")


def load_model():
    with open("model.pickle", "rb") as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in model.pickle\n")
    return trained_model

def main():

    with open("data/annotation_val.odgt", "r") as f:
        y = [json.loads(line) for line in filter(None, f.read().split("\n"))]

    # TODO
    images_dir = 'data/Images'
    image_paths = [
        os.path.join(images_dir, f) for f in os.listdir(images_dir) 
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]

    print(f"Found {len(image_paths)} images")

    x = np.asarray([[1, 2, 3], [3, 2, 1]], dtype=np.double)
    y = np.asarray([[0], [1]], dtype=np.double)
    x_val = np.asarray([[1, 2, 3], [3, 2, 1]], dtype=np.double)
    y_val = np.asarray([[0], [1]], dtype=np.double)
    x_test = np.asarray([[1, 2, 3], [3, 2, 1]], dtype=np.double)
    y_test = np.asarray([[0], [1]], dtype=np.double)

    model = Model(x)

    train_losses, val_losses = model.fit(x, y, x_val, y_val)

    train_accuracy = model.score(x, y)
    print(f"\nTraining accuracy: {train_accuracy}\n")

    val_accuracy = model.score(x_val, y_val)
    print(f"\nValidation accuracy: {val_accuracy}\n")

    test_accuracy = model.score(x_test, y_test)
    print(f"\nTest accuracy: {test_accuracy}\n")

    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.show()

    save_model(model)


if __name__ == "__main__":
    main()
