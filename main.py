import math
import os
import pickle
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from PIL import Image
import json
import cv2
from tqdm import tqdm
import imagesize

torch.set_default_dtype(torch.float64)
# np.set_printoptions(threshold=np.inf)

IMAGE_SIZE = 448
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

    def _preprocessor(self, x_raw, y_raw=None):

        num_channels = 3  # Replace with the required number of channels

        metadata = []
        x_processed = []
        for img in x_raw:
            # Pad the image to make it square before cropping
            # get the max width / height
            target_length = max(img.shape[0], img.shape[1])
            # resize to the max width / height
            metadata.append({"offset_y": (target_length - img.shape[0])//2, "offset_x": (target_length - img.shape[1])//2, "scale": target_length / IMAGE_SIZE})
            img = np.pad(img, (((target_length - img.shape[0])//2, math.ceil((target_length - img.shape[0])//2)), ((target_length - img.shape[1])//2, math.ceil((target_length - img.shape[1])//2)), (0, 0)))
            # resize to 448 x 448
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            
            img_array = np.asarray(img) / 255.0  # Normalize pixel values
            x_processed.append(img_array)

        # If the images are supposed to be greyscale , ensure that they have the correct shape
        if num_channels == 1:
        # Add an extra dimension to align with 'channels_last' format expected by some frameworks
            x_processed = x_processed[..., np.newaxis]
        # Convert the list of processed images to a numpy array
        x_processed = np.asarray(x_processed)


        y_processed = None
        if y_raw:
            y_processed = np.zeros((len(x_raw), S, S, 5))
            for i, datum in enumerate(y_raw):
                for box in datum["gtboxes"]:
                    if box["tag"] == "person" and not box["extra"].get("ignore"):
                        [x, y, w, h] = box["vbox"]
                        x += metadata[i]["offset_x"]
                        y += metadata[i]["offset_y"]
                        x /= metadata[i]["scale"]
                        y /= metadata[i]["scale"]
                        w /= metadata[i]["scale"]
                        h /= metadata[i]["scale"]
                        s_x = int(x // (IMAGE_SIZE / S))
                        s_y = int(y // (IMAGE_SIZE / S))
                        if w < y_processed[i][s_x][s_y][2]: # Take largest box per cell
                            continue
                        y_processed[i][s_x][s_y][0] = round(x % (IMAGE_SIZE / S))
                        y_processed[i][s_x][s_y][1] = round(y % (IMAGE_SIZE / S))
                        y_processed[i][s_x][s_y][2] = w
                        y_processed[i][s_x][s_y][3] = h
                        y_processed[i][s_x][s_y][4] = 1

            y_processed = np.asarray(y_processed)

        print(x_processed.shape, y_processed.shape if y_raw else None)

        # Return the processed images and labels 
        return x_processed, y_processed

    def forward(self, x):
        return self.model(x)

    def fit(self, x, y, x_val=None, y_val=None):
        x, y = self._preprocessor(x, y)
        x_val, y_val = self._preprocessor(x_val, y_val)

        train_losses = []
        val_losses = []

        print("Training...")
        for epoch in tqdm(range(self.nb_epoch)):
            indices = np.random.choice(x.shape[0], self.batch_size)
            x_batch = torch.from_numpy(x[indices]).double()
            y_batch = torch.from_numpy(y[indices]).double()
            y_pred = self.model(x_batch)

            train_loss = self.loss_function(y_pred, y_batch)
            self.optimiser.zero_grad()
            train_loss.backward()
            self.optimiser.step()

            if epoch % 100 == 0 and x_val:
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
        x, y = self._preprocessor(x, y)

        predictions = self.predict(x)
        labels = self.get_people_in_labels(y.numpy())
        total_error = 0
        for i in range(len(labels)):
            total_error += abs(labels[i] - predictions[i]) / labels[i]

        score = total_error / len(labels)
        return score


def save_model(trained_model):
    with open("model.pickle", "wb") as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in model.pickle\n")


def load_model():
    with open("model.pickle", "rb") as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in model.pickle\n")
    return trained_model

def get_labels(filepath):
    labels = {}
    with open(filepath, "r") as f:
        labels_raw = [json.loads(line) for line in filter(None, f.read().split("\n"))]
        for label in labels_raw:
            labels[label["ID"]] = label
    return labels


def main():



    labels = {}
    labels = labels | get_labels("data/annotation_val.odgt")
    labels = labels | get_labels("data/annotation_train.odgt")

    x = []
    y = []
    image_paths = os.listdir("data/Images")[:10]
    num_images = len(image_paths)
    print(f"Loading {num_images} images...")
    for i in tqdm(range(len(image_paths))):
        image = image_paths[i]
        im = cv2.imread(f"data/Images/{image}")
        x.append(im)
        y.append(labels[image[:-4]])

    x_train = x[:int(num_images * 0.8)]
    y_train = y[:int(num_images * 0.8)]
    x_val = x[int(num_images * 0.8):int(num_images * 0.9)]
    y_val = y[int(num_images * 0.8):int(num_images * 0.9)]
    x_test = x[int(num_images * 0.9):]
    y_test = y[int(num_images * 0.9):]

    model = Model(x_train)

    train_losses, val_losses = model.fit(x_train, y_train, x_val, y_val)

    train_accuracy = model.score(x_train, y_train)
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
