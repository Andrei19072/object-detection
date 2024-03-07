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
from torch.autograd import Variable

torch.set_default_dtype(torch.float64)
# np.set_printoptions(threshold=np.inf)

IMAGE_SIZE = 448
S = 30
B = 2
CONFIDENCE_THRESHHOLD = 0.5

class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()

    def forward(self, y_pred, y_true, use_cuda=False):
        SS = S * S
        scale_object_conf = 1
        scale_noobject_conf = 0.5
        scale_coordinate = 5
        batch_size = y_pred.size(0)

        # ground truth
        y_true = y_true.view(-1, SS, B, 5)
        y_pred = y_pred.view(-1, SS, B, 5)
        _coord = y_true[:,:,:,:4]
        _wh = torch.pow(_coord[:, :, :, 2:4], 2)
        _areas = _wh[:, :, :, 0] * _wh[:, :, :, 1]
        _upleft = _coord[:, :, :, 0:2]
        _bottomright = _upleft + _wh
        _confs = torch.sigmoid(y_true[:,:,:,4:])

        # Extract the coordinate prediction from y_pred
        coords = y_pred[:,:,:,:4].contiguous().view(-1, SS, B, 4)
        wh = torch.pow(coords[:, :, :, 2:4], 2)
        areas = wh[:, :, :, 0] * wh[:, :, :, 1]
        upleft = coords[:, :, :, 0:2].contiguous()
        bottomright = upleft + wh

        # Calculate the intersection areas
        intersect_upleft = torch.max(upleft, _upleft)
        intersect_bottomright = torch.min(bottomright, _bottomright)
        intersect_wh = intersect_bottomright - intersect_upleft
        zeros = Variable(torch.zeros(batch_size, SS, B, 2)).cuda() if use_cuda else Variable(torch.zeros(batch_size, SS, B, 2))
        intersect_wh = torch.max(intersect_wh, zeros)
        intersect = intersect_wh[:, :, :, 0] * intersect_wh[:, :, :, 1]

        # Calculate the best IOU, set 0.0 confidence for worse boxes
        iou = intersect / (_areas + areas - intersect)
        best_box = torch.eq(iou, torch.max(iou, 2).values.unsqueeze(2)).unsqueeze(-1)
        confs = best_box.float() * _confs

        # Take care of the weight terms
        conid = scale_noobject_conf * (1. - confs) + scale_object_conf * confs
        weight_coo = torch.cat(4 * [confs.unsqueeze(-1)], 3)
        cooid = scale_coordinate * weight_coo

        def flatten(x):
            return x.reshape(x.size(0), -1)

        # Flatten 'em all
        confs = flatten(confs)
        conid = flatten(conid)
        coord = flatten(_coord)
        cooid = flatten(cooid)
        y_pred = flatten(y_pred)

        true = torch.cat([confs, coord], 1)
        wght = torch.cat([conid, cooid], 1)
        loss = torch.pow(y_pred - true, 2)
        loss = loss * wght
        loss = torch.sum(loss, 1)
        return .5 * torch.mean(loss)

class Model(nn.Module):
    def __init__(
        self,
        x,
    ):
        super(Model, self).__init__()
        data, _ = self._preprocessor(x)

        self.x = x
        self.input_size = data.shape[2:4]
        self.output_size = 1

        self.nb_epoch = 5
        self.learning_rate = 0.01
        self.batch_size = 64

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 192, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),

            nn.Flatten(),
            nn.Linear(math.ceil(self.input_size[0]/64) * math.ceil(self.input_size[1] / 64)*1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S*S*B*5)
        )

        self.loss_function = YoloLoss()

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
            x_processed.append(np.transpose(img_array, (2, 0, 1)))

        # If the images are supposed to be greyscale , ensure that they have the correct shape
        if num_channels == 1:
        # Add an extra dimension to align with 'channels_last' format expected by some frameworks
            x_processed = x_processed[..., np.newaxis]
        # Convert the list of processed images to a numpy array
        x_processed = np.asarray(x_processed)


        y_processed = None
        if y_raw:
            y_processed = np.zeros((len(x_raw), S, S, B, 5))
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
                        w = math.sqrt(w)
                        h = math.sqrt(h)
                        s_x = int(x // (IMAGE_SIZE / S))
                        s_y = int(y // (IMAGE_SIZE / S))
                        if w < y_processed[i][s_x][s_y][0][2]: # Take largest box per cell
                            continue
                        for b in range(B):
                            y_processed[i][s_x][s_y][b][0] = round(x % (IMAGE_SIZE / S))
                            y_processed[i][s_x][s_y][b][1] = round(y % (IMAGE_SIZE / S))
                            y_processed[i][s_x][s_y][b][2] = w
                            y_processed[i][s_x][s_y][b][3] = h
                            y_processed[i][s_x][s_y][b][4] = 1

            y_processed = np.asarray(y_processed)

        # print(x_processed.shape, y_processed.shape if y_raw else None)

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

            if epoch % 100 == 0 or True:
                with torch.no_grad():
                    indices = np.random.choice(x_val.shape[0], self.batch_size)
                    x_val_batch = torch.from_numpy(x_val[indices])
                    y_val_batch = torch.from_numpy(y_val[indices])
                    y_val_pred = self.model(x_val_batch)
                    val_loss = self.loss_function(y_val_pred, y_val_batch)

                    train_losses.append(train_loss.item())
                    val_losses.append(val_loss.item())

                    # print(f"Training loss: {train_loss}")
                    # print(f"Validation loss: {val_loss}")

        print("\nFinished Training...")
        return train_losses, val_losses

    def predict(self, x):
        x, _ = self._preprocessor(x)
        with torch.no_grad():
            predictions = self.model(torch.from_numpy(x).double()).detach().view(-1, S, S, B, 5)
        people = self.get_people_in_labels(1.0 / (1.0 + np.exp(-predictions.numpy())))
        return people

    def score(self, x, y):
        _, y = self._preprocessor(x, y)

        predictions = self.predict(x)
        labels = self.get_people_in_labels(y)
        total_error = 0
        for i in range(len(labels)):
            total_error += abs(labels[i] - predictions[i]) / labels[i]

        score = total_error / len(labels)
        return score

    def get_people_in_labels(self, labels):
        people_arr = np.zeros((labels.shape[0],))
        for index, label in enumerate(labels):
            people = 0
            for i in range(S):
                for j in range(S):
                    for b in range(B):
                        if label[i][j][b][4] > CONFIDENCE_THRESHHOLD:
                            people += 1
            people_arr[index] = people
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
    image_paths = os.listdir("data/Images")[:50]
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
    print(f"\nTraining inaccuracy: {round(train_accuracy * 100)}%\n")

    val_accuracy = model.score(x_val, y_val)
    print(f"\nValidation inaccuracy: {round(val_accuracy * 100)}%\n")

    test_accuracy = model.score(x_test, y_test)
    print(f"\nTest inaccuracy: {round(test_accuracy * 100)}%\n")

    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.show()

    save_model(model)


if __name__ == "__main__":
    main()
