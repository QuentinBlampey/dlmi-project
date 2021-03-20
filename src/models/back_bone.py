import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.optim import Adam
from tqdm import tqdm


class BackBone(nn.Module):
    def __init__(self, cnn, aggregator, top_head, device):
        super(BackBone, self).__init__()
        self.device = device
        self.cnn = cnn
        self.aggregator = aggregator
        self.top_head = top_head
        self.best_thresholds = []

    def forward(self, images, medical_data):
        x = self.cnn(images)
        x = self.aggregator(x)
        return self.top_head(x, medical_data)

    def step(self, loader, batch_size, lambdas, cutting_threshold):
        epoch_loss = 0.0
        y_preds, probas, y_true = [], [], []
        count_batch = 0
        loss = 0
        for images, medical_data, label in tqdm(loader):
            count_batch += 1
            images = images.to(self.device)[0, :]
            medical_data = medical_data.to(self.device)[0, :]
            label = label.type(torch.FloatTensor).to(self.device)
            y_med, y_cnn, y_joint = self(images, medical_data)
            pred = torch.sigmoid(y_joint) >= cutting_threshold
            del images
            del medical_data

            loss += (lambdas[0] * self.loss_function(y_med, label) + \
                     lambdas[1] * self.loss_function(y_cnn, label) + \
                     lambdas[2] * self.loss_function(y_joint, label)) / np.sum(lambdas)
            epoch_loss += loss.item()

            probas.append(torch.sigmoid(y_joint).item())
            y_preds.append(int(pred.item()))
            y_true.append(label.item())

            if self.training and count_batch == batch_size:
                count_batch = 0
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss = 0

        if self.training and count_batch > 0:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        print(self.training, np.mean(y_preds), np.mean(y_true))
        if not self.training:
            print(y_preds)
            print(list(map(int, y_true)))
        acc = balanced_accuracy_score(np.array(y_preds), np.array(y_true))
        print(f"   > {sum(y_preds)}/{len(y_preds)} positive predicted labels instead of {sum(y_true)}")

        probas = np.array(probas)
        best = (0, 0)
        ba_list = []
        thresholds = sorted(set(probas))
        for threshold in thresholds:
            preds = (probas >= threshold).astype(int)
            ba = balanced_accuracy_score(np.array(preds), np.array(y_true))
            ba_list.append(ba)
            if ba > best[0]:
                best = (ba, threshold)
        print('   > Best balance accuracy', best[0], 'at threshold', best[1])
        plt.plot(thresholds, ba_list)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        path_fig = f"{datetime.now().strftime('%y-%m-%d_%Hh%Mm%Ss')}.png"
        try:
            plt.savefig(f"../submissions/{path_fig}")
            print("Fig saved. Named:", path_fig)
        except:
            pass
        plt.close()

        if not self.training:
            self.best_thresholds.append(best[1])

        return epoch_loss, acc, y_true, probas

    def train_and_eval(self, train_loader, val_loader, n_epochs, loss_function, learning_rate, weight_decay,
                       lambdas, cutting_threshold, batch_size=1):
        self.optimizer = Adam(self.parameters(), learning_rate, weight_decay=weight_decay)
        self.loss_function = loss_function

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")

            self.train()
            train_loss, train_acc, train_y_true, train_probas = self.step(train_loader, batch_size, lambdas, cutting_threshold)
            print(f"Train loss: {train_loss} | Train acc: {train_acc}")

            self.eval()
            with torch.no_grad():
                _, val_acc, val_y_true, val_probas = self.step(val_loader, batch_size, lambdas, cutting_threshold)
            print(f"Val acc: {val_acc} ")
        with open(f"../submissions/train_{datetime.now().strftime('%y-%m-%d_%Hh%Mm%Ss')}.json", "w") as f:
            json.dump({"y_true": list(train_y_true), "probas": list(train_probas)}, f)
        with open(f"../submissions/val_{datetime.now().strftime('%y-%m-%d_%Hh%Mm%Ss')}.json", "w") as f:
            json.dump({"y_true": list(val_y_true), "probas":list(val_probas)}, f)
        return val_acc

    def train_only(self, train_loader, n_epochs, loss_function, learning_rate, weight_decay, lambdas, cutting_threshold, batch_size=1):
        self.optimizer = Adam(self.parameters(), learning_rate, weight_decay=weight_decay)
        self.loss_function = loss_function

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            self.train()
            train_loss, train_acc, _, _ = self.step(train_loader, batch_size, lambdas, cutting_threshold)
            print(f"Train loss: {train_loss} | Train acc: {train_acc}")

    def predict(self, test_loader, cutting_threshold):
        print("\nComputing predictions")
        self.best_thresholds = np.array(self.best_thresholds)
        # best_threshold = self.best_thresholds[-10:].mean()
        print(f"Cutting at threshold {cutting_threshold}")

        y_preds = []
        self.eval()
        with torch.no_grad():
            for images, medical_data, _ in tqdm(test_loader):
                images = images.to(self.device)[0, :]
                medical_data = medical_data.to(self.device)[0, :]
                y_med, y_cnn, y_joint = self(images, medical_data)
                if cutting_threshold == 0:
                    y_preds.append(torch.sigmoid(y_joint).item())
                else:
                    pred = torch.sigmoid(y_joint) >= cutting_threshold
                    y_preds.append(int(pred.item()))

        return y_preds
