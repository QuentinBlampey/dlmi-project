import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
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

    def step(self, loader):
        epoch_loss = 0.0
        y_preds, probas, y_true = [], [], []

        for images, medical_data, label in tqdm(loader):
            images = images.to(self.device)[0, :]
            medical_data = medical_data.to(self.device)[0, :]
            label = label.type(torch.FloatTensor).to(self.device)
            logits = self(images, medical_data)
            pred = torch.round(torch.sigmoid(logits))

            loss = self.loss_function(logits, label)
            epoch_loss += loss
            probas.append(torch.sigmoid(logits).item())
            y_preds.append(int(pred.item()))
            y_true.append(label.item())

            if self.training:
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
        best = (0,0)
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
        plt.xlim(0,1)
        plt.ylim(0,1)
        path_fig = f"{datetime.now().strftime('%y-%m-%d_%Hh%Mm%Ss')}.png"
        plt.savefig(f"../submissions/{path_fig}")
        print("Fig saved. Named:", path_fig)
        plt.close()

        if not self.training:
            self.best_thresholds.append(best[1])
        
        return epoch_loss, acc

    def train_and_eval(self, train_loader, val_loader, n_epochs, loss_function, learning_rate, weight_decay):
        self.optimizer = Adam(self.parameters(), learning_rate, weight_decay=weight_decay)
        self.loss_function = loss_function

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")

            self.train()
            train_loss, train_acc = self.step(train_loader)
            print(f"Train loss: {train_loss} | Train acc: {train_acc}")

            self.eval()
            with torch.no_grad():
                _, val_acc = self.step(val_loader)
            print(f"Val acc: {val_acc} ")
        return val_acc

    def train_only(self, train_loader, n_epochs, loss_function, learning_rate, weight_decay):
        self.optimizer = Adam(self.parameters(), learning_rate, weight_decay=weight_decay)
        self.loss_function = loss_function

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            self.train()
            train_loss, train_acc = self.step(train_loader)
            print(f"Train loss: {train_loss} | Train acc: {train_acc}")

    def predict(self, test_loader, cutting_threshold):
        print("\nComputing predictions")
        self.best_thresholds = np.array(self.best_thresholds)
        #best_threshold = self.best_thresholds[-10:].mean()
        print(f"Cutting at threshold {cutting_threshold}")

        y_preds = []
        self.eval()
        with torch.no_grad():
            for images, medical_data, _ in tqdm(test_loader):
                images = images.to(self.device)[0, :]
                medical_data = medical_data.to(self.device)[0, :]
                logits = self(images, medical_data)
                if cutting_threshold == 0:
                    y_preds.append(torch.sigmoid(logits).item())
                else:
                    pred = torch.sigmoid(logits) >= cutting_threshold
                    y_preds.append(int(pred.item()))

        return y_preds
