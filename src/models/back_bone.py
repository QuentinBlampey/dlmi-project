from torch.optim import Adam
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import balanced_accuracy_score
        

class BackBone(nn.Module):
    def __init__(self, cnn, aggregator, top_head, device):
        super(BackBone, self).__init__()
        self.cnn = cnn
        self.aggregator = aggregator
        self.top_head = top_head
        self.device = device
        self.to(self.device)

    def forward(self, images, medical_data):
        x = self.cnn(images)
        x = self.aggregator(x)
        return self.top_head(x, medical_data)

    def step(self, loader):
        epoch_loss = 0.0
        y_preds, y_true = [], []
        
        for images, medical_data, label in tqdm(loader):
            images = images.to(self.device)[0,:]
            medical_data = medical_data.to(self.device)[0,:]
            label = label.type(torch.FloatTensor).to(self.device)
            logits = self(images, medical_data)
            pred = torch.round(torch.sigmoid(logits))

            loss = self.loss_function(logits, label)
            epoch_loss += loss
            y_preds.append(int(pred.item()))
            y_true.append(label.item())

            if self.training:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        acc = balanced_accuracy_score(np.array(y_preds), np.array(y_true))
        return epoch_loss, acc

    def train_and_eval(self, train_loader, val_loader, n_epochs, loss_function, learning_rate):
        self.optimizer = Adam(self.parameters(), learning_rate)
        self.loss_function = loss_function

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")

            self.train()
            train_loss, train_acc = self.step(train_loader)
            print(f"Train loss: {train_loss} | Train acc: {train_acc}")

            self.eval()
            _, val_acc = self.step(val_loader)
            print(f"Val acc: {val_acc} ")

    def predict(self, test_loader):
        print("\nComputing predictions")
        y_preds = []
        self.eval()
        for images, medical_data, _ in tqdm(test_loader):
            images = images.to(self.device)[0,:]
            medical_data = medical_data.to(self.device)[0,:]
            logits = self(images, medical_data)
            pred = torch.round(torch.sigmoid(logits))
            y_preds.append(int(pred.item()))

        return y_preds