from abc import ABC, abstractmethod
from torch.optim import Adam
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import balanced_accuracy_score

class ModelABC(ABC):
    @abstractmethod
    def train(self, train_loader, val_loader, n_epochs, loss_function, learning_rate):
        pass
    
    @abstractmethod
    def predict(self, test_loader):
        pass

def train_epoch(model, train_loader, optimizer, loss_function):
    epoch_loss = 0.0
    y_preds = []
    y_true = []
    model.train()
    for images, lymph_counts, gender, age, label in tqdm(train_loader):
        images = images.to(model.device)
        label = label.type(torch.FloatTensor).to(model.device)
        probas = model(images[0,:])
        pred = torch.round(probas)
        loss = loss_function(probas, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss
        y_preds.append(pred.item())
        y_true.append(label.item())
    acc = balanced_accuracy_score(np.array(y_preds), np.array(y_true))
    return loss, acc

def val_epoch(model, val_loader):
    y_preds = []
    y_true = []
    model.eval()
    for images, lymph_counts, gender, age, label in tqdm(val_loader):
        images = images.to(model.device)
        label = label.to(model.device)
        probas = model(images[0,:])
        pred = torch.round(probas)
        y_preds.append(int(pred.item()))
        y_true.append(label.item())
    acc = balanced_accuracy_score(np.array(y_true), np.array(y_preds))
    return acc

class BaselineModel(ModelABC):
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(self.device)
        self.model.device = device
        
    def train(self, train_loader, val_loader, n_epochs, loss_function, learning_rate):
        optimizer = Adam(self.model.parameters(), learning_rate)
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            train_loss, train_acc = train_epoch(self.model, train_loader, optimizer, loss_function)
            print(f"Train loss: {train_loss} | Train acc: {train_acc}")
            val_acc = val_epoch(self.model, val_loader)
            print(f"Val acc: {val_acc} ")
    
    def predict(self, test_loader):
        preds = []
        for images, lymph_counts, gender, age, _ in test_loader:
            probas = self.model(images[0,:])[None, :]
            _, pred = torch.max(probas, axis=1)
            preds.append(pred.item())
        return preds

    def get_predictions(self, test_loader):
        print("\nComputing predictions")
        y_preds = []
        self.model.eval()
        for images, lymph_counts, gender, age, _ in tqdm(test_loader):
            images = images.to(self.model.device)
            probas = self.model(images[0,:])
            pred = torch.round(probas)
            y_preds.append(int(pred.item()))
        return y_preds