import torch

from tqdm.notebook import tqdm

import numpy as np


class Trainer(object):
    
    def __init__(self,model,criterion, optimizer,  train_loader, val_loader, n_epochs,model_dir):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        ### Training and validation ###
        self.loss_history = {'training': [], 'validation': []}
        self.acc_history = {'training': [], 'validation': []}
        self.n_epochs = n_epochs
        self.model_dir = model_dir


    def train(self,loss_history,acc_history):
        self.loss_history['training'].append(0)
        self.acc_history['training'].append(0)
        
        for batch_idx, (images, targets) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="#train_batches", leave=False):
            self.model.train()
            self.optimizer.zero_grad()

            images = images.float()
            targets = targets.float()

            predictions = self.model(images)
            loss = self.criterion(predictions, targets)
            loss.backward()

            self.optimizer.step()

            #Accuracy
            accuracy_predictions = (torch.sigmoid(predictions)>0.5).float()

            correct = (accuracy_predictions == targets).float().sum()/accuracy_predictions.shape[0]

            self.loss_history['training'][-1] += float(loss.data)
            self.acc_history['training'][-1] += float(correct)

        self.loss_history['training'][-1] /= batch_idx + 1
        self.acc_history['training'][-1] /= batch_idx + 1
    
    def validate(self,loss_history,acc_history):
        loss_history['validation'].append(0)
        acc_history['validation'].append(0)

        for batch_idx, (images, targets) in tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc="#test_batches", leave=False):
            self.model.eval()

            images = images.float()
            targets = targets.float()

            predictions = self.model(images)
            loss = self.criterion(predictions, targets)

            #Accuracy
            accuracy_predictions = (torch.sigmoid(predictions)>0.5).float()
            correct = (accuracy_predictions == targets).float().sum()/accuracy_predictions.shape[0]

            loss_history['validation'][-1] += float(loss.data)
            acc_history['validation'][-1] += float(correct)

        loss_history['validation'][-1] /= batch_idx + 1
        acc_history['validation'][-1] /= batch_idx + 1


    def train_and_validate(self):
        print(f'Running {self.model.name}')


        best_val_loss = 9999999

        for epoch in tqdm(range(self.n_epochs), desc="#epochs"):
            self.train(self.loss_history, self.acc_history)

            self.validate(self.loss_history, self.acc_history)
            
            if self.loss_history['validation'][-1] < best_val_loss:
                best_val_loss = self.loss_history['validation'][-1]
                torch.save(self.model.state_dict(), '{:s}/{:s}_{:03d}.npz'.format(self.model_dir, self.model.name, epoch))

            print('epoch: {:3d} / {:03d}, training loss: {:.4f}, validation loss: {:.4f}, training accuracy: {:.3f}, validation accuracy: {:.3f}.'.format(epoch + 1, self.n_epochs, self.loss_history['training'][-1], self.loss_history['validation'][-1], self.acc_history['training'][-1], self.acc_history['validation'][-1]))
            np.savez('{:s}/{:s}_loss_history_{:03d}.npz'.format(self.model_dir, self.model.name, epoch), self.loss_history)