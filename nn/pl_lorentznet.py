from argparse import ArgumentParser
import torch
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn as nn
from torch import optim
from nn.models.lorentznet import LorentzNet
from data.transforms import  LorentzAugmentation_LorentzNet
from data.lorentznet_top_loader import initialize_top, collate_fn_top
from data.lorentznet_qg_loader import initialize_qg, collate_fn_qg, CustomTensorDataset
from torch.utils.data import DataLoader 



available_models = {
    'LorentzNet': LorentzNet,
}



class LorentzNet_TopTagging(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        model = available_models.get(hparams['model'], None)
        if model is None:
            raise ValueError(f"Model {hparams['model']} not available. Please choose from: {available_models.keys()}")

        self.model = model(n_scalar = 2, n_hidden = hparams['n_hidden'], n_class = 2,
                       dropout = hparams['dropout'], n_layers = hparams['n_layers'],
                       c_weight = hparams['c_weight'])
        self.model = self.model.to('cuda')
        
        self.criterion = nn.CrossEntropyLoss()
        
        if hparams['augmentation_params'].get('max_beta') != 0:
            augmentation = LorentzAugmentation_LorentzNet(**self.hparams['augmentation_params'])
            self.datasets = initialize_top(datadir='dataset/toptagging/converted', augmentation=augmentation)
        else:
            self.datasets = initialize_top(datadir='dataset/toptagging/converted')
        
        self.learning_rate = hparams['learning_rate']
        self.weight_decay = hparams['weight_decay']
        self.batch_size = hparams['batch_size']

        self.collate = lambda data: collate_fn_top(data, scale=1, add_beams=True, beam_mass=1)

    def forward(self, x):
        batch_size, n_nodes, _ = x['Pmu'].size()
        atom_positions = x['Pmu'].view(batch_size * n_nodes, -1).to('cuda', torch.float)
        atom_mask = x['atom_mask'].view(batch_size * n_nodes, -1).to('cuda')
        edge_mask = x['edge_mask'].reshape(batch_size * n_nodes * n_nodes, -1).to('cuda')
        nodes = x['nodes'].view(batch_size * n_nodes, -1).to('cuda', torch.float)
        nodes = torch.sign(nodes) * torch.log(torch.abs(nodes) + 1)
        edges = [a.to('cuda') for a in x['edges']]
        return self.model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask,
                         edge_mask=edge_mask, n_nodes=n_nodes)
    
    def train_dataloader(self):
        return DataLoader(self.datasets['train'], 
                          num_workers=8,
                          batch_size=self.batch_size, shuffle=True, collate_fn=self.collate)
    
    def val_dataloader(self):
        return DataLoader(self.datasets['valid'], 
                          num_workers=8,
                          batch_size=self.batch_size, shuffle=False, collate_fn=self.collate)
    
    def test_dataloader(self):
        return DataLoader(self.datasets['test'],
                          num_workers=8,
                          batch_size=self.batch_size, shuffle=False, collate_fn=self.collate)

    def training_step(self, batch, batch_idx):
        labels = batch['is_signal'].to('cuda', torch.float).long()
        y_hat = self(batch)

        loss = self.criterion(y_hat, labels)
            
        self.log('train_loss', loss, batch_size=labels.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch['is_signal'].to('cuda', torch.float).long()
        y_hat = self(batch)

        loss = self.criterion(y_hat, labels)    
        self.log('val_loss', loss, batch_size=labels.size(0))
    
    def test_step(self, batch, batch_idx):
        labels = batch['is_signal'].to('cuda', torch.float).long()
        y_hat = self(batch)

        loss = self.criterion(y_hat, labels)
        accuracy = (y_hat.argmax(dim=1) == labels).float().mean()
        self.log('test_accuracy', accuracy, batch_size=labels.size(0))
        self.log('test_loss', loss, batch_size=labels.size(0))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.hparams.get('use_scheduler'):
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, 4, 2, verbose=True),
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        return optimizer

    def evaluate_invariance(self, betas=None, logdir=None):
        if betas is None:
            betas = np.arange(0.0, 1.01, 0.02)

        accuracy_array = []
        
        self.model.eval()
        with torch.no_grad():
            for beta in tqdm(betas, desc="Evaluating invariance"):
                accuracy = []
                num_pts = {'test': 5000}
                augmentation = LorentzAugmentation_LorentzNet(max_beta=beta, p=1.0, fixed_beta=True)
                test_dataset = initialize_top(datadir='dataset/toptagging/converted', num_pts=num_pts, augmentation=augmentation)['test']
                dataloader = DataLoader(test_dataset,
                                num_workers=8,
                                batch_size=self.batch_size, shuffle=False, collate_fn=self.collate)
                
                for data in dataloader:
                    labels = data['is_signal'].to('cuda', torch.float).long()
                    y_hat = self(data)
                    accuracy.append((y_hat.argmax(dim=1) == labels).float().mean().to('cpu'))
                accuracy_array.append(np.mean(accuracy))

        import wandb
        data = [[x, y] for (x, y) in zip(betas, accuracy_array)]
        table = wandb.Table(data=data, columns=["beta", "accuracy"])
        wandb.log({"Invariance" : wandb.plot.scatter(table, "beta", "accuracy")})
        if logdir is not None:
            res = pd.DataFrame(data, columns=["beta", "accuracy"])
            res.to_csv(logdir / f"invariance_beta.csv", index=False)



class LorentzNet_QuarkGluon(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        model = available_models.get(hparams['model'], None)
        if model is None:
            raise ValueError(f"Model {hparams['model']} not available. Please choose from: {available_models.keys()}")

        self.model = model(n_scalar = 8, n_hidden = hparams['n_hidden'], n_class = 2,
                       dropout = hparams['dropout'], n_layers = hparams['n_layers'],
                       c_weight = hparams['c_weight'])
        self.model = self.model.to('cuda')
        
        self.criterion = nn.CrossEntropyLoss()
        
        if hparams['augmentation_params'].get('max_beta') != 0:
            augmentation = LorentzAugmentation_LorentzNet(**self.hparams['augmentation_params'])
            self.datasets = initialize_qg(num_data=-1, cache_dir='dataset/quarkgluon/cache', use_one_hot=True, augmentation=augmentation)

        else:
            self.datasets = initialize_qg(num_data=-1, cache_dir='dataset/quarkgluon/cache', use_one_hot=True)

        self.learning_rate = hparams['learning_rate']
        self.weight_decay = hparams['weight_decay']
        self.batch_size = hparams['batch_size']

        self.collate = collate_fn_qg

    def forward(self, x):
        label, p4s, nodes, atom_mask, edge_mask, edges = x
        batch_size, n_nodes, _ = p4s.size()
        atom_positions = p4s.view(batch_size * n_nodes, -1).to('cuda', torch.float)
        atom_mask = atom_mask.view(batch_size * n_nodes, -1).to('cuda')
        edge_mask = edge_mask.reshape(batch_size * n_nodes * n_nodes, -1).to('cuda')
        nodes = nodes.view(batch_size * n_nodes, -1).to('cuda', torch.float)
        edges = [a.to('cuda') for a in edges]
        label = label.to('cuda', torch.float).long()
        return self.model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask,
                         edge_mask=edge_mask, n_nodes=n_nodes), label
    
    def train_dataloader(self):
        return DataLoader(self.datasets['train'], 
                          num_workers=8,
                          batch_size=self.batch_size, shuffle=True, collate_fn=self.collate)
    
    def val_dataloader(self):
        return DataLoader(self.datasets['val'], 
                          num_workers=8,
                          batch_size=self.batch_size, shuffle=False, collate_fn=self.collate)
    
    def test_dataloader(self):
        return DataLoader(self.datasets['test'],
                          num_workers=8,
                          batch_size=self.batch_size, shuffle=False, collate_fn=self.collate)

    def training_step(self, batch, batch_idx):
        y_hat, labels = self(batch)

        loss = self.criterion(y_hat, labels)            
        self.log('train_loss', loss, batch_size=labels.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, labels = self(batch)

        loss = self.criterion(y_hat, labels)    
        self.log('val_loss', loss, batch_size=labels.size(0))
    
    def test_step(self, batch, batch_idx):
        y_hat, labels = self(batch)

        loss = self.criterion(y_hat, labels)
        accuracy = (y_hat.argmax(dim=1) == labels).float().mean()
        self.log('test_accuracy', accuracy, batch_size=labels.size(0))
        self.log('test_loss', loss, batch_size=labels.size(0))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.hparams.get('use_scheduler'):
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, 4, 2, verbose=True),
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        return optimizer

    def evaluate_invariance(self, betas=None, logdir=None):
        if betas is None:
            betas = np.arange(0.0, 1.01, 0.02)

        accuracy_array = []
        
        self.model.eval()
        with torch.no_grad():
            for beta in tqdm(betas, desc="Evaluating invariance"):
                accuracy = []
                augmentation = LorentzAugmentation_LorentzNet(max_beta=beta, p=1.0, fixed_beta=True)
                test_dataset = initialize_qg(num_data=10000, cache_dir='dataset/quarkgluon/cache', use_one_hot=True, augmentation=augmentation)['test']
                # loaded_dataset = torch.load('dataset/quarkgluon/cache/lorentznet_qg_5000.pt')
                # test_dataset = CustomTensorDataset((loaded_dataset[0], loaded_dataset[1], loaded_dataset[2], loaded_dataset[3]), augmentation=augmentation)
                dataloader = DataLoader(test_dataset,
                                num_workers=8,
                                batch_size=self.batch_size, shuffle=False, collate_fn=self.collate)
                
                for data in dataloader:
                    y_hat, labels = self(data)
                    accuracy.append((y_hat.argmax(dim=1) == labels).float().mean().to('cpu'))
                accuracy_array.append(np.mean(accuracy))

        import wandb
        data = [[x, y] for (x, y) in zip(betas, accuracy_array)]
        table = wandb.Table(data=data, columns=["beta", "accuracy"])
        wandb.log({"Invariance" : wandb.plot.scatter(table, "beta", "accuracy")})
        if logdir is not None:
            res = pd.DataFrame(data, columns=["beta", "accuracy"])
            res.to_csv(logdir / f"invariance_beta.csv", index=False)