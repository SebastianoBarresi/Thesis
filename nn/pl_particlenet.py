from argparse import ArgumentParser
import torch
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn as nn
from torch import optim
from nn.models.particlenet import ParticleNet
from data.transforms import LorentzAugmentation_FeatureTransformation, FeaturesTransformation, transform_coordinates
from data.dataloaders import TopDataset, QGDataset
from torch_geometric.loader import DataLoader 



available_models = {
    'ParticleNet': ParticleNet,
}



class Particlenet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        model = available_models.get(hparams['model'], None)
        if model is None:
            raise ValueError(f"Model {hparams['model']} not available. Please choose from: {available_models.keys()}")
        self.datasetCl = eval(hparams['dataset']) 
        self.pid = hparams['dataset_params']['pid']

        self.model = model(hparams)
        self.criterion = nn.NLLLoss()
        
        if hparams['augmentation_params'].get('max_beta') != 0 and not hparams['lorentz_coeff']:
            self.hparams['dataset_params']['transform'] = LorentzAugmentation_FeatureTransformation(**self.hparams['augmentation_params'])
        elif hparams['lorentz_coeff']:
            self.hparams['dataset_params']['transform'] = None # Feat Transform must be applied after Lorentz Augmentation
        else:
            self.hparams['dataset_params']['transform'] = FeaturesTransformation(pid=self.pid)

        self.learning_rate = hparams['learning_rate']
        self.batch_size = hparams['batch_size']
        

    def forward(self, x, transformed=True):
        x_tmp = x.clone()
        if transformed==False:
            dimensions = torch.cat([x.x[:, 0:1], x.pos], dim=1)
            dimensions = transform_coordinates(dimensions)
            if self.pid:
                pids = x.x[:, 1:]
                dimensions = torch.cat([dimensions, pids], dim=1)
            x_tmp.x = dimensions
            x_tmp.pos = dimensions[:, 2:4]
        return self.model(x_tmp)
    
    def train_dataloader(self):
        return DataLoader(self.datasetCl(set='train', **self.hparams['dataset_params']), 
                          num_workers=8,
                          batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.datasetCl(set='val', **self.hparams['dataset_params']), 
                          num_workers=8,
                          batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.datasetCl(set='test', **self.hparams['dataset_params']),
                          num_workers=8,
                          batch_size=self.batch_size, shuffle=False)

    def training_step(self, batch, batch_idx):
        if not self.hparams.get('lorentz_coeff'):
            y_hat = self(batch)   
            loss = self.criterion(y_hat, batch['y'])
        else:
            y_hat = self(batch, transformed=False)   
            loss = self.criterion(y_hat, batch['y'], batch, self)
            
        self.log('train_loss', loss, batch_size=batch['y'].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.hparams.get('lorentz_coeff'):
            y_hat = self(batch)   
            loss = self.criterion(y_hat, batch['y'])
        else:
            y_hat = self(batch, transformed=False)   
            loss = self.criterion(y_hat, batch['y'], batch, self) 

        self.log('val_loss', loss, batch_size=batch['y'].size(0))
    
    def test_step(self, batch, batch_idx):
        if not self.hparams.get('lorentz_coeff'):
            y_hat = self(batch)   
            loss = self.criterion(y_hat, batch['y'])
        else:
            y_hat = self(batch, transformed=False)   
            loss = self.criterion(y_hat, batch['y'], batch, self)

        accuracy = (y_hat.argmax(dim=1) == batch['y']).float().mean()
        self.log('test_accuracy', accuracy, batch_size=batch['y'].size(0))
        self.log('test_loss', loss, batch_size=batch['y'].size(0))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.hparams.get('use_scheduler'):
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, patience=3, verbose=True),
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        return optimizer

    def evaluate_invariance(self, betas=None, logdir=None):
        if betas is None:
            betas = np.arange(0.0, 1.01, 0.02)

        accuracy_array = []
        # samples withouth augmentation
        dl = DataLoader(self.datasetCl(set='test', n_limit=5000, transform=None, pid=self.pid), 
                num_workers=8, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        with torch.no_grad():
            for beta in tqdm(betas, desc="Evaluating invariance"):
                accuracy = []
                beta_transform = LorentzAugmentation_FeatureTransformation(max_beta=beta, p=1.0, fixed_beta=True, pid=self.pid)
                tdata = []
                for batch in dl:
                    tdata += [beta_transform(d) for d in batch.to_data_list()]
                tdataloader = DataLoader(tdata, num_workers=8, batch_size=self.batch_size, shuffle=False)    
                
                for data in tdataloader:
                    y_hat = self(data.to('cuda'))
                    accuracy.append((y_hat.argmax(dim=1) == data['y']).float().mean().to('cpu'))
                accuracy_array.append(np.mean(accuracy))

        import wandb
        data = [[x, y] for (x, y) in zip(betas, accuracy_array)]
        table = wandb.Table(data=data, columns=["beta", "accuracy"])
        wandb.log({"Invariance" : wandb.plot.scatter(table, "beta", "accuracy")})
        if logdir is not None:
            res = pd.DataFrame(data, columns=["beta", "accuracy"])
            res.to_csv(logdir / f"invariance_beta.csv", index=False)