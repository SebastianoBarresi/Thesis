from pathlib import Path
import torch
from torch_geometric.data import InMemoryDataset
from data.reader import read_data
from data.transforms import data2torch



class TopDataset(InMemoryDataset):
    def __init__(self, set='train', n_limit=-1, root='dataset/toptagging', transform=None, pre_transform=None, pid=False):
        assert set in ['train', 'val', 'test']
        self.set = set
        self.n_limit = n_limit

        super(TopDataset, self).__init__(root=root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        self.raw_paths = [f'{self.set}.h5']
        return [f'{self.set}.h5']
    
    @property
    def processed_file_names(self):
        fname = f"{self.set}_processed_data{self.n_limit if self.n_limit != -1 else ''}"
        return [f"{fname}.pt"]

    def process(self):
        jets, labels = read_data(Path(self.raw_dir)/f'{self.set}.h5', n_limit=self.n_limit)
        transf_jets = data2torch(jets, labels)

        data, slices = self.collate(transf_jets)
        torch.save((data, slices), self.processed_paths[0])



class QGDataset(InMemoryDataset):
    def __init__(self, set='train', n_limit=-1, root='dataset/quarkgluon', transform=None, pre_transform=None, pid=False):
        assert set in ['train', 'val', 'test']
        self.set = set
        self.n_limit = n_limit
        self.pid = pid

        super(QGDataset, self).__init__(root=root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        self.raw_paths = sorted(Path(self.raw_dir).glob(f'{set}_*.parquet'))
        return self.raw_paths
    
    @property
    def processed_file_names(self):
        fname = f"{self.set}_processed_data{self.n_limit if self.n_limit != -1 else ''}"
        if self.pid:
            fname += "_pid"
        return [f"{fname}.pt"]

    def process(self):
        if self.set == 'train':
          paths = sorted(Path(self.raw_dir).glob(f'{self.set}_*.parquet'))[:15]
        elif self.set == 'val': 
          paths = sorted(Path(self.raw_dir).glob('train_*.parquet'))[15:]
        else:
          paths = sorted(Path(self.raw_dir).glob(f'{self.set}_*.parquet'))
        jets, labels = read_data(paths, dataset="QG", n_limit=self.n_limit, pid=self.pid)
        transf_jets = data2torch(jets, labels, pid=self.pid)

        data, slices = self.collate(transf_jets)
        torch.save((data, slices), self.processed_paths[0])