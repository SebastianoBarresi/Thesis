from typing import Any
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


    
def data2torch(dataset_jets, labels, pid=False):
    """
    Transforms the readed dataset into the jet coordinates, obtaining a torch Tensor.
    """

    if pid:
        dataset = [Data(x=torch.tensor(np.hstack((jet[:, 0:1], jet[:, 4:]))).float(), pos=torch.tensor(jet[:, 1:4]).float(), y=torch.tensor(lab).type(torch.LongTensor)) for jet, lab in zip(dataset_jets, labels)]
    else:
        dataset = [Data(x=torch.tensor(jet[:, 0:1]).float(), pos=torch.tensor(jet[:, 1:4]).float(), y=torch.tensor(lab).type(torch.LongTensor)) for jet, lab in zip(dataset_jets, labels)]
    
    return [d for d in dataset if d is not None]



def random_lorentz_matrix(max_beta, device, rotation=False, fixed_beta=False):
    # random rotation matrix on xy plane
    if fixed_beta:
        beta = torch.tensor(max_beta).float()
    else:
        beta = np.random.rand() * max_beta

    #rotation on x-y axis
    theta = torch.rand(1) * 2 * np.pi
    rotation_matrix = torch.tensor([[1, 0, 0, 0],     
                                    [0, torch.cos(theta), -torch.sin(theta), 0],
                                    [0, torch.sin(theta), torch.cos(theta), 0],
                                    [0, 0, 0, 1]], device=device).float()
    
    phi = torch.rand(1) * np.pi
    rotation_matrix = torch.matmul(torch.tensor([[1, 0, 0, 0],
                                                 [0, torch.cos(phi), 0, torch.sin(phi)],
                                                 [0, 0, 1, 0],
                                                 [0, -torch.sin(phi), 0, torch.cos(phi)]], device=device).float(), rotation_matrix)

    #boost on x-t axis
    gamma = np.float32(1.0 / np.sqrt(1.0 - beta**2))
    tranform_matrix = torch.tensor([[gamma, -beta*gamma, 0, 0],
                                    [-beta*gamma, gamma, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], device=device).float()
    
    if rotation:
        tranform_matrix = torch.matmul(rotation_matrix, tranform_matrix)
    return tranform_matrix



def transform_coordinates(jet):
    """
    input:
    jet = [E, px, py, pz]
    =====
    output:
    v = [log(E), log(pt), eta, phi]
    """
    #assert (jet[:, 0] > 0.0).all(), "Energy must be positive"

    j_eta = lambda x, y, z: (z / (x**2 + y**2).sqrt()).arcsinh()
    j_pt = lambda x, y: (x*x +y*y).sqrt()
    j_phi = lambda x, y: torch.atan2(y, x)
    j_delta_phi = lambda j1, j2: (j_phi(j1[:,1], j1[:,2]) - j_phi(j2[1], j2[2]) + np.pi) % (2*np.pi) - np.pi

    jet_p4 = jet.sum(axis=0)

    jet_eta = j_eta(jet_p4[1], jet_p4[2], jet_p4[3])
    _jet_etasign = torch.sign(jet_eta)
    if _jet_etasign==0: _jet_etasign = 1

    eta = j_eta(jet[:, 1], jet[:, 2], jet[:, 3])
    pt = j_pt(jet[:, 1], jet[:, 2])

    v = torch.stack([jet[:,0].log(), 
                      pt.log(), 
                      (eta - jet_eta) * _jet_etasign, 
                      j_delta_phi(jet, jet_p4)], axis=1)
    return v.float()



@functional_transform('lorentz_transform')
class LorentzAugmentation(BaseTransform):
    """
    Random Lorentz transformation of the input position (Rotation + x-axis Lorentz transformation).
    """
    def __init__(self, p=0.5, max_beta=1.0, rotation=True):
        self.p = p
        self.max_beta = max_beta
        self.rotation = rotation

    def __call__(self, data: Data) -> Data:
        if torch.rand(1) > self.p:
            return data
        
        dimensions = torch.cat([data.x[:, 0:1], data.pos], dim=1) # assuming the first dimension is the energy
        tranform_matrix = random_lorentz_matrix(self.max_beta, device=data.pos.device, rotation=self.rotation)
        # apply the transformation to the jet
        dimensions = torch.matmul(tranform_matrix, dimensions.T).T
        data.x = dimensions[:, 0:1] # TODO: can be more (cat(old, transformed))
        data.pos = dimensions[:, 1:]
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.p)
    


@functional_transform('lorentz_features_transform')
class LorentzAugmentation_FeatureTransformation(BaseTransform):
    """
    Random Lorentz transformation of the input position (Rotation + x-axis Lorentz transformation).
    """
    def __init__(self, p=0.5, max_beta=1.0, rotation=False, fixed_beta=False, pid=False):
        self.p = p
        self.max_beta = max_beta
        self.rotation = rotation
        self.fixed_beta = fixed_beta
        self.pid = pid

    def __call__(self, data: Data) -> Data:
        """
        input:
        Data(x={E (, extra_features)}, pos={px, py, pz}, y={label})
        =====
        output:
        Data(x={log_E, log_pt, eta, phi (, extra_features)}, pos={eta, phi}, y={label})
        """
        dimensions = torch.cat([data.x[:, 0:1], data.pos], dim=1) # assuming the first dimension is the energy

        if torch.rand(1) < self.p:
            tranform_matrix = random_lorentz_matrix(self.max_beta, device=data.pos.device, rotation=self.rotation, fixed_beta = self.fixed_beta)
            dimensions = torch.matmul(tranform_matrix, dimensions.T).T

        dimensions = transform_coordinates(dimensions) 
        if self.pid:
            pids = data.x[:, 1:]
            dimensions = torch.cat([dimensions, pids], dim=1)
        data.x = dimensions
        data.pos = dimensions[:, 2:4]
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.p)



@functional_transform('features_transform')
class FeaturesTransformation(BaseTransform):
    def __init__(self, pid=False):
        self.pid = pid
        super().__init__()
    
    def __call__(self, data: Any) -> Any:
        dimensions = torch.cat([data.x[:, 0:1], data.pos], dim=1)
        dimensions = transform_coordinates(dimensions)
        if self.pid:
            pids = data.x[:, 1:]
            dimensions = torch.cat([dimensions, pids], dim=1)
        data.x = dimensions
        data.pos = dimensions[:,2:4]
        return data
    


class LorentzAugmentation_LorentzNet(object):
    """
    Random Lorentz transformation of the input position (Rotation + x-axis Lorentz transformation).
    """
    def __init__(self, p=0.5, max_beta=1.0, rotation=False, fixed_beta=False):
        self.p = p
        self.max_beta = max_beta
        self.rotation = rotation
        self.fixed_beta = fixed_beta

    def __call__(self, data):
        if torch.rand(1) > self.p:
            return data
        
        tranform_matrix = random_lorentz_matrix(self.max_beta, device=data.device, rotation=self.rotation, fixed_beta=self.fixed_beta)
        # apply the transformation to the jet
        data = torch.matmul(tranform_matrix.type(data.dtype), data.T).T
        return data
    