import pandas as pd
import numpy as np
from tqdm import trange



def _get_data(table, n_limit, n_particles=200, filename=""):
    arr = []
    labels = table[table.columns[-1]].values[0:n_limit]
    table = table[table.columns[:-6]].values

    for ix in trange(n_limit, desc=f"Loading {filename}"):
        jet = table[ix]
        particles = jet.reshape(n_particles, 4)
        particles = particles[particles[:, 0] != 0]
        arr.append(particles)
    return arr, labels



def _get_data_QG(table, n_limit, n_particles=200, filename="", pid=False):
    arr = []
    labels = table[table.columns[0]].values[0:n_limit]
    if pid:
        pid_table = table[table.columns[14:]].values
    table = table[table.columns[7:11]].values
    

    for ix in trange(n_limit, desc=f"Loading {filename}"):
        jet = table[ix]
        particles = np.hstack((jet[3].reshape(-1,1), jet[0].reshape(-1,1), jet[1].reshape(-1,1), jet[2].reshape(-1,1)))
        if pid:
            jet_pid = pid_table[ix]
            particles_pid = np.hstack((jet_pid[0].reshape(-1,1), jet_pid[1].reshape(-1,1), jet_pid[2].reshape(-1,1), 
                                       jet_pid[3].reshape(-1,1), jet_pid[4].reshape(-1,1), jet_pid[5].reshape(-1,1)))
            particles = np.hstack((particles, particles_pid))
        particles = particles[particles[:, 0] != 0]
        arr.append(particles)
    return arr, labels



def read_TQ(filepath, n_limit=-1):
    table_data = pd.read_hdf(filepath, key='table').reset_index(drop=True)
    if n_limit == -1:
        n_limit = len(table_data)
    ret, ret2 = _get_data(table_data, min(n_limit, len(table_data)), filename=filepath.name)
    return ret, ret2



def read_QG(filepaths, n_limit=-1, pid=False):
    ret, ret2 = [], []
    if n_limit != -1:
        filepaths = filepaths[:1]
    for filepath in filepaths:
        table_data = pd.read_parquet(filepath).reset_index(drop=True)
        if n_limit == -1:
            n_limit = len(table_data)
        mid, mid2 = _get_data_QG(table_data, min(n_limit, len(table_data)), filename=filepath.name, pid=pid)
        ret = ret + mid
        ret2 = np.concatenate((ret2,mid2), axis=0)
    return ret, ret2



def read_data(filename, dataset="TQ", n_limit=-1, pid=False):
    if dataset == "TQ":
        four_momentum, labels = read_TQ(filename, n_limit=n_limit)
    elif dataset == "QG":
        four_momentum, labels = read_QG(filename, n_limit=n_limit, pid=pid)
    else:
        raise ValueError("Dataset not supported")
    return four_momentum, labels