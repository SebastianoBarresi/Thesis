# Lorentz-invariant augmentation for high-energy physics deep-learning models

**Abstract:** In recent years, machine learning models for Jet tagging in high-energy physics have gained considerable attention. However, many existing approaches overlook the physical invariants that jets must adhere to, particularly the fundamental spacetime symmetry governed by Lorentz transformations. Setting this statement as the starting point of this work, it is proposed a model-agnostic training strategy that incorporates theory-guided data augmentation to simulate the effects of Lorentz transformations on jet data. The study starts with focusing on the state-of-the-art baseline ParticleNet, a neural network architecture designed for the direct processing of particle clouds for Jet tagging. To evaluate the effectiveness of the proposed approach, several experiments are conducted with different augmentation strategies and assess the performance of the augmented models on the widely used top-tagging and quark-gluon reference datasets. The results show that even a small application of the data augmentation strategy increases the robustness of the model to Lorentz boost attacks, i.e., high transformation $\beta$. While the accuracy of the baseline model decreases rapidly with increasing intensity of the transformation $\beta$, the augmented models exhibit more stable performance. Remarkably, models that underwent a moderate level of augmentation demonstrated a statistically significant performance boost on transformations beyond the ones seen at train time. Then the same experimental setup is applied to a second state-of-the-art baseline LorentzNet, a neural network architecture developed to be invariant to Lorentz transformations by design. The performance of the model are also evaluated both on the top quark tagging and quark-gluon reference datasets, making possible a full comparison between each experimental setup applied to the two chosen models. The results shows that LorentzNet is more robust to Lorentz boost attacks than ParticleNet, as it is expected to be. Nevertheless the application of the data augmentation strategy to an already invariant architecture, tends to further increase the robustness of the model. This finding highlights the potential of the model-agnostic data augmentation strategy in enhancing model accuracy and robustness while preserving the essential physical properties of the jets

## Table of Contents
<details>
<summary>Click to expand</summary>

- [Requirements](#requirements)
- [Datasets](#datasets)
  - [Top quark tagging](#top-quark-tagging)
  - [Quark-Gluon tagging](#quark-gluon-tagging)
- [Training](#training)

## Requirements
- Use the following command to install required packages.
    - ```pip install -r requirements.txt```
 
## Datasets

### Top quark tagging
The Top quark tagging dataset can be downloaded from [zenodo](https://doi.org/10.5281/zenodo.2603256). The default path is [`./dataset/toptagging/raw`](./dataset/toptagging/raw). 
The converted dataset used by LorentzNet can be downloaded from [this link](https://osf.io/7u3fk/?view_only=8c42f1b112ab4a43bcf208012f9db2df). Its default path is [`./dataset/toptagging/converted`](./dataset/toptagging/converted). 

### Quark-gluon tagging
The Quark-gluon tagging dataset can be downloaded from [cern](https://hqu.web.cern.ch/datasets/QuarkGluon/QuarkGluon.tar). The default path is [`./dataset/quarkgluon/raw`](./dataset/quarkgluon/raw). 
In the LorentzNet implementation the dataset is automatically downloaded from the package [EnergyFlow](https://energyflow.network/docs/datasets/) to its default path [`./dataset/quarkgluon/cache`](./dataset/quarkgluon/cache). 

## Training
To train the models:

```sh
python train.py --config confing/model.yaml
```

All the insights and the model with best validation accuracy are automatically saved from [Weight&Biases](https://wandb.ai/)
