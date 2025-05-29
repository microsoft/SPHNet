# Introduction
Welcome! This repository is the official implement of SPHNet in the ICML'25 paper: [Efficient and Scalable Density Functional Theory Hamiltonian Prediction through Adaptive Sparsity](https://arxiv.org/abs/2502.01171). 

![](SPHNet.png)

SPHNet is an efficient and scalable equivariant network that incorporates adaptive SParsity into Hamiltonian prediction task. SPHNet employs two innovative sparse gates to selectively constrain non-critical interaction combinations, significantly reducing tensor product computations while maintaining accuracy. 



# Setup environment
## Local environment
By running the following command, you will create a virtual environment with all the necessary packages to run the code.
Note that it is better to use CUDA driver more than 520, due to the requirement of the package `CUDA Toolkit 12`.
```bash
# Create a model training environment.
conda env create -n sphnet -f environment.yaml
conda activate sphnet
```

# Proprocess data
First you need to proprocess the data. We support data file in lmdb and mdb format. You need to calculate the initial guess of Hamiltonian matrix and also the short range and long range edge index used in the Vectorial Node Interaction Blocks. ``src/dataset/preprocess_data.py`` is an example of converting md17 dataset into the mdb format. You can modify this file to preprocess dataset.

We also provide an example data in the ./example_data folder. You can use
```
dataset = MdbDataset(path = '/data/hami/qh9stable/data.mdb',remove_init=False)
```
to read the data directly. Note that when ``remove_init=True``, the ``fock`` = Hamiltonian - initial guess.

# Model training
## Halmintonian model train on local machine
There are three config files that set the model and training process. The config file in config/model is the model configuration. The config file in config/schedule set the learning schedule. The config/config.yaml is the overall configuration file. Please see the comments in config/config.yaml for more details.
1. Run the following command to train the model.
    ```bash
     python pipelines/train.py --wandb=True --wandb-group="train" --data_name="qh9_stable_iid" --basis="def2-svp" --dataset-path="/path/to/your/data.mdb" \
    ```
    Specifically, when running QH9 dataset, we support four kinds of data_name: "qh9_stable_iid","qh9_stable_ood","qh9_dynamic_mol", and "qh9_dynamic_geo". The basis should set to "def2-svp". When running PubChemQH dataset, set the data_name to "pubchem" and the basis to "def2-tzvp". In the mean time, you should modify to model config file to suit different basis and dataset (Please see the comments in config/model/sphnet.yaml for detail).
2. Run the following command to test the model.
    ```bash
     python pipelines/test.py --wandb=True --wandb-group="test" --data_name="qh9_stable_iid" --basis="def2-svp" --dataset-path="/path/to/your/data.mdb" \
    ```


# Citation

```
@article{sphnet,
  title={Efficient and Scalable Density Functional Theory Hamiltonian Prediction through Adaptive Sparsity},
  author={Luo, Erpai and Wei, Xinran and Huang, Lin and Li, Yunyang and Yang, Han and Xia, Zaishuo and Wang, Zun and Liu, Chang and Shao, Bin and Zhang, Jia},
  journal={arXiv preprint arXiv:2502.01171},
  year={2025}
}
```