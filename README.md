# Introduction
Welcome! This repository is the official implement of SPHNet, an efficient and scalable equivariant network that incorporates adaptive SParsity into Hamiltonian prediction task. SPHNet employs two innovative sparse gates to selectively constrain non-critical interaction combinations, significantly reducing tensor product computations while maintaining accuracy. See the paper for more details: https://arxiv.org/abs/2502.01171.

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
First you need to proprocess the data. We support data file in lmdb and mdb format. You need to calculate the initial guess of Hamiltonian matrix and also the short range and long range edge index used in the Vectorial Node Interaction Blocks. ``src/data_prepare/preprocess.py`` is an example of converting md17 dataset into the mdb format. You can modify this file to preprocess dataset.


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

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.