
## Setup

### Create a conda environment
Our code is based on python 3.10. We suggest you create a conda environment for this project to avoid conflicts with others.
```bash
conda create -n AGGD python=3.10
```
Then, you can activate the conda environment:
```bash
conda activate AGGD
```

### Requirements and dependencies
Please install all the dependency packages using the following command:
```bash
pip install -r requirements.txt
```

#### Download source code of Contriever
In order to run Contriever model, please clone the Contriever source code.
```bash

git clone https://github.com/facebookresearch/contriever.git
```

### Datasets
You do not need to download datasets. When running our code, the datasets will be automatically downloaded and saved in `datasets`.



### Reproduce the exerimental result
In order to reproduce the experimental result 
```bash

bash run.sh
```

## Evaluation
To evaluate the result, first, run 
```bash

bash ./scripts/eval_beir.sh
```


Then run `
```bash

bash run_eval.sh
```
