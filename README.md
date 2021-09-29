# Code for SSPF Clustering
## List of Files:
* Readme.md: this readme file.
* requirements.txt: the requirements (use pip install) to run SSPF.
* requirements_others.txt: the requirements to run other methods.
* sspf.py: the SSPF code.
* other_methods.py: other algorithms tested in the paper.
* utils.py: utilities used in the code.

---
## Minimum Effort to Run:
### 1. Install required packages:

```bash
pip install -r requirements.txt
```

WARNING: here we only install and use the sklearn's nearest-neighbor-search, which is slow. To enable fast computation and reproduce paper's results,
please install FAISS (GPU and CPU version). The steps can be found at https://github.com/facebookresearch/faiss .

### 2. Run sspf.py to perform clustering on 1 million data points (Gaussian blobs), where feature dimension is 100, true number of clusters is 100, batch size is 200.

```bash
python3 sspf.py --no_faiss
```

This will generate results like the following:

```bash
Hyper parameters: lr = 1
Generate data
Data shape: torch.Size([1000000, 100]) N_clusters: 100
Start Training ...
Optimization step takes (seconds):  5.57167649269104
Get assignments ...
Get results ...
Total time (seconds): 9.958990573883057
NMI Score: 0.9954640682925067
# of clusters: 97
```

---
## Full Arguments

The sspf.py accept following arguments:

```bash
usage: sspf.py [-h] [-K K] [-N N] [-m M] [-b B] [-d D] [--lr LR] [-v] [--cpu] [--no_faiss]
SSPF Clustering
optional arguments:
  -h, --help     show this help message and exit
  -K K           K: number of clusters, default=100
  -N N           N: number of data points, default=1m
  -m M           m: feature dim, default=100
  -b B           b: batch size, equal to U dim0, default=200
  -d D           dataset path, if using existing numpy data
  --lr LR        learning rate for U, default=1
  -v, --verbose  verbose
  --cpu          cpu only
  --no_faiss     do not use FAISS, only use sklearn nearest-neighbor search. WARNING: this will be slow.
```

In order to use FAISS fast nearest-neighbor search and reproduce the paper's results, please follow https://github.com/facebookresearch/faiss to install FAISS.

---
## Run other methods
### 1. Supported methods:
The other_methods.py support: minibatch_kmeans, kmeans, dbscan,  hdbscan, hierarchical, kmeans_cuda, xmeans, gmeans, finch, faiss_kmeans, faiss_kmeans_gpu, ksum, rcc, optics, ap, spectral.

To install pyclustering (for xmeans, gmeans), hdbscan and pyflann (for finch), please run:

```bash
pip install Cython
pip install -r requirements_others.txt
```

To use a specified method, simply run the following as an example:

```bash
python3 other_methods.py --method minibatch_kmeans -N 100000 -m 100 -K 100
```

### 2. Some methods require installing additional packages:
* faiss_kmeans and faiss_kmeans_gpu: please follow https://github.com/facebookresearch/faiss to install FAISS.
* kmeans_cuda: please follow https://github.com/src-d/kmcuda to install libKMCUDA.
* finch: please copy https://github.com/ssarfraz/FINCH-Clustering/blob/master/python/finch.py to the current directory.
* ksum: please follow https://github.com/ShenfeiPei/KSUMS to install KSUMS.
* rcc: please copy the folder 'pyrcc/' from https://github.com/yhenon/pyrcc to the current directory.