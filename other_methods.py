import sys
import argparse
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
import time
from utils import progress_bar, output, read_data

parser = argparse.ArgumentParser(description='Other Clustering Methods')
parser.add_argument('--method', default='kmeans', type=str, help='method to run, default=kmeans')
parser.add_argument('-K', default=100, type=int, help='K: number of clusters, default=100')
parser.add_argument('-N', default=1000000, type=int, help='N: number of data points, default=1m')
parser.add_argument('-m', default=100, type=int, help='m: feature dim, default=100')
parser.add_argument('-d', default=None, type=str, help='dataset path, if using existing numpy data')
parser.add_argument('--maxk', default=200, type=int, help='specify maximum k')
args = parser.parse_args()

# load existing numpy data
if args.d:
    data, labels = read_data(args.d)
    args.N, args.m = data.shape
    args.K = len(set(labels))

# generate data
else:
    print('Generate data')
    data, labels = make_blobs(n_samples=args.N, centers=args.K, n_features=args.m)
    data = data.astype(np.float32)

print('Data shape:', data.shape, 'N_centers:', args.K)

if args.method == 'minibatch_kmeans':
    start = time.time()
    clusterer = MiniBatchKMeans(n_clusters=args.K, random_state=0, max_iter=1000).fit(data) # default max_iter=100, the NMI is much worse
    output(time.time()-start, labels, clusterer.labels_)

if args.method == 'kmeans':
    start = time.time()
    clusterer = KMeans(n_clusters=args.K, random_state=0).fit(data)
    output(time.time()-start, labels, clusterer.labels_)

if args.method == 'dbscan':
    from sklearn.cluster import DBSCAN
    start = time.time()
    clusterer = DBSCAN(n_jobs=-1, eps=20).fit(data)  # tested best eps value for make_blobs
    output(time.time()-start, labels, clusterer.labels_)

if args.method == 'optics':
    from sklearn.cluster import OPTICS
    start = time.time()
    clusterer = OPTICS(n_jobs=-1, max_eps=50).fit(data)
    output(time.time()-start, labels, clusterer.labels_)

if args.method == 'ap':
    from sklearn.cluster import AffinityPropagation
    start = time.time()
    clusterer = AffinityPropagation(random_state=5).fit(data)
    output(time.time()-start, labels, clusterer.labels_)

if args.method == 'spectral':
    from sklearn.cluster import SpectralClustering
    start = time.time()
    clusterer = SpectralClustering(n_clusters=args.K, assign_labels="discretize", random_state=0).fit(data)
    output(time.time()-start, labels, clusterer.labels_)
 
if args.method == 'hdbscan':
    import hdbscan
    start = time.time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10).fit(data)
    output(time.time()-start, labels, clusterer.labels_)

if args.method == 'hierarchical':
    from sklearn.cluster import AgglomerativeClustering
    start = time.time()
    clusterer = AgglomerativeClustering(n_clusters=args.K).fit(data)
    output(time.time()-start, labels, clusterer.labels_)

if args.method == 'kmeans_cuda':
    # sys.path.append("/path/to/kmcuda/src/")
    from libKMCUDA import kmeans_cuda
    start = time.time()
    centroids, pred_labels = kmeans_cuda(data, args.K, verbosity=0, device=1)
    output(time.time()-start, labels, pred_labels)

if args.method == 'xmeans':
    from pyclustering.cluster.xmeans import xmeans
    start = time.time()
    xmeans_instance = xmeans(data, kmax=args.maxk)
    xmeans_instance.process()
    pred_labels = xmeans_instance.predict(data)
    output(time.time()-start, labels, pred_labels)

if args.method == 'gmeans':
    from pyclustering.cluster.gmeans import gmeans
    start = time.time()
    gmeans_instance = gmeans(data, k_max=args.maxk).process()
    pred_labels = gmeans_instance.predict(data)
    output(time.time()-start, labels, pred_labels)

if args.method == 'finch':
    from finch import FINCH
    assert args.maxk >= args.K
    start = time.time()
    c, num_clust, req_c = FINCH(data)
    best_nmis, best_est, avg_nmis, avg_nclus = -1, np.inf, [], []
    elapsed_time = time.time() - start

    for j in range(c.shape[1]): # pick the best during the partition
        pred_labels = c[:,j]
        nmis = output(elapsed_time, labels, pred_labels, verbose=False)
        print('Scores@partition', j, 'cluste#', num_clust[j], 'scores%', nmis)
        if num_clust[j] <= args.maxk and num_clust[j] >= args.K/5:
            avg_nmis.append(nmis)
            avg_nclus.append(num_clust[j])
            if abs(num_clust[j] - args.K) <= best_est:
                best_nmis, n_clus = nmis, num_clust[j]
                best_est = abs(n_clus - args.K)

    avg_nmis, avg_nclus = np.mean(avg_nmis), np.mean(avg_nclus)
    print('time: ' + str(elapsed_time))
    print('avg scores (nmi, #clus): ', avg_nmis, avg_nclus)
    print('best scores (nmi, #clus):', best_nmis, n_clus)

if args.method == 'rcc':
    from pyrcc import RccCluster
    start = time.time()
    clusterer = RccCluster()
    pred_labels = clusterer.fit(data)
    output(elapsed_time, labels, pred_labels)

if args.method == 'faiss_kmeans':
    import faiss
    start = time.time()
    kmeans = faiss.Kmeans(args.m, args.K)
    kmeans.train(data)
    D, I = kmeans.index.search(data, 1)
    pred_labels = np.reshape(I, -1)
    output(time.time()-start, labels, pred_labels)

if args.method == 'faiss_kmeans_gpu':
    import faiss
    ngpu = 1
    start = time.time()
    clus = faiss.Clustering(args.m, args.K)
    clus.verbose = False
    res = [faiss.StandardGpuResources() for i in range(ngpu)]
    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)
    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(res[0], args.m, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], args.m, flat_config[i]) for i in range(ngpu)]
        index = faiss.IndexReplicas()
        for sub_index in indexes:
            index.addIndex(sub_index)
    clus.train(data, index)

    if args.N > 1e8:
        I = []
        chunk_len = int(1e8)
        i = 0
        while i < len(data):
            D, I_single = index.search(data[i:i+chunk_len], 1)
            i += chunk_len
            I.append(I_single)
        I = np.vstack(I)
    else:
        D, I = index.search(data, 1)

    pred_labels = np.reshape(I, -1)
    output(time.time()-start, labels, pred_labels)

if args.method == 'ksum':
    # sys.path.append('/path/to/KSUMS')
    import funs as Ifuns
    from KSUMS import KSUMS
    start = time.time()
    data = data.astype(np.double)

    knn = 20
    D = Ifuns.EuDist2(data, data, squared=True)
    np.fill_diagonal(D, -1)
    ind_M = np.argsort(D, axis=1)
    np.fill_diagonal(D, 0)
    
    NN = ind_M[:, :knn]
    NND = Ifuns.matrix_index_take(D, NN)
    
    obj = KSUMS(NN.astype(np.int32), NND, args.K)
    obj.clu()
    pred_labels = obj.y_pre

    output(time.time()-start, labels, pred_labels)

