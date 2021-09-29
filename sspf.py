import torch
import torch.nn.functional as F
import os
import argparse
import numpy as np
from sklearn.datasets import make_blobs
from utils import progress_bar, output, read_data
import time

parser = argparse.ArgumentParser(description='SSPF Clustering')
parser.add_argument('-K', default=100, type=int, help='K: number of clusters, default=100')
parser.add_argument('-N', default=1000000, type=int, help='N: number of data points, default=1m')
parser.add_argument('-m', default=100, type=int, help='m: feature dim, default=100')
parser.add_argument('-b', default=200, type=int, help='b: batch size, equal to U dim0, default=200')
parser.add_argument('-d', default=None, type=str, help='dataset path, if using existing numpy data')
parser.add_argument('--lr', default=1, type=float, help='learning rate for U, default=1')
parser.add_argument('-v', '--verbose', action="store_true", help='verbose')
parser.add_argument('--cpu', action="store_true", help='cpu only')
parser.add_argument('--no_faiss', action="store_true", help='do not use FAISS, only use sklearn nearest neighbor search. WARNING: this will be slow.')
args = parser.parse_args()

print('Hyper parameters: lr =', args.lr)

if args.verbose:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
if args.no_faiss:
    from sklearn.neighbors import NearestNeighbors
else:
    import faiss

device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

# load existing data, numpy format
if args.d:
    data, labels = read_data(args.d)
    args.N, args.m = data.shape
    args.K = len(set(labels))

# generate data
else:
    print('Generate data')
    data, labels = make_blobs(n_samples=args.N, centers=args.K, n_features=args.m) 
    data = data.astype(np.float32)
    
assert args.b <= args.N
assert args.b >= args.K
    
# build data loader
data = torch.from_numpy(data)
print('Data shape:', data.shape, 'N_clusters:', args.K)

start = time.time()
sqrtm = np.sqrt(args.m)
dataset = torch.utils.data.TensorDataset(data)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.b, shuffle=True, drop_last=True, num_workers=os.cpu_count())
cnt_thres = 10 if len(train_loader) > 10 else len(train_loader) - 1

# define parameter
if device == 'cuda':
    u = torch.cuda.FloatTensor(args.b, args.m).normal_().requires_grad_()
else:
    u = torch.FloatTensor(args.b, args.m).normal_().requires_grad_()

def L_minibatch(u, x):
    if u.shape[0] > x.shape[0]: #handle the last batch
        x = x.repeat(np.int(np.ceil(u.shape[0]/x.shape[0])), 1)
        x = x[u.shape[0]]

    u1 = torch.unsqueeze(x, dim=1).expand(args.b, args.b, args.m)
    u2 = torch.unsqueeze(u, dim=0).expand(args.b, args.b, args.m)
    v = torch.sum((u1 - u2) ** 2, dim=2) / sqrtm
    qk = F.softmax(-v, dim=1)
    return torch.sum(qk * v) / args.b 

# early stopping condition
def early_stop(loss_seq, diff_thres, cnt_curr, cnt_thres=10):
    if len(loss_seq) < cnt_thres:
        return False
    if abs(loss_seq[-1] - loss_seq[-2]) < loss_seq[-1]*diff_thres:
        return True
    if loss_seq[-1] > loss_seq[-2] and cnt_curr[0] >= cnt_thres:
        return True
    if loss_seq[-1] > loss_seq[-2] and cnt_curr[0] < cnt_thres:
        cnt_curr[0] += 1
    else:
        cnt_curr[0] = 0
    return False

def train(loss_func, optimizer):
    cnt = [0]
    loss_seq = []
    for epoch in range(1000):
        if args.verbose:
            print('Epoch: %d' % epoch)
        train_loss = 0
        for batch_idx, (inputs,) in enumerate(train_loader):
            x = inputs.to(device)
            optimizer.zero_grad()
            loss = loss_func(u, x)
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss += loss.item()
            if args.verbose:
                progress_bar(batch_idx, len(train_loader), 'Loss: %.3f' % (train_loss/(batch_idx+1)), term_width=term_width)
            loss_seq.append(train_loss/(batch_idx+1))
            if early_stop(loss_seq, 1e-2, cnt, cnt_thres=cnt_thres):
                if epoch > 0: return
                else: break

# define optimizer
optimizer_u = torch.optim.Adam([u], lr=args.lr)

print('Start Training ...')

# warm start
u.data = data[:u.shape[0]].to(device)

# main loop
train(L_minibatch, optimizer_u)

# final assignments 
print('Optimization step takes (seconds): ', time.time()-start)
print('Get assignments ...')

u = u.detach().cpu().numpy()

if args.no_faiss:
    nbrs = NearestNeighbors(n_neighbors=1).fit(u)
    D, I = nbrs.kneighbors(data)
    pred_labels = np.squeeze(I)

else:
    if 'StandardGpuResources' in dir(faiss) and not args.cpu: # GPU FAISS
        torch.cuda.empty_cache()
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0
        index = faiss.GpuIndexFlatL2(res, args.m, cfg)
    else: # CPU FAISS
        index = faiss.IndexFlatL2(args.m)
    index.add(np.ascontiguousarray(u))
    
    # when using GPU FAISS, N has to be smaller than 1.6e8 to make sure fit into GPU memory, for a TitanX.
    if 'StandardGpuResources' in dir(faiss) and args.N > 1.6*1e8:
        data = np.ascontiguousarray(data.numpy())
        I = []
        chunk_len = int(1.6*1e8)
        i = 0
        while i < len(data):
            D, I_single = index.search(data[i:i+chunk_len], 1)
            i += chunk_len
            I.append(I_single)
        I = np.vstack(I)
    else:
        D, I = index.search(np.ascontiguousarray(data.numpy()), 1)
    pred_labels = np.squeeze(I)

# the cpu version have tiny sets of size 1 or 2, due to pytorch, should be filtered out.
pred_set = list(set(pred_labels))
if len(pred_set) > args.b * 4 / 5:
    f_size = args.N // args.b // 100 # filter out clusters that has a size < N/b/100 
    filtered_set = list(filter(lambda e: len(pred_labels[pred_labels==e]) > f_size, pred_set))
    for e in pred_set:
        pred_labels[pred_labels==e] = -1 if e not in filtered_set else e

elapsed_time = time.time()-start
print('Get results ...')
output(elapsed_time, labels, pred_labels)


