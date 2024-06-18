import random
import numpy as np

def partition_data(args, y_train):
    n_train = y_train.shape[0]

    # split dataset
    partition = args.partition
    n_parties = args.n_parties
    alpha = args.alpha
    K = args.num_classes # num_cls
    if partition == 'iid': # IID case
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    # distribution-based label imbalance
    elif partition == 'noniid-c-dir': # label obey Dirichlet distribution $p \sim Dir_N(\alpha)$
        min_size = 0
        min_require_size = 10
        
        N = n_train
        net_dataidx_map = {}
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    # quantity-based label imbalance: each party only has cnum different labels; for each label, random and equally divide them and distribute them to party which need them
    elif partition > "noniid-cnum-0" and partition <= "noniid-cnum-9": # each client contain how many labels?
        num = int(partition.split('-')[2]) # label number
        if num == 10:
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(num):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
        else:
            times = [0 for _ in range(K)] # how many times each label is picked for different client
            contain = [] # store the labels for each clients
            for i in range(n_parties):
                current = [i % K]
                times[i % K] += 1
                j = 1
                while j < num:
                    ind = random.randint(0, K - 1)
                    if ind not in current:
                        j += 1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, times[i]) 
                ids = 0
                for j in range(n_parties):
                    if i in contain[j]: # if label i is picked by client j
                        net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                        ids += 1

    # quantity skew
    elif partition == 'iid-diff-quantity': 
        print('quantity skew: clients with different size of iid samples')
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * len(idxs))
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs, proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
    else:
        # noise-based feature imbalance is controlled by args.noise, and it always combined with iid
        raise NotImplementedError('Unknown imbalance type')
    
    return net_dataidx_map