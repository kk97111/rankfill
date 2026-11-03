"""Exact float32 inner-product retrieval on normalized vectors."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

@torch.inference_mode()
def retrieve(q,gallery,depth=1000,batch=32,prefix=None):
    shape=(len(q),depth)
    if prefix is None:
        ids=np.empty(shape,np.int32);scores=np.empty(shape,np.float32)
    else:
        if Path(str(prefix)+'.ids.npy').exists():raise FileExistsError(f'partial cache needs inspection: {prefix}')
        ids=np.lib.format.open_memmap(str(prefix)+'.ids.npy',mode='w+',dtype=np.int32,shape=shape)
        scores=np.lib.format.open_memmap(str(prefix)+'.scores.npy',mode='w+',dtype=np.float32,shape=shape)
    for start in range(0,len(q),batch):
        query=F.normalize(torch.tensor(np.asarray(q[start:start+batch],np.float32),device=gallery.device),dim=1)
        value,index=torch.topk(query@gallery.T,depth,dim=1)
        ids[start:start+batch]=index.cpu().numpy();scores[start:start+batch]=value.cpu().numpy()
        if start==0 or (start//batch)%100==0:print(f'[retrieve] {start+len(query)}/{len(q)}',flush=True)
    if prefix is not None:ids.flush();scores.flush()
    return ids,scores

@torch.inference_mode()
def normalized_gallery(corpus,device,batch=65536):
    g=torch.empty(corpus.shape,dtype=torch.float32,device=device)
    for start in range(0,len(corpus),batch):
        g[start:start+batch]=F.normalize(torch.tensor(np.asarray(corpus[start:start+batch],np.float32),device=device),dim=1)
    return g
