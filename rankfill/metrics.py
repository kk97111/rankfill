"""Alignment Recall and binary NDCG against full-new top-K."""
import numpy as np

def per_query_metrics(ranked,teacher,k):
    weights=1/np.log2(np.arange(k)+2);idcg=float(weights.sum())
    align=np.empty(len(ranked),dtype=np.float64);rba=np.empty(len(ranked),dtype=np.float64)
    for qi in range(len(ranked)):
        relevant=set(np.asarray(teacher[qi,:k]).tolist())
        hit=np.fromiter((doc in relevant for doc in ranked[qi,:k]),dtype=bool,count=k)
        align[qi]=hit.mean();rba[qi]=float(np.sum(weights*hit)/idcg)
    return align,rba
