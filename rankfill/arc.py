"""Query-independent residual-CDF proposals and fixed-threshold decoding."""
import numpy as np
import torch

from rankfill.weighted_cdf import anchor_features, flip_from_cdf
from rankfill.graph import try_add_edge, stable_graph_order


def fixed_anchor_order(mapped, ranks, known):
    mapped=np.asarray(mapped);ranks=np.asarray(ranks,dtype=np.int64);known=np.asarray(known)
    if ranks.shape!=known.shape:raise ValueError('anchor-only labels must match anchor ranks')
    fixed=mapped.copy()
    fixed[ranks]=mapped[ranks[np.argsort(-known,kind='stable')]]
    return fixed


@torch.inference_mode()
def query_proposals(model,mapped,scores,ranks,known,device='cpu',methods=('UniformCDF','LearnedCDF'),
                    repair_radius=5,head_depth=20,initial_order=None):
    """Return (probability, before-id, after-id, target-rank, anchor-rank)."""
    if (not isinstance(repair_radius,(int,np.integer)) or repair_radius<1
            or not isinstance(head_depth,(int,np.integer)) or head_depth<1):
        raise ValueError('repair_radius and head_depth must be positive integers')
    mapped=np.asarray(mapped);scores=np.asarray(scores,dtype=np.float64)
    ranks=np.asarray(ranks,dtype=np.int64);known=np.asarray(known,dtype=np.float64)
    base=mapped if initial_order is None else np.asarray(initial_order)
    if base.shape!=mapped.shape or len(np.unique(base))!=len(base) or not np.array_equal(np.sort(base),np.sort(mapped)):
        raise ValueError('initial_order must be a permutation of the mapped candidate IDs')
    if initial_order is None:
        fixed=fixed_anchor_order(mapped,ranks,known)
    else:
        anchor_truth=dict(zip(map(int,mapped[ranks]),known))
        base_ranks=np.array([i for i,d in enumerate(base) if int(d) in anchor_truth],dtype=np.int64)
        fixed=fixed_anchor_order(base,base_ranks,[anchor_truth[int(base[i])] for i in base_ranks])
    output={name:[] for name in methods}
    if len(ranks)==0:return fixed,output
    pos={int(doc):i for i,doc in enumerate(mapped)}
    base_pos={int(doc):i for i,doc in enumerate(base)}
    is_anchor=np.zeros(len(mapped),bool);is_anchor[ranks]=True
    delta=known-scores[ranks];delta_by_rank=dict(zip(ranks,delta))
    local=[]
    for fp,doc in enumerate(fixed):
        h=pos[int(doc)]
        if is_anchor[h]:continue
        if fp>=head_depth+repair_radius:continue
        for ap in range(max(0,fp-repair_radius),min(len(fixed),fp+repair_radius+1)):
            s=pos[int(fixed[ap])]
            if not is_anchor[s] or min(fp,ap)>=head_depth:continue
            gap=scores[h]-scores[s]
            if gap==0 and initial_order is None:continue
            h_before=base_pos[int(mapped[h])]<base_pos[int(mapped[s])]
            before,after=(int(mapped[s]),int(mapped[h])) if h_before else (int(mapped[h]),int(mapped[s]))
            # Thresholds and residuals remain in mapped-score coordinates. Only
            # the direction being corrected follows the actual serving order.
            direction=gap if initial_order is None else (1. if h_before else -1.)
            local.append((h,s,delta_by_rank[s]-gap,direction,before,after))
    if not local:return fixed,output
    targets=np.unique([pair[0] for pair in local]);target_index={h:i for i,h in enumerate(targets)}
    features=torch.from_numpy(anchor_features(scores,targets,ranks,delta)).to(device)
    mask=torch.ones(features.shape[:-1],dtype=torch.bool,device=device)
    weights=model.weights(features,mask)
    idx=torch.tensor([target_index[p[0]] for p in local],device=device)
    centers=torch.as_tensor(delta,dtype=torch.float64,device=device).expand(len(local),-1)
    thresholds=torch.tensor([[p[2]] for p in local],dtype=torch.float64,device=device)
    gaps=torch.tensor([p[3] for p in local],dtype=torch.float64,device=device)
    for name,w in [('LearnedCDF',weights[idx]),('UniformCDF',torch.ones_like(weights[idx])/len(ranks))]:
        if name not in methods:continue
        probability=flip_from_cdf(model.cdf(w,centers,thresholds).squeeze(1),gaps).cpu().numpy()
        output[name]=[(float(prob),p[4],p[5],p[0],p[1]) for prob,p in zip(probability,local)]
    return fixed,output


def decode_query(fixed,anchor_docs,proposals,threshold):
    """Only this query's proposals affect its ordering; ties follow proposal order."""
    pos={int(doc):i for i,doc in enumerate(fixed)};outgoing=[set() for _ in fixed]
    anchor_set=set(map(int,anchor_docs));anchors=[int(d) for d in fixed if int(d) in anchor_set]
    for before,after in zip(anchors[:-1],anchors[1:]):
        assert try_add_edge(outgoing,pos[before],pos[after])
    chosen=sorted((p for p in proposals if p[0]>threshold),key=lambda p:-p[0])
    accepted=[];rejected=0
    for proposal in chosen:
        _,before,after,_,_=proposal
        if try_add_edge(outgoing,pos[before],pos[after]):accepted.append(proposal)
        else:rejected+=1
    return stable_graph_order(np.asarray(fixed),outgoing),dict(proposals=len(proposals),selected=len(chosen),accepted=len(accepted),cycle_rejected=rejected),accepted
