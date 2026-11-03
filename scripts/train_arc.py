#!/usr/bin/env python3
"""Train a budget-safe episode-based conditional residual distribution."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rankfill.weighted_cdf import AnchorEpisodes, WeightedCDF, collate_episodes, episode_loss, FEATURES, FEATURE_SETS
from rankfill.query_validation import split_query_rows


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compact',required=True)
    p.add_argument('--scores',required=True)
    p.add_argument('--ids',required=True)
    p.add_argument('--selected',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch',type=int,default=256)
    p.add_argument('--max-targets',type=int,default=4);p.add_argument('--max-queries',type=int,default=0)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda:0')
    p.add_argument('--bandwidth',type=float,default=.01);p.add_argument('--pair-weight',type=float,default=.2)
    p.add_argument('--residual-weight',type=float,default=1.)
    p.add_argument('--feature-set','--feature-groups',choices=tuple(FEATURE_SETS),default='full')
    p.add_argument('--mapper-checkpoint',required=True)
    p.add_argument('--new-tag',default='phi_new_x1')
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--query-val-size',type=int,default=0,help='Use the same global query split as mapper/acquisition; 0 retains legacy split')
    p.add_argument('--query-val-seed',type=int,default=42)
    return p


def main():
    a=build_parser().parse_args()
    if a.epochs<1 or a.batch<1 or a.max_targets<1:raise ValueError('positive epochs/batch/targets required')
    if not np.isfinite([a.residual_weight,a.pair_weight]).all() or min(a.residual_weight,a.pair_weight)<0 or a.residual_weight+a.pair_weight<=0:
        raise ValueError('loss weights must be finite, nonnegative, and not both zero')
    out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint_best.pt').exists():raise FileExistsError('use a new output directory; existing checkpoint preserved')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed)
    scores=np.load(ROOT/a.scores,mmap_mode='r');ids=np.load(ROOT/a.ids,mmap_mode='r')
    assert scores.shape==ids.shape and scores.shape[1]==1000
    print('[load] compact refreshed-only scores; no full teacher corpus/teacher rankings',flush=True)
    with np.load(ROOT/a.compact) as c:
        data=AnchorEpisodes(c,scores,ids,np.load(ROOT/a.selected))
    queries=np.asarray(sorted(data.query_rows),dtype=np.int64)
    if a.query_val_size:
        if a.max_queries:raise ValueError('fixed global query split cannot be combined with max-queries')
        train_rows,val_rows=split_query_rows(len(ids),a.query_val_size,a.query_val_seed)
        train=train_rows[np.isin(train_rows,queries)];val=val_rows[np.isin(val_rows,queries)]
        if not len(train) or not len(val):raise ValueError('empty fixed-query partition')
    else:
        np.random.default_rng(a.seed).shuffle(queries)
        if a.max_queries:queries=queries[:a.max_queries]
        if len(queries)<10:raise ValueError('at least ten queries required')
        nval=max(1,round(.1*len(queries)));val,train=queries[:nval],queries[nval:]
    assert not np.intersect1d(train,val).size
    np.savez_compressed(out/'query_split.npz',train=train,validation=val)
    config=vars(a)|{'features':FEATURES,'depth':1000,'support_fraction':.5,
                   'train_queries':len(train),'validation_queries':len(val),
                   'active_features':[FEATURES[i] for i in FEATURE_SETS[a.feature_set]],
                   'feature_indices':FEATURE_SETS[a.feature_set],
                   'input_paths':{k:str((ROOT/getattr(a,k)).resolve()) for k in ('compact','scores','ids','selected','mapper_checkpoint')},
                   'non_S_teacher_training':False,'checkpoint_selection':'qtrain-validation total loss only',
                   'kernel':'logistic','feature_scales':[1,10,10,1,1,100]}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    model=WeightedCDF(a.bandwidth,a.feature_set).to(a.device);optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
    best=float('inf');log=[]
    def run(qs,epoch,training):
        model.train(training);sums=np.zeros(3);count=skipped=0
        if training:qs=np.random.default_rng(a.seed+epoch).permutation(qs)
        mask_seed=a.seed+10000+epoch if training else a.seed+900000
        for start in range(0,len(qs),a.batch):
            eps=[data.episode(int(q),mask_seed,a.max_targets) for q in qs[start:start+a.batch]]
            n=sum(e is not None for e in eps);skipped+=len(eps)-n
            batch=collate_episodes(eps,a.device)
            if batch is None:continue
            with torch.set_grad_enabled(training):
                loss,nll,bce=episode_loss(model,batch['features'],batch['centers'],batch['mask'],
                    batch['target'],batch['gap'],batch['rank'],a.pair_weight,batch['sample_weight'],
                    batch['pair_labels'],batch['pair_valid'],residual_weight=a.residual_weight)
                if not torch.isfinite(loss):raise FloatingPointError('non-finite episode loss')
                if training:
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            sums+=np.asarray([loss.item(),nll.item(),bce.item()])*n;count+=n
            if training and (start//a.batch)%100==0:
                print(f'[epoch {epoch}] {start+n}/{len(qs)} loss={loss.item():.5f} nll={nll.item():.5f} bce={bce.item():.5f}',flush=True)
        if not count:raise ValueError('no valid episodes')
        return dict(zip(('loss','nll','flip_bce'),(sums/count).tolist()))|{'episodes':count,'skipped':skipped}
    for epoch in range(1,a.epochs+1):
        tr=run(train,epoch,True);va=run(val,epoch,False)
        row={'epoch':epoch}|{'train_'+k:v for k,v in tr.items()}|{'val_'+k:v for k,v in va.items()}
        log.append(row)
        with (out/'train_log.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(log)
        checkpoint={'state_dict':model.state_dict(),'config':config,'epoch':epoch,'validation':va}
        torch.save(checkpoint,out/'checkpoint_last.pt')
        if va['loss']<best:
            best=va['loss'];torch.save(checkpoint,out/'checkpoint_best.pt')
        print(json.dumps(row),flush=True)
    (out/'TRAINING_COMPLETE').touch()
    print('[done] training; qdev not used for selection',flush=True)


if __name__=='__main__':main()
