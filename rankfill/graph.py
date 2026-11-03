"""Acyclic comparison insertion and stable topological decoding."""
import heapq
import numpy as np

def reaches(outgoing,start,target):
    stack=[start];seen={start}
    while stack:
        u=stack.pop()
        if u==target:return True
        for v in outgoing[u]:
            if v not in seen:seen.add(v);stack.append(v)
    return False

def try_add_edge(outgoing,before,after):
    if before==after:return False
    if after in outgoing[before]:return True
    if reaches(outgoing,after,before):return False
    outgoing[before].add(after);return True

def stable_graph_order(baseline,outgoing):
    baseline=np.asarray(baseline);indegree=np.zeros(len(baseline),np.int32)
    for targets in outgoing:
        for v in targets:indegree[v]+=1
    heap=[i for i in range(len(baseline)) if indegree[i]==0];heapq.heapify(heap);order=[]
    while heap:
        u=heapq.heappop(heap);order.append(u)
        for v in outgoing[u]:
            indegree[v]-=1
            if indegree[v]==0:heapq.heappush(heap,v)
    if len(order)!=len(baseline):raise ValueError('cycle survived guarded insertion')
    return baseline[np.asarray(order)]
