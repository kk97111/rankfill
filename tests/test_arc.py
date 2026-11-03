import numpy as np
from rankfill.arc import fixed_anchor_order,decode_query,query_proposals
from rankfill.weighted_cdf import WeightedCDF


def test_no_anchor_fallback_and_candidate_conservation():
    docs=np.array([10,20,30]);model=WeightedCDF()
    fixed,proposals=query_proposals(model,docs,np.array([.9,.8,.7]),np.array([],int),np.array([]))
    np.testing.assert_array_equal(fixed,docs)
    result,stats,_=decode_query(fixed,[],proposals['LearnedCDF'],.85)
    np.testing.assert_array_equal(result,docs);assert stats['accepted']==0


def test_exact_anchor_order_and_cycle_rejection():
    docs=np.array([1,2,3,4])
    fixed=fixed_anchor_order(docs,np.array([0,2]),np.array([.1,.8]))
    np.testing.assert_array_equal(fixed,[3,2,1,4])
    result,stats,_=decode_query(fixed,[1,3],[(.99,1,3,0,2),(.9,4,2,3,1)],.85)
    assert stats['cycle_rejected']==1 and stats['accepted']==1
    assert sorted(result)==sorted(docs) and list(result).index(3)<list(result).index(1)


def test_threshold_is_fixed_and_not_changed_by_other_query_calls():
    fixed=np.array([1,2,3]);p=[(.8,2,1,1,0)]
    before=decode_query(fixed,[1],p,.7)[0]
    decode_query(np.array([5,6]),[5],[(.999,6,5,1,0)],.7)
    np.testing.assert_array_equal(decode_query(fixed,[1],p,.7)[0],before)
    np.testing.assert_array_equal(decode_query(fixed,[1],p,.8)[0],fixed)


def test_zero_initialized_learned_weights_equal_uniform():
    model=WeightedCDF();docs=np.arange(12);scores=np.linspace(.9,.1,12)
    _,p=query_proposals(model,docs,scores,np.array([0,3,8]),np.array([.7,.8,.2]))
    assert p['LearnedCDF']==p['UniformCDF']
