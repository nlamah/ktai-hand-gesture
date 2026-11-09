"""Exact RBF distance decomposition for repeated grouped permutation predictions.

No model refit: K=exp(-gamma*(distance_all-distance_group+distance_group_permuted)).
LIBSVM evaluates the same saved coefficients on those kernels. Near-zero pair
margins fall back to the ordinary estimator. Float64 scratch maps bound RAM.
"""
from pathlib import Path
import tempfile

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC,_libsvm
from threadpoolctl import threadpool_limits


class RBFPermutation:
    def __init__(self,model,values,chunk=128):
        self.model=model;self.raw=np.asarray(values);self.chunk=chunk
        self.learner=model[-1] if isinstance(model,Pipeline) else model
        if not isinstance(self.learner,SVC) or self.learner.kernel!='rbf' or self.learner.break_ties:
            raise ValueError('Only the frozen dense RBF C-SVC profile is supported.')
        transformed=model[:-1].transform(values) if isinstance(model,Pipeline) else values
        self.x=np.ascontiguousarray(transformed,dtype=np.float64)
        self.sv=np.ascontiguousarray(self.learner.support_vectors_,dtype=np.float64)
        if self.x.shape[1]!=self.raw.shape[1] or not np.isfinite(self.x).all():
            raise ValueError('The shortcut requires finite one-to-one per-column preprocessing.')
        self.directory=tempfile.TemporaryDirectory(prefix='ktai-rbf-permutation-')
        self.base=self._distances(self.x,self.sv,'base')
        self.support=np.arange(len(self.sv),dtype=np.int32)
        self.fallback_rows=0

    def _distances(self,x,sv,name):
        result=np.memmap(Path(self.directory.name)/(name+'.bin'),dtype=np.float64,mode='w+',shape=(len(x),len(sv)))
        svnorm=np.einsum('ij,ij->i',sv,sv)
        with threadpool_limits(limits=1,user_api='blas'):
            for start in range(0,len(x),self.chunk):
                stop=min(start+self.chunk,len(x));part=x[start:stop]
                distance=-2*np.matmul(part,sv.T)
                distance+=np.einsum('ij,ij->i',part,part)[:,None]
                distance+=svnorm[None,:]
                np.maximum(distance,0,out=distance);result[start:stop]=distance
        # Numerical guard against unexpected scale/cancellation in the scratch kernel.
        direct=np.sum((x[:8,None,:]-sv[None,:8,:])**2,axis=2)
        np.testing.assert_allclose(result[:8,:8],direct,atol=1e-9,rtol=1e-10)
        return result

    def _labels(self,kernel):
        m=self.learner
        decisions=_libsvm.decision_function(np.ascontiguousarray(kernel),self.support,
            np.empty((0,0),dtype=np.float64),m._n_support,m._dual_coef_,m._intercept_,m._probA,m._probB,
            svm_type=0,kernel='precomputed',degree=m.degree,coef0=m.coef0,gamma=m._gamma,
            cache_size=m.cache_size)
        votes=np.zeros((len(kernel),len(m.classes_)),dtype=np.int32);k=0
        for a in range(len(m.classes_)):
            for b in range(a+1,len(m.classes_)):
                positive=decisions[:,k]>0
                votes[:,a]+=positive;votes[:,b]+=~positive;k+=1
        return m.classes_[votes.argmax(axis=1)],np.min(np.abs(decisions),axis=1)<1e-4

    def _predict(self,group=None,columns=None,order=None):
        result=np.empty(len(self.x),dtype=self.learner.classes_.dtype)
        for start in range(0,len(self.x),self.chunk):
            stop=min(start+self.chunk,len(self.x))
            distance=np.array(self.base[start:stop],copy=True)
            if group is not None:
                distance-=group[start:stop];distance+=group[order[start:stop]]
            np.maximum(distance,0,out=distance)
            distance*=-self.learner._gamma;np.exp(distance,out=distance)
            pred,near=self._labels(distance)
            if near.any():
                rows=np.flatnonzero(near)+start
                raw=self.raw[rows].copy()
                if group is not None:raw[:,columns]=self.raw[order[rows]][:,columns]
                pred[near]=self.model.predict(raw);self.fallback_rows+=int(near.sum())
            result[start:stop]=pred
        return result

    def baseline(self):return self._predict()

    def group_predictions(self,columns,orders):
        group=self._distances(self.x[:,columns],self.sv[:,columns],'group')
        try:return [self._predict(group,columns,order) for order in orders]
        finally:
            group._mmap.close()
            (Path(self.directory.name)/'group.bin').unlink()

    def close(self):
        self.base._mmap.close();self.directory.cleanup()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
