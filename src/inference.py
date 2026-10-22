"""Readable uncertainty helpers: comparisons condition on a fixed experimental protocol."""
import numpy as np
from scipy.stats import t
import pandas as pd


def corrected_interval(values, ratio=.25):
    # Nadeau–Bengio correction avoids treating overlapping CV training folds as independent.
    values=np.asarray(values,dtype=float)
    n=len(values); sd=values.std(ddof=1)
    se=sd*np.sqrt(1/n+ratio)
    margin=t.ppf(.975,n-1)*se
    p=2*t.sf(abs(values.mean()/se),n-1) if se>0 else (1. if values.mean()==0 else 0.)
    return dict(mean=float(values.mean()),sd=float(sd),ci_low=float(values.mean()-margin),ci_high=float(values.mean()+margin),p_value=float(p),n=n,validation_train_ratio=ratio)


def holm(frame):
    frame=frame.copy(); order=np.argsort(frame.p_value.to_numpy()); adjusted=np.zeros(len(frame)); high=0.
    for rank,idx in enumerate(order):
        high=max(high,min(1.,(len(order)-rank)*frame.p_value.iloc[idx])); adjusted[idx]=high
    frame['p_holm']=adjusted
    return frame


def paired_comparisons(folds, reference, group='configuration', metric='pr_auc'):
    pivot=folds.pivot(index='fold',columns=group,values=metric)
    rows=[]
    for label in pivot:
        if label==reference:continue
        rows.append({'option':label,'reference':reference,'metric':metric,**corrected_interval(pivot[label]-pivot[reference])})
    return holm(pd.DataFrame(rows))
