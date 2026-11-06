"""Exact streaming fp32 weight quantiles on CPU, without a full-model sort/copy."""
import math
import struct
from kit.nm_common import need,ready


def summarize(paths,chunk=65536):
    import numpy as np
    import torch
    paths=list(paths);ready(paths,'Fisher shards absent')
    def chunks():
        for path in paths:
            values=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            for name,value in values.items():
                need(value.dtype==torch.float32,'Fisher summary requires fp32 weights')
                flat=value.reshape(-1)
                for start in range(0,flat.numel(),chunk):yield name,flat[start:start+chunk].numpy()
    high=np.zeros(65536,dtype=np.int64);count=0;total=0.;square=0.;maximum=0.;positive=0;by_tensor={}
    for name,x in chunks():
        need(bool(np.isfinite(x).all() and (x>=0).all()),'nonfinite/negative Fisher weights')
        # Nonnegative IEEE float32 bit order is numeric order, including +0.
        bits=x.view(np.uint32)&np.uint32(0x7fffffff)
        high+=np.bincount((bits>>16).astype(np.int64),minlength=65536)
        value=float(x.sum(dtype=np.float64));squared=float(np.square(x.astype(np.float64)).sum())
        count+=x.size;total+=value;square+=squared;positive+=int(np.count_nonzero(x));maximum=max(maximum,float(x.max(initial=0)))
        item=by_tensor.setdefault(name,dict(count=0,sum=0.,sum_squares=0.));item['count']+=x.size;item['sum']+=value;item['sum_squares']+=squared
    ready(count>0 and total>0 and math.isfinite(total),'empty/zero Fisher weights')
    qs=(0.,.25,.5,.75,.9,.99,1.);orders={k for q in qs for k in (math.floor((count-1)*q),math.ceil((count-1)*q))}
    cumulative=high.cumsum();buckets={k:int(np.searchsorted(cumulative,k,side='right')) for k in orders}
    lows={b:np.zeros(65536,dtype=np.int64) for b in set(buckets.values())}
    for _,x in chunks():
        bits=x.view(np.uint32)&np.uint32(0x7fffffff);prefix=bits>>16
        for b,hist in lows.items():
            selected=bits[prefix==b]&65535
            if selected.size:hist+=np.bincount(selected.astype(np.int64),minlength=65536)
    selected={}
    for k,b in buckets.items():
        below=int(cumulative[b-1]) if b else 0
        low=int(np.searchsorted(lows[b].cumsum(),k-below,side='right'))
        selected[k]=struct.unpack('!f',struct.pack('!I',(b<<16)|low))[0]
    quantiles={}
    for q in qs:
        location=(count-1)*q;lo,hi=math.floor(location),math.ceil(location)
        quantiles[str(q)]=selected[lo]+(selected[hi]-selected[lo])*(location-lo)
    return dict(parameters=count,mean=total/count,sum=total,maximum=maximum,quantiles=quantiles,
        effective_support=total*total/square,nonzero=positive,nonzero_fraction=positive/count,tensors=by_tensor,
        method='exact two-pass radix order statistics, linear interpolation, unique local shards')
