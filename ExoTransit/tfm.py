from . import transit
import numpy as np
from .utils import NamedParam

__all__ = ['direct_tfm', 'direct_tfm_with_detrend', 'get_detrend_factor']

def direct_tfm(param,t,per,ldmethod='quad'):
    if np.ndim(t) == 1 and np.ndim(t[0])==0 and isinstance(param, (dict,NamedParam)):
        par = [param['tcen'],param['b'],param['rsa'],param['rprs'],param['fout']]
        if ldmethod=='uniform':
            func = transit.occultuniform
        else:
            c1 = param.get('c1', 0)
            c2 = param.get('c2', 0)
            c3 = param.get('c3', 0)
            c4 = param.get('c4', 0)
            if ldmethod=='linear':
                par += [c2,0]
                func = transit.occultquad
            if ldmethod=='quadratic' or ldmethod=='quad':
                par += [c2,c4]
                func = transit.occultquad
            if ldmethod=='nonlinear':
                par += [c1,c2,c3,c4]
                func = transit.occultnonlin
        return transit.modeltransit(par,func,per,t)
    modelop = []
    if np.ndim(t) == 1 and np.ndim(t[0])==0:
        for i in range(len(param)): modelop.append(direct_tfm(param[i],t,per,ldmethod))
    elif isinstance(param, (dict,NamedParam)):
        for i in range(len(t)): modelop.append(direct_tfm(param, t[i], per, ldmethod))
    else:
        for i in range(len(t)): modelop.append(direct_tfm(param[i],t[i],per,ldmethod))
    return modelop


def direct_tfm_with_detrend(param,t,per,detrendvar,ldmethod='quad'):
    if np.ndim(t)==1 and np.ndim(t[0])==0 and isinstance(param, NamedParam):
        detrend_fac = 1
        for varname in detrendvar:
            var, varlen = detrendvar[varname]
            par = [param[varname+f'_{i}'] for i in range(varlen-1,-1,-1)]+[1]
            detrend_fac *= np.polyval(par,var)
        return  direct_tfm(param,t,per,ldmethod)*detrend_fac
    modelop = []
    if np.ndim(t)==1 and np.ndim(t[0])==0:
        for i in range(len(param)): modelop.append(direct_tfm_with_detrend(param[i],t,per,detrendvar,ldmethod))
    elif isinstance(param, (dict,NamedParam)):
        for i in range(len(t)): modelop.append(direct_tfm_with_detrend(param, t[i], per, detrendvar[i], ldmethod))
    else:
        for i in range(len(t)): modelop.append(direct_tfm_with_detrend(param[i],t[i],per,detrendvar[i],ldmethod))
    return modelop

def get_detrend_factor(param,detrendvar={}):
    if isinstance(param, dict):
        detrend_fac = 1
        for varname in detrendvar:
            var, varlen = detrendvar[varname]
            par = [param[varname + f'_{i}'] for i in range(varlen - 1, -1, -1)] + [1]
            detrend_fac *= np.polyval(par, var)
        return detrend_fac
    detrend_fac = []
    if isinstance(param, dict):
        for i in range(len(param)):
            detrend_fac.append(get_detrend_factor(param[i], detrendvar))
    if isinstance(detrendvar, dict):
        for i in range(len(detrendvar)):
            detrend_fac.append(get_detrend_factor(param, detrendvar[i]))
    for i in range(len(param)):
        detrend_fac.append(get_detrend_factor(param[i], detrendvar[i]))
    return detrend_fac

