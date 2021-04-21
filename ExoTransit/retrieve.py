from ExoTransit.MCMC_engine import *
from ExoTransit.utils import *
from ExoTransit.tfm import *
import numpy as np
import os
from george import GP, BasicSolver

__all__ = ['model_transit_lightcurve','mcmc_moves']


def test_func():
    import tqdm
    import time
    for i in tqdm.trange(10):
        time.sleep(0.5)

class model_transit_lightcurve(object):

    per = 0
    ldmethod = 'quad'

    def __init__(self, data=None, param={}, bounds={}, fixed=[], group=None, **kwargs):
        """

        :param data:
        :param param:
        :param bounds:
        :param group:
        """
        self.data = []
        self.param, self.param_name, self.param_bounds = np.array([]), np.array([]), []
        self.Ndata = 0
        self.index_param_indiv = []
        self.index_bounds_indiv = []
        self.index_param_common = []
        self.index_bounds_common = []
        self.group_indiv = []
        self.fixed = []
        self.group_common = []
        self.detrendvar = {}
        self._varind = 0
        self.gp = GP(solver=BasicSolver)
        self._mcmc_parinit_optimized = False
        self.add_data(data, param, bounds, fixed, group, **kwargs)

    def add_data(self, data, param, bounds, fixed=[], group=None, **kwargs):
        if data:
            if type(data)==str: data = np.loadtxt(data, unpack=True, **kwargs)
            if self.Ndata==0: self.data = data
            if self.Ndata==1: self.data = [self.data]
            if self.Ndata>0: self.data.append(data)
            parlen = len(self.param)
            bounlen = len(self.param_bounds)
            if param:
                if isinstance(param,dict) and type(param)!=NamedParam: param = NamedParam(param)
                self.param_name = np.append(self.param_name, param.keys())
                self.param = np.append(self.param, [param[key] for key in param])
                fixedkeys = [param[fx] if type(fx)==int else fx for fx in fixed]
                self.param_bounds += [bounds[key] for key in bounds if key not in fixedkeys]
                self.group_indiv.append(group)
                self.fixed += [(fx if type(fx)==int else list(param).index(fx))+parlen for fx in fixed]
            self.index_param_indiv.append(np.arange(len(param))+parlen)
            self.index_bounds_indiv.append(np.arange(len(bounds))+bounlen)
            if self.Ndata==1: self.detrendvar = [self.detrendvar]
            if self.Ndata>0: self.detrendvar.append({})
            self.Ndata += 1

    def add_indiv_param(self, param, bounds, fixed=[], dataindex=-1):
        if self.Ndata<1: return
        if param:
            if isinstance(param, dict) and type(param) != NamedParam: param = NamedParam(param)
            parlen = len(self.param)
            bounlen = len(self.param_bounds)
            self.param_name = np.append(self.param_name, param.keys())
            self.param = np.append(self.param, [param[key] for key in param])
            fixedkeys = [param[fx] if type(fx) == int else fx for fx in fixed]
            self.param_bounds += [bounds[key] for key in bounds if key not in fixedkeys]
            self.index_param_indiv[dataindex] = np.append(self.index_param_indiv[dataindex], np.arange(len(param))+parlen)
            self.index_bounds_indiv[dataindex] = np.append(self.index_bounds_indiv[dataindex], np.arange(len(bounds))+bounlen)
            self.fixed += [parlen+fx for fx in fixed]

    def add_common_param(self, param, bounds={}, fixed=[], group=None): # TODO: ignore bounds keys if fixed is not []
        if self.Ndata<1: return
        if param:
            if isinstance(param, dict) and type(param) != NamedParam: param = NamedParam(param)
            parlen = len(self.param)
            bounlen = len(self.param_bounds)
            self.param_name = np.append(self.param_name, param.keys())
            self.param = np.append(self.param, [param[key] for key in param])
            fixedkeys = [param[fx] if type(fx) == int else fx for fx in fixed]
            self.param_bounds += [bounds[key] for key in bounds if key not in fixedkeys]
            self.group_common.append(group)
            self.fixed += [(fx if type(fx) == int else list(param.keys()).index(fx)) + parlen for fx in fixed]
            self.index_param_common.append(np.arange(len(param)) + parlen)
            self.index_bounds_common.append(np.arange(len(bounds)) + bounlen)

    def add_detrend_param(self, variable=0, name='', dataindex=-1, coeff=[], bounds=[], fixed=[]):
        if type(variable)==int:
            varnames = ['t','flux','err']
            # var = self.__getattribute__(varnames[variable])[self.dataindex[dataindex]]
            if self.Ndata==1 and dataindex in [0,-1]:
                var = self.data[variable]
            elif self.Ndata>1: var = self.data[dataindex][variable]
            if not name: name = varnames[variable]
        else:
            var = variable
            if not name:
                name = 'var'+str(self._varind)
                self._varind += 1
        name = 'det_'+name
        if coeff:
            parlen = len(self.param)
            bounlen = len(self.param_bounds)
            self.param = np.append(self.param, coeff)
            self.param_name = np.append(self.param_name, [name+f'_{i}' for i in range(len(coeff))])
            self.param_bounds += bounds
            if self.Ndata==1 and dataindex in [0,-1]: self.detrendvar.update({name : (var,len(coeff))})
            elif self.Ndata>1:  self.detrendvar[dataindex].update({name : (var,len(coeff))})
            self.index_param_indiv[dataindex] = np.append(self.index_param_indiv[dataindex], np.arange(len(coeff))+parlen)
            self.index_bounds_indiv[dataindex] = np.append(self.index_bounds_indiv[dataindex], np.arange(len(bounds))+bounlen)
            self.fixed += [parlen+fx for fx in fixed]

    def get_named_param(self, param):
        if type(param) != np.ndarray: param = np.array(param)
        if self.Ndata==1:
            nparam = NamedParam(zip(self.param_name[self.index_param_indiv[0]],param[self.index_param_indiv[0]]))
            for j,icarr in enumerate(self.index_param_common):
                if not self.group_indiv[0] or not self.group_common[j] or self.group_indiv[0]==self.group_common[j]:
                    nparam.update(zip(self.param_name[icarr],param[icarr]))
            # print(nparam)
            return nparam
        paramseg = [NamedParam() for _ in self.index_param_indiv]
        for i,iiarr in enumerate(self.index_param_indiv):
            for j,icarr in enumerate(self.index_param_common):
                if not self.group_indiv[i] or not self.group_common[j] or self.group_indiv[i]==self.group_common[j]:
                    paramseg[i].update(zip(self.param_name[icarr],param[icarr]))
            paramseg[i].update(zip(self.param_name[iiarr],param[iiarr]))
        return paramseg

    def _model_function(self, param, t, named=False, detrend=True):
        if not named: param = self.get_named_param(param)
        if detrend:
            return direct_tfm_with_detrend(param,t,self.per,self.detrendvar,self.ldmethod)
        return direct_tfm(param,t,self.per,self.ldmethod)

    # @staticmethod
    # def _contains_gppar(param):
    #     if isinstance(param,(dict,NamedParam)):
    #         # gpa = param.pop('gpa',None)
    #         # gptau = param.pop('gptau',None)
    #         # return gpa is not None and gptau is not None
    #         return 'gpa' in param and 'gptau' in param
    #     cond = []
    #     for par in param:
    #         gpa = par.pop('gpa', None)
    #         gptau = par.pop('gptau', None)
    #         cond.append(gpa is not None and gptau is not None)
    #     return cond

    @staticmethod
    def _contains_gppar(param):
        if isinstance(param, (dict, NamedParam)):
            return 'gpa' in param and 'gptau' in param
        return ['gpa' in par and 'gptau' in par for par in param]

    @staticmethod
    def _contais_detrendvar(detrendvar):
        if isinstance(detrendvar, dict):
            return len(detrendvar)==0
        return [len(dtv)==0 for dtv in detrendvar]

    def log_likelihood_gp(self, param):
        if self.Ndata==0: return
        if not self._calcgp: return
        nparam = self.get_named_param(param)
        if self.Ndata==1:
            detfac = get_detrend_factor(nparam, self.detrendvar) if self._calcdetrend else 1
            self.gp.kernel = kernel(nparam['gpa'],nparam['gptau'])
            if np.any(np.isnan(detfac)): print('detfac nan')
            if np.any(np.isinf(detfac)): print('detfac inf')
            if np.any(np.isnan(nparam['gpa'])): print('gpa nan')
            if np.any(np.isinf(nparam['gpa'])): print('gpa inf')
            if np.any(np.isnan(nparam['gptau'])): print('gptau nan')
            if np.any(np.isinf(nparam['gptau'])): print('gptau inf')
            if np.any(np.isnan(self.data[0])): print('t nan')
            if np.any(np.isinf(self.data[0])): print('t inf')
            if np.any(np.isnan(self.data[2])): print('err nan')
            if np.any(np.isinf(self.data[2])): print('err inf')
            try: self.gp.compute(self.data[0], self.data[2]/detfac)
            except:
                print(nparam)
                print(np.any(detfac==0))
                raise
            return self.gp.log_likelihood(self.data[1]/detfac-self._model_function(nparam,self.data[0],named=True,detrend=False))
        detfac = get_detrend_factor(nparam, self.detrendvar) if self._calcdetrend else np.ones(self.Ndata)
        llhood = []
        for i in range(self.Ndata):
            if self._calcgp[i]:
                self.gp.kernel = kernel(nparam[i]['gpa'], nparam[i]['gptau'])
                self.gp.compute(self.data[i][0], self.data[i][2]/detfac[i])
                llhood.append(self.gp.log_likelihood(self.data[i][1]/detfac[i]-self._model_function(nparam[i],self.data[i][0],named=True,detrend=False)))
            else: llhood.append(None)
        return llhood

    def run_mcmc(self,*args,**kwargs):
        data = self.data if self.Ndata==1 else list(map(list,zip(*self.data)))
        param = kwargs.pop('param_init',self.param)
        bounds = kwargs.pop('param_bounds',self.param_bounds)
        priorpdf_type = kwargs.pop('priorpdftype','uniform')
        priorpdf_args = kwargs.pop('priorpdfargs',())
        self._calcgp = self._contains_gppar(self.get_named_param(param))
        if not np.any(self._calcgp): self._calcgp = False
        self._calcdetrend = self._contais_detrendvar(self.detrendvar)
        if not np.any(self._calcdetrend): self._calcdetrend = False
        llhood = self.log_likelihood_gp if self._calcgp else None
        self.mcmc = MCMC(np.array(data),param,bounds,fixpos=self.fixed,modelfunc=self._model_function,loglikelihood=llhood,
                         ignorex_loglikelihood=True,priorpdf_type=priorpdf_type,priorpdf_args=priorpdf_args)
        self.mcmc.mcmc_name = kwargs.pop('mcmc_name', '')
        self.mcmc.param_names = self.param_name_plus_group
        optimize = kwargs.pop('preoptimize',False)
        if optimize:
            self.mcmc.optimized_initpar()
            self._mcmc_parinit_optimized = True
        self.mcmc(*args, **kwargs)
        self.mcmc_accepted = self.mcmc.accepted
        self.mcmc_nwalker = self.mcmc.Nwalker
        self.mcmc_niterate = self.mcmc.Niterate

    @property
    def skeleton(self):
        keys = ['Ndata','index_param_indiv','index_bounds_indiv','index_param_common','index_bounds_common','group_indiv',
                'group_common','fixed','detrendvar','param_name','param_bounds','_varind', 'mcmc_accepted','mcmc_nwalker','mcmc_niterate']
        skeleton = dict(input_data=self.data, param_init=self.param)
        for key in keys:
            try: skeleton[key] = self.__getattribute__(key)
            except: skeleton[key] = None
        skeleton['mcmc_param_init_optimized'] = self.mcmc.param_init if self._mcmc_parinit_optimized else []
        return skeleton
            
    @property
    def param_name_plus_group(self):
        groupexp = np.full(len(self.param), '', dtype=object)
        for i in range(len(self.group_indiv)):
            if self.group_indiv[i]: groupexp[self.index_param_indiv[i]] = '-'+self.group_indiv[i]
        for i in range(len(self.group_common)):
            if self.group_common[i]: groupexp[self.index_param_common[i]] = self.group_common[i]
        return list(map(''.join, zip(self.param_name,groupexp)))

    def saveall(self, retrieval_skeleton='', params_mcmc='', mcmc_params_rawsample='', results_optimized='', **kwargs): # TODO:
        import pickle as pkl
        if retrieval_skeleton: pkl.dump(self.skeleton, open(retrieval_skeleton+'.pkl','wb'))
        if params_mcmc:
            if 'header' not in kwargs: kwargs['header'] = '\t'.join(self.param_name_plus_group)
            np.savetxt(params_mcmc,self.params_mcmc,**kwargs)

    def load_retrieval_skeleton(self, skeleton='',param_init='init'):
        import pickle as pkl
        if skeleton:
            if type(skeleton)==str and os.path.exists(skeleton+'.pkl'): skeleton = pkl.load(open(skeleton+'.pkl','rb'))
            self.data = skeleton.pop('input_data')
            parami = skeleton.pop('param_init')
            parammo = skeleton.pop('mcmc_param_init_optimized')
            paramo = skeleton.pop('param_optimized',[])
            if param_init=='init': self.param = parami
            if param_init=='mcmc_preoptimized': self.param = parammo
            if param_init=='mcmc_final': self.param = self.median_err_params_mcmc[:,0]
            if param_init=='optimized_final': self.param = paramo
            for key in skeleton:
                self.__setattr__(key, skeleton[key])

    def load_params_mcmc(self, source):
        if source:
            if type(source)==str and os.path.exists(source): source = np.loadtxt(source)
            self.params_mcmc = source

    def get_flatsamples(self, saveto='', **kwargs):
        self.params_mcmc = self.mcmc.get_flatsamples(**kwargs)

    def load_backend_mcmc(self, source, mcmc_name=''):
        if not hasattr(self,'mcmc'):
            self.mcmc = MCMC([],[])
            self.mcmc.mcmc_name = mcmc_name
            self.mcmc.param_names = self.param_name_plus_group
        self.mcmc.load_backend(source)
        self.mcmc.Niterate, self.mcmc.Nwalker, self.mcmc.Ndim = self.mcmc.get_samples().shape

    @staticmethod
    def chooseNflatten_from_sample(samples, burn=0, thin=1, accepted=[]):
        return samples[burn::thin,:,:].reshape(-1,samples.shape[2]) if len(accepted)!=0 else samples[burn::thin,accepted,:].reshape(-1,samples.shape[2])

    def get_best_fit(self, **kwargs):
        data = self.data if self.Ndata == 1 else list(map(list, zip(*self.data)))
        param = kwargs.pop('param_init', self.param)
        bounds = kwargs.pop('param_bounds', self.param_bounds)

        self.mcmc = MCMC(np.array(data), param, bounds, fixpos=self.fixed, modelfunc=self._model_function, loglikelihood=self.log_likelihood_gp, ignorex_loglikelihood=True)

    def get_transit_model(self,param,t,named=False,detrend=True,denoiseGP=True):
        if not named:
            if np.ndim(param): param = self.get_named_param(param)
            elif np.ndim(param)==2:
                param = list(param)
                for i in range(len(param)): param[i] = self.get_named_param(param[i])
        modelop = self._model_function(param, t, named=True, detrend=detrend)
        if not np.any(self._contains_gppar(param)) or not denoiseGP: return modelop
        if self.Ndata == 1:
            self.gp.kernel = kernel(param['gpa'], param['gptau'])
            self.gp.compute(self.data[0], self.data[2])
            return self.gp.sample_conditional(
                self.data[1] - modelop, t) + modelop
        for i in range(self.Ndata):
            if self._contains_gppar(param[i]):
                self.gp.kernel = kernel(param[i]['gpa'], param[i]['gptau'])
                self.gp.compute(self.data[i][0], self.data[i][2])
                modelop[i] += self.gp.sample_conditional(self.data[i][1]-modelop[i], t, size=100).mean(0)
        return modelop

    def get_adjusted_data(self, param, named=False, detrend=True, denoiseGP=True):
        if self.Ndata==0: return
        if not named:
            if np.ndim(param): param = self.get_named_param(param)
            elif np.ndim(param)==2:
                param = list(param)
                for i in range(len(param)): param[i] = self.get_named_param(param[i])
        if detrend and self._calcdetrend:
            detfac = get_detrend_factor(param, self.detrendvar)
        if self.Ndata==1:
            t, f, e = self.data
            if detrend and self._calcdetrend:
                f /= detfac
                e /= detfac
            if denoiseGP and self._calcgp:
                self.gp.kernel = kernel(param['gpa'], param['gptau'])
                self.gp.compute(t,e)
                f -= self.gp.sample_conditional(f-self._model_function(param,t,named=True,detrend=False), t, size=100).mean(0)
            return t,f,e
        t, f, e = [], [], []
        for i in range(self.Ndata):
            ti,fi,ei = self.data[i]
            if detrend and self._calcdetrend:
                fi /= detfac[i]
                ei /= detfac[i]
            if denoiseGP and self._calcgp:
                self.gp.kernel = kernel(param[i]['gpa'], param[i]['gptau'])
                self.gp.compute(ti, ei)
                fi -= self.gp.sample_conditional(fi-self._model_function(param[i],ti,named=True,detrend=False), t, size=100).mean(0)
            t.append(ti)
            f.append(fi)
            e.append(ei)
        return t,f,e




    @property
    def median_err_params_mcmc(self):
        return get_median_error_from_distribution(self.params_mcmc, sigma=1, method='percentile', saveas='')

    def save_median_err_params_mcmc(self, saveto='', parnames=[], display=True):
        parfinal = self.median_err_params_mcmc.T
        print(parfinal.shape)
        if not parnames: parnames = self.mcmc.param_names
        if os.path.exists(saveto): os.remove(saveto)
        for i, parname in enumerate(parnames):
            if display: print(parname + f': {parfinal[i, 0]} +{parfinal[i, 1]} -{parfinal[i, 2]}')
            if saveto: print(parname + f': {parfinal[i, 0]} +{parfinal[i, 1]} -{parfinal[i, 2]}', file=open(saveto, 'a'))

    def overplot_model_median_err_params_mcmc(self,multifig=False,axis='auto',figsize=(10,10)):
        import matplotlib.pyplot as plt
        params = self.median_err_params_mcmc.T
        t, f, e = self.get_adjusted_data(params[:, 0])
        if self.Ndata==1:
            fig, ax = plt.subplots(figsize=figsize)
            midfluxfit = self.get_transit_model(params[:, 0], t, detrend=False, denoiseGP=False)
            ax.errorbar(t, f, e, fmt='.')
            ax.plot(t, midfluxfit, c='r', lw=3, label='Model corr. to median of parameters')
            return fig, ax
        if not multifig:
            if axis=='auto': figure, axes = plt.subplots(self.Ndata,1,figsize=figsize)
            elif isinstance(axis, (tuple,list)) and len(axis)==2:
                figure, axes = plt.subplots(axis[0], axis[1], figsize=figsize)
            axes = np.ravel(axes)
        else:
            figure, axes = [], []
        for i in range(self.Ndata):
            if multifig:
                fig, ax = plt.subplots(figsize=figsize)
                figure.append(fig)
                axes.append(ax)
            else:
                ax = axes[i]
            # params[:, 1] = params[:, 0] + params[:, 1]
            # params[:, 2] = params[:, 0] - params[:, 2]
            midfluxfit = self.get_transit_model(params[i][:,0],t[i],detrend=False,denoiseGP=False)
            # extremefluxfit1 = self.get_transit_model(params[:,1],t,detrend=False,denoiseGP=False)
            # extremefluxfit2 = self.get_transit_model(params[:,2],t,detrend=False,denoiseGP=False)
            # params = np.transpose([mesh.ravel() for mesh in np.meshgrid(*params[:, :3])])
            # for par in params[np.random.choice(np.arange(params.shape[0]),20),:]:
            #     if not any([np.array_equal(par, params[:, col]) for col in range(params.shape[1])]):
            #         fluxfit = self.get_transit_model(par,t,detrend=False,denoiseGP=False)
            #         ax.plot(t, fluxfit, c='c', lw='1', )
            ax.errorbar(t[i], f[i], e[i], fmt='.')
            ax.plot(t[i], midfluxfit, c='r', lw=3, label='Model corr. to median of parameters')
            # ax.plot(t, extremefluxfit1, c='m', lw=3, label='Model corr. to median+1-$\sigma$ of parameters')
            # ax.plot(t, extremefluxfit2, c='g', lw=3, label='Model corr. to median-1-$\sigma$ of parameters')
        return figure,axes


def mcmc_moves():
    return MCMC.listofmoves



