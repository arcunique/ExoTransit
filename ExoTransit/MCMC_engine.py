import numpy as np
import emcee
from scipy.optimize import minimize
import warnings
import matplotlib.pyplot as plt
from ExoTransit.shared.MCMC_with_checkpoint import *
from ExoTransit.utils import classproperty, warn, MCMCWarning, MCMCException


try:
    from multiprocessing import cpu_count, Pool
    mpimport = True
except:
    mpimport = False

__all__ = ['MCMC']


class MCMC(object):

    def __init__(self, data, param, bounds=[], fixpos=[], freepos=[], modelfunc=None, loglikelihood=None,
                 priorpdf_args=[], priorpdf_type='uniform', **kwargs):
        """

        :param data:
        :param modelfunc:
        :param param:
        :param bounds:
        :param fixpos:
        :param freepos:
        :param kwargs:
        """
        if len(data) != 0:
            if type(data) == str: data = np.loadtxt(data, unpack=True)
            if data.shape[0] == 3:
                self.x, self.y, self.yerr = data
            else:
                self.y, self.yerr = data
                self.x = None
        self.param_names = kwargs.pop('param_names', [])
        if len(param) != 0:
            if fixpos:
                self.param_init = [param[i] for i in range(len(param)) if i not in fixpos]
                freepos = [i for i in range(len(param)) if i not in fixpos]
            elif freepos:
                self.param_init = [param[fr] for fr in freepos]
            else:
                self.param_init = param.copy()
                freepos = [i for i in range(len(param))]
            mprange = kwargs.pop('mprange', [])
            if len(bounds) != 0:
                self.param_bounds = bounds
            elif len(mprange) != 0:
                self.param_bounds = [[] for _ in mprange]
                for i in range(len(mprange)):
                    if isinstance(mprange[i], (int, float)):
                        self.param_bounds[i] = self.param_init[i] - mprange[i], self.param_init[i] + mprange[i]
                    else:
                        self.param_bounds[i] = self.param_init[i] - mprange[i][0], self.param_init[i] + mprange[i][0]
            if self.param_names and len(self.param_names) > len(self.param_init): self.param_names = [
                self.param_names[fr] for fr in freepos]
            if priorpdf_type: self.initiate_log_prior_function(*priorpdf_args, pdftype=priorpdf_type)
        self.mcmc_name = kwargs.pop('mcmc_name', '')
        self._modelfunc = None
        self._loglikelihood = None
        self._model = []
        ignorex_model = kwargs.pop('ignorex_modelfunc', False)
        ignorex_like = kwargs.pop('ignorex_loglikelihood', False)
        if modelfunc:
            if self.x is not None and len(self.x) != 0 and not ignorex_model:
                self._modelfunc = lambda par: modelfunc(
                    [param[i] if i not in freepos else par[freepos.index(i)] for i in range(len(param))], self.x,
                    **kwargs)
            else:
                self._modelfunc = lambda par: modelfunc(
                    [param[i] if i not in freepos else par[freepos.index(i)] for i in range(len(param))], **kwargs)
        if loglikelihood:
            if (self.x is not None or len(self.x) != 0) and not ignorex_like:
                self._loglikelihood = lambda par: loglikelihood(
                    [param[i] if i not in freepos else par[freepos.index(i)] for i in range(len(param))], self.x,
                    **kwargs)
            else:
                self._loglikelihood = lambda par: loglikelihood(
                    [param[i] if i not in freepos else par[freepos.index(i)] for i in range(len(param))], **kwargs)
        self.sampler = None
        self.autocorr_nanlen = 0

    def eval_model(self, param):
        if self._modelfunc:
            if np.ndim(param) == 1: return self._modelfunc(param)
            return np.array([self.eval_model(par) for par in param])

    def initiate_log_prior_function(self, *args, pdftype='uniform'):
        from scipy.stats import uniform, norm
        if type(pdftype) == str: pdftype = [pdftype] * len(self.param_bounds)
        self._priorpdffuncs = []
        j = 0
        for i, pt in enumerate(pdftype):
            if pt.lower() in ('u', 'uniform'):
                if len(args) < len(pdftype):
                    self._priorpdffuncs.append(uniform(*self.param_bounds[i]).pdf)
                else:
                    self._priorpdffuncs.append(uniform(*args[i]).pdf)
            elif pt.lower() in ('g', 'n', 'gaussian', 'normal', 'norm'):
                if len(args) < len(pdftype):
                    self._priorpdffuncs.append(norm(*args[j]).pdf)
                    j += 1
                else:
                    self._priorpdffuncs.append(norm(*args[i]).pdf)

    def log_prior_uniform(self, param):  # depreceted
        if np.all([(param[i] >= self.param_bounds[i][0]) & (param[i] <= self.param_bounds[i][1]) for i in
                   range(len(param))]):
            return 0
        return -np.inf

    def log_prior(self, param):
        with np.errstate(divide='ignore'):
            return np.log(np.prod([pf(par) for pf, par in zip(self._priorpdffuncs, param)]))

    def log_likelihood(self, param):
        llhood = []
        if self._loglikelihood:
            llhood = self._loglikelihood(param)
            if isinstance(llhood, (int, float)): return llhood
        if self._modelfunc:
            model = self._modelfunc(param)
            try:
                exterm = np.log(2 * np.pi * self.yerr ** 2)
            except AttributeError:
                exterm = np.array([np.log(yerr) for yerr in self.yerr])
            llhoodm = -0.5 * ((self.y - model) ** 2 / self.yerr ** 2 + exterm)
            if np.ndim(llhoodm) == 1 and not hasattr(llhoodm[0], '__len__'):
                self._model = model.copy()
                return np.sum(llhoodm)
            if llhood:
                for i in range(len(llhood)):
                    if not llhood[i]: llhood[i] = np.sum(llhoodm[i])
            else:
                llhood = [sum(llh) for llh in llhoodm]
        return np.sum(llhood)

    def log_probability(self, param):
        logprior = self.log_prior_uniform(param)
        if np.isinf(logprior): return logprior
        return self.log_likelihood(param) + logprior

    def chisquare(self, param):
        return -self.log_likelihood(param)

    def optimized_initpar(self, saveto=''):
        soln = minimize(self.chisquare, self.param_init, bounds=self.param_bounds)
        self.param_init = soln.x
        if saveto and type(saveto) == str:
            np.savetxt(saveto, soln.x)
        elif saveto and callable(saveto):
            saveto(soln.x)

    @classproperty
    def listofmoves(self):
        allmoves = ["GaussianMove", "StretchMove", "WalkMove", "KDEMove", "DEMove", "DESnookerMove"]
        lmv = {move.lower(): eval('emcee.moves.' + move) for move in allmoves}
        lmv.update(
            useinfo='Use the keywords (as a string) or the values (as a function object) for "moves" entry. Note that, '
                    'for entry as a string it is case-insensitive. For entry as a function object use either of '
                    'the structures and remember to import "emcee" beforehand.')
        lmv.update(
            moveinfo='GaussianMove is a subclass of "MHMove" and all other moves are subclasses of "RedBlueMove".')
        return lmv

    @property
    def accepted(self):
        if self.sampler and self.sampler.backend and hasattr(self.sampler.backend, 'accepted'):
            return self.sampler.backend.accepted.astype(bool)
        if self.sampler and hasattr(self.sampler, 'accepted'):
            return self.sampler.accepted.astype(bool)
        return None

    @property
    def backend(self):
        return self.sampler.backend if self.sampler and self.sampler.backend else None

    def __call__(self, Nwalker, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        self.Nwalker = Nwalker
        self.Niterate = kwargs.pop('Niterate', None)
        if not self.Niterate: self.Niterate = 100000
        checkconvergence = kwargs.pop('checkconvergence', False)
        checkstep = kwargs.pop('checkstep', 100)
        eps = kwargs.pop('epsilon', 0.01)
        if checkconvergence:
            if self.Niterate and self.Niterate <= 20 * checkstep:
                warn('Not enough number of iteration compared to checkstep. It is suggested to increase the value of '
                     'Niterate', MCMCWarning)
        showprogress = kwargs.pop('showprogress', True)
        windowtitles = kwargs.pop('windowtitles', [])
        monitor = kwargs.pop('monitor', None)
        monitor_parameters = kwargs.pop('monitor_parameters', {})
        figsize = kwargs.pop('figsize', (6, 7))
        monitorstep = kwargs.pop('monitorstep', 10)
        if monitor and monitor == 'all': monitor = ['autocorrtime', 'corner', 'walkerprogress']
        if monitor:
            if len(figsize) == 2 and type(figsize[0]) == int: figsize = [figsize for _ in monitor]
        walker_sigma = kwargs.pop('walker_sigma', 1e-4)
        Npool = kwargs.pop('Npool', 1)
        if Npool > 1 and not mpimport:
            warn('Continuing without parallelization (Npool set to 1) as the package multiprocessing not found.')
            Npool = 1
        pool = None if Npool == 1 else Pool(Npool)
        moves = kwargs.pop('moves', None)
        if moves:
            if type(moves) == str:
                if moves.lower() not in self.listofmoves:
                    raise MCMCException(
                        f'Entered move {moves} not found. Type MCMC.listofmoves to check the available moves.')
                else:
                    moves = self.listofmoves[moves.lower()]()
            elif isinstance(moves, (list, tuple)):
                moves = list(moves)
                for i, move in enumerate(moves):
                    if type(move) == str:
                        if move.lower() not in self.listofmoves:
                            raise MCMCException(
                                f'Entered move {move} not found. Type <instance of mcmc>.listofmoves to check the available moves.')
                        else:
                            moves[i] = self.listofmoves[move.lower()]()
                    elif isinstance(move, (list, tuple)):
                        moves[i] = list(moves[i])
                        if moves[i][0].lower() not in self.listofmoves:
                            raise MCMCException(
                                f'Entered move {moves[i][0]} not found. Type <instance of mcmc>.listofmoves to check the available moves.')
                        else:
                            moves[i][0] = self.listofmoves[moves[i][0].lower()]()
        self.Ndim = len(self.param_init)
        ldbk = kwargs.pop('loadbackendfrom', None)
        svbk = kwargs.pop('savebackendto', None)
        if ldbk:
            backend = emcee.backends.HDFBackend(ldbk + '.h5')
        elif svbk:
            backend = emcee.backends.HDFBackend(svbk + '.h5')
            backend.reset(self.Nwalker, self.Ndim)
        else:
            backend = None
        self.sampler = emcee.EnsembleSampler(self.Nwalker, self.Ndim, self.log_probability, pool=pool, moves=moves,
                                             backend=backend)
        walkers = self.param_init + np.array(walker_sigma) * np.random.randn(self.Nwalker, self.Ndim)
        if not monitor and not checkconvergence:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.sampler.run_mcmc(walkers, self.Niterate, progress=showprogress)
        else:
            monitor_func = {'autocorrtime': self.show_autocorrelation_time, 'corner': self.show_corner,
                            'walkerprogress': self.show_walker_progress}
            run_mcmc_with_checkpoint(self.sampler, walkers, self.Ndim, self.Niterate, showprogress, monitor,
                                     monitorstep, monitor_func, monitor_parameters, checkconvergence, checkstep, eps,
                                     self.param_names, self.mcmc_name, windowtitles, figsize)

    def load_backend(self, source=''):
        import os
        if source and type(source) == str and os.path.exists(source + '.h5'):
            self.sampler = emcee.backends.HDFBackend(source + '.h5')
            self.sampler.backend = None
            # return emcee.backends.HDFBackend(source+'.h5')

    def get_samples(self, burn=0, thin=1, accepted_only=False, flat=False):
        if burn == 'auto' or thin == 'auto': tau = self.sampler.get_autocorr_time(tol=0)
        if burn == 'auto': burn = int(2 * np.max(tau))
        if thin == 'auto': thin = int(0.5 * np.min(tau))
        if self.sampler.iteration != self.autocorr_nanlen: burn = max([burn, self.autocorr_nanlen])
        if not accepted_only: return self.sampler.get_chain(discard=burn, flat=flat, thin=thin)
        sample = self.sampler.get_chain(discard=burn, thin=thin)[:, self.accepted, :]
        return sample.reshape(-1, self.Ndim) if flat else sample

    def get_flatsamples(self, **kwargs):
        return self.get_samples(flat=True, **kwargs)

    def show_corner(self, figure=None, labels=[], figsize=None, burn=0, thin=1, accepted_only=False, **kwargs):
        import corner, sys, os
        if figure:
            figure.clf()
            figure.subplots(self.Ndim, self.Ndim)
        elif figsize:
            figure, axes = plt.subplots(self.Ndim, self.Ndim, figsize=figsize)
        if not labels: labels = self.param_names
        sys.stderr = open(os.devnull, "w")
        fig = corner.corner(self.get_flatsamples(burn=burn, thin=thin, accepted_only=accepted_only), labels=labels,
                            fig=figure, **kwargs)
        sys.stderr = sys.__stderr__
        fig.suptitle(f'Iteration: {self.sampler.iteration}')
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        # print('showing corner')
        show_accto_backend(fig)
        # plt.pause(0.2)
        return fig

    def show_walker_progress(self, figure=None, labels=[], burn=0, thin=1, accepted_only=False, figsize=None,
                             subplot_kw={}, plot_kw={}):
        if figure:
            # print('walker progress', figure.number)
            figure.clf()
            axes = figure.subplots(self.Ndim, sharex='all', **subplot_kw)
        else:
            figure, axes = plt.subplots(self.Ndim, figsize=figsize, sharex='all', **subplot_kw)
        if not labels: labels = self.param_names
        # print('lab',labels)
        new_plot_kw = [{} for _ in range(self.Ndim)]
        for key in plot_kw:
            if isinstance(plot_kw[key], (list, tuple)) and len(plot_kw[key]) == self.Ndim:
                val = plot_kw.pop(key)
                for i in range(self.Ndim): new_plot_kw[i].update({key: val[i]})
        for i in range(self.Ndim):
            new_plot_kw[i].update(plot_kw.copy())
            ax = axes[i]
            # print(self.get_samples(burn=burn, thin=thin, accepted_only=accepted_only).shape)
            ax.plot(self.get_samples(burn=burn, thin=thin, accepted_only=accepted_only)[:, :, i], **new_plot_kw[i])
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        figure.suptitle(f'Iteration: {self.sampler.iteration}')
        figure.subplots_adjust(hspace=0)
        axes[-1].set_xlabel("Step number")
        # figure.canvas.draw()
        # figure.canvas.flush_events()
        show_accto_backend(figure)
        # plt.pause(0.2)
        return figure, axes

    def show_autocorrelation_time(self, figure=None, labels=[], burn=0, thin=1, accepted_only=False, figsize=None,
                                  **kwargs):
        from emcee.autocorr import integrated_time
        if figure:
            # print('autocorr', figure.number)
            figure.clf()
        else:
            figure = plt.figure(figsize=figsize)
        ax = figure.subplots()
        samples = self.get_samples(burn=burn, thin=thin, accepted_only=accepted_only)
        rep, last = len(samples) // 50, len(samples) % 50
        if rep == 0:
            warn('Too low number of iterations to calculate autocorrelation time (min 50 required).')
            return figure
        tau = np.empty((rep + 1, self.Ndim)) if last else np.empty((rep, self.Ndim))
        Niterate = np.arange(1, rep + 1) * 50
        # color = kwargs.pop('color',['r','g','b'])
        # if type(color)==str or len(color)==1: color = list(np.atleast_1d(color))*self.Ndim
        newkwargs = [{} for _ in range(self.Ndim)]
        for key in kwargs:
            if isinstance(kwargs[key], (list, tuple)) and len(kwargs[key]) == self.Ndim:
                val = kwargs.pop(key)
                for i in range(self.Ndim): newkwargs[i].update({key: val[i]})
        for i in range(rep): tau[i] = integrated_time(samples[:50 * (i + 1)], tol=0)
        if last:
            tau[-1] = integrated_time(samples, tol=0)
            Niterate = np.append(Niterate, len(samples))
        # print(tau)
        for i in range(self.Ndim):
            newkwargs[i].update(kwargs.copy())
            ax.plot(Niterate, tau[:, i], **newkwargs[i])
            # print(tau[:,i])
        if np.any(np.isnan(tau[-1])): self.autocorr_nanlen = len(samples)
        ax.set_xlabel('Step number')
        ax.set_ylabel('Autocorrelation time ($\\tau$)')
        ax.xaxis.set_label_coords(0.5, -0.1)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        figure.suptitle(f'Iteration: {self.sampler.iteration}')
        # figure.canvas.draw()
        # figure.canvas.flush_events()
        show_accto_backend(figure)
        # plt.pause(0.2)
        return figure
