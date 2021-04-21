import numpy as np
import emcee
from run2learn import *
# from ExoTransit import environ
from ExoTransit import environ
# environ['mpl_backend'] = 'nbAgg'
# from ExoTransit.transit import *
from ExoTransit.utils import *
from ExoTransit.retrieve import model_transit_lightcurve
import matplotlib.pyplot as plt

import matplotlib

# matplotlib.rcParams['backend'] = 'Qt5Agg'
import sys, os
import corner

mlc = model_transit_lightcurve()
mlc.per = 1.5 * 1440
file = 'testdata/simdata'
t, flux, err = np.loadtxt(file, unpack=True, usecols=(0, 1, 2))
tmod, fmod = np.loadtxt(file, unpack=True, usecols=(0, 3))
# plt.errorbar(t, flux, err, fmt='.')
# plt.show()

parlabelnames = ['tcen', 'b', 'Rs/a', 'Rp/Rs', 'fout', 'C2', 'C4']
parnames = ['tcen', 'b', 'rsa', 'rprs', 'fout', 'c2', 'c4']
par0 = [190, 0.1, 0.2, 0.2, 0.98, 0.4, 0.3]
bounds = [(170, 220), (0, 0.5), (0.1, 0.3), (0.1, 0.3), (0.96, 1.06), (0.3, 0.5), (0.25, 0.35)]
par = NamedParam(zip(parnames, par0))
bounds = NamedParam(zip(parnames, bounds))
# par.update(gpa=0.005, gptau=1)
# bounds.update(gpa=(0.001, 0.1), gptau=(1, 6))

mlc.add_data(file, par, bounds, usecols=(0, 1, 2))
# mlc.add_indiv_param(NamedParam(gpa=0.01,gptau=1),NamedParam(gpa=(0.001,0.1),gptau=(0,5)))

mlc.run_mcmc(mcmc_name='sim3x_gp', Nwalker=100, Niterate=20, monitor='all', savebackendto=file+'_mcmcbackend')
plt.ion()
fig, _ = mlc.mcmc.show_walker_progress(labels=parlabelnames + ['GP_a', 'GP_tau'])
# fig.canvas.draw()
plt.pause(3)
plt.show(block=False)
# fig.canvas.flush_events()

while True:
    inp = input('burn: ')
    if inp.isdigit():
        burn = int(inp)
        break
    if inp in ('none', 'no', 'cancel'):
        burn = None
        break
plt.close()
plt.ioff()
mlc.get_flatsamples(burn=burn)
mlc.saveall(retrieval_skeleton=file+'_skel', params_mcmc=file+'_allmcmcsamples')
# parammc = mlc.median_err_params_mcmc.T
mlc.save_median_err_params_mcmc(saveto=file + '_params_mcmc')
mlc.mcmc.show_corner(burn=burn, labels=parlabelnames + ['GP_a', 'GP_tau'])
_, ax = mlc.overplot_model_median_err_params_mcmc()
# t,f,e = mlc.get_adjusted_data(parammc[:,0])
ax.plot(tmod, fmod, 'k')
# ax[0].legend()
plt.show()