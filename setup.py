from setuptools import setup
import ExoTransit
setup(name='ExoTransit',
      version=ExoTransit.__version__,
      description='Python package for modeling transit light curves of exoplanets using Markov chain Monte Carlo '
                  'technique on the basis of the transit model of Eric & Agol (2002). It also allows simultaneous '
                  'detrending of light curves and modeling of correlated noise with Gaussian process regression.',
      long_description=open('README.md').read(),
      classifiers=['Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 3.7'],
      url='https://github.com/arcunique/Cplotter',
      author='Aritra Chakrabarty',
      author_email='aritra@iiap.res.in',
      install_requires=['numpy', 'matplotlib', 'scipy', 'emcee', 'george'],
      zip_safe=False)
