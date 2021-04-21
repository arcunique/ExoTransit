from george import kernels
import warnings

__all__ = ['get_median_error_from_distribution', 'NamedParam', 'kernel', 'legend_colortxt_nosym']


def kernel(a, tau, func=kernels.Matern32Kernel):
    return a ** 2 * func(tau)


def get_median_error_from_distribution(sample, sigma=1, method='percentile', saveas=''):
    import numpy as np
    if method == 'percentile':
        from scipy.stats import norm
        percentile = [100 - int(norm.cdf(sigma) * 100), 50, int(norm.cdf(sigma) * 100)]
        if np.ndim(sample) == 1: return (lambda v: (v[1], v[2] - v[1], v[1] - v[0]))(np.percentile(sample, percentile))
        result = np.array([[v[1], v[2] - v[1], v[1] - v[0]] for v in np.percentile(sample, percentile, axis=0).T]).T
        if saveas and type(saveas) == str:
            np.savetxt(saveas, result)
        elif saveas and callable(saveas):
            saveas(result)
        return result


class classproperty(object):

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class MCMCWarning(Warning): pass


class MCMCException(Exception): pass


def warn(message, category=MCMCWarning):
    warnings.showwarning(message, category, '', '')


class NamedParam(dict):

    def __add__(self, other):
        if isinstance(other, dict) and not isinstance(other, NamedParam): other = NamedParam(other)
        if isinstance(other, NamedParam):
            if not other: return NamedParam(super().copy())
            if sorted(self.keys()) != sorted(other.keys()): raise ValueError(
                'The two NmedParam instances should have same keys for any arithmatic operation.')
            return NamedParam({key: self[key] + other[key] for key in self})
        return NamedParam({key: self[key] + other for key in self})

    def __sub__(self, other):
        if isinstance(other, dict) and not isinstance(other, NamedParam): other = NamedParam(other)
        other = -other
        # if isinstance(other,NamedParam): other = {key:-other[key] for key in other}
        # else: other = -other
        return self.__add__(other)

    def __pos__(self):
        return self.copy()

    def __neg__(self):
        return NamedParam({key: -value for key, value in self.items()}) if self else self

    def __mul__(self, other):
        if isinstance(other, dict) and not isinstance(other, NamedParam): other = NamedParam(other)
        if isinstance(other, NamedParam):
            if sorted(self.keys()) != sorted(other.keys()): raise ValueError(
                'The two NmedParam instances should have same keys for any arithmatic operation.')
            return NamedParam({key: self[key] * other[key] for key in self})
        return NamedParam({key: self[key] * other for key in self})

    def __truediv__(self, other):
        if isinstance(other, dict) and not isinstance(other, NamedParam): other = NamedParam(other)
        if isinstance(other, NamedParam):
            other = {key: 1 / other[key] for key in other}
        else:
            other = 1 / other
        return self.__mul__(other)

    def dotinto(self, other):
        import numpy as np
        if isinstance(other, dict): return self.__mul__(other)
        # if isinstance(other, list)

    def concatenate(self, other):
        if isinstance(other, dict) and not isinstance(other, NamedParam): other = NamedParam(other)
        if isinstance(other, NamedParam):
            if sorted(self.keys()) != sorted(other.keys()): raise ValueError(
                'The two NamedParam instances should have same keys for concatenation.')
            return NamedParam({key: (self[key], other[key]) for key in self})
        return NamedParam({key: (self[key], other) for key in self})

    def keys(self):
        return list(super().keys())

    def remove(self, *args, inplace=False):
        if inplace:
            for arg in args: self.pop(arg, None)
        selfcopy = self.copy()
        for arg in args: self.pop(arg, None)
        return selfcopy

    def copy(self):
        return NamedParam(self + {})

    @property
    def T(self):
        keys = list(self.keys())
        if any([len(self[keys[i]]) != len(self[keys[0]]) for i in range(1, len(keys))]): raise ValueError(
            'Length of all the keyed values of the NamedParam instance should be same to transpose it.')
        return [NamedParam({key: self[key][i] for key in keys}) for i in range(len(self[keys[0]]))]

def legend_colortxt_nosym(ax=None, *args, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    if ax is None: ax = plt.gca()
    ha = kwargs.pop('hor_align', None)
    ebsym = kwargs.pop('ebsym', False)
    leg = ax.legend(*args, **kwargs, )
    for handle, txt in zip(leg.legendHandles, leg.get_texts()):
        if type(handle) != LineCollection or not ebsym: handle.set_visible(False)
        try:
            txt.set_color(handle.get_c())
        except AttributeError:
            txt.set_color(handle.get_color()[0])
        if ha is not None: txt.set_ha(ha)
    return leg
