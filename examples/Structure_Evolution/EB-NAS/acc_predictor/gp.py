from pydacefit.regr import regr_constant
from pydacefit.dace import DACE, regr_linear, regr_quadratic
from pydacefit.corr import corr_gauss, corr_cubic, corr_exp, corr_expg, corr_spline, corr_spherical


class GP:
    """ Gaussian Process (Kriging) """
    def __init__(self, regr='linear', corr='gauss'):
        self.regr = regr
        self.corr = corr
        self.name = 'gp'
        self.model = None

    def fit(self, train_data, train_label):
        if self.regr == 'linear':
            regr = regr_linear
        elif self.regr == 'constant':
            regr = regr_constant
        elif self.regr == 'quadratic':
            regr = regr_quadratic
        else:
            raise NotImplementedError("unknown GP regression")

        if self.corr == 'gauss':
            corr = corr_gauss
        elif self.corr == 'cubic':
            corr = corr_cubic
        elif self.corr == 'exp':
            corr = corr_exp
        elif self.corr == 'expg':
            corr = corr_expg
        elif self.corr == 'spline':
            corr = corr_spline
        elif self.corr == 'spherical':
            corr = corr_spherical
        else:
            raise NotImplementedError("unknown GP correlation")

        self.model = DACE(
            regr=regr, corr=corr, theta=1.0, thetaL=0.00001, thetaU=100)
        self.model.fit(train_data, train_label)

    def predict(self, test_data):
        assert self.model is not None, "GP does not exist, call fit to obtain GP first"
        return self.model.predict(test_data)
