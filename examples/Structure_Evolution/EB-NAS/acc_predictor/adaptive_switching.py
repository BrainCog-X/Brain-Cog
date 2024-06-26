import utils
import numpy as np
from acc_predictor.factory import get_acc_predictor


class AdaptiveSwitching:
    """ ensemble surrogate model """
    """ try all available models, pick one based on 10-fold crx vld """
    def __init__(self, n_fold=10):
        # self.model_pool = ['rbf', 'gp', 'mlp', 'carts']
        self.model_pool = ['rbf', 'gp', 'carts']
        self.n_fold = n_fold
        self.name = 'adaptive switching'
        self.model = None
        # self.predictor_pool = []

    def fit(self, train_data, train_target):
        self._n_fold_validation(train_data, train_target, n=self.n_fold)
        # for p in self.predictor_pool:
        #     p.fit(train_data,train_target)

    def _n_fold_validation(self, train_data, train_target, n=10):

        n_samples = len(train_data)
        perm = np.random.permutation(n_samples)


        kendall_tau = np.full((n, len(self.model_pool)), np.nan)

        all_predict_result=[]

        for i, tst_split in enumerate(np.array_split(perm, n)):
            trn_split = np.setdiff1d(perm, tst_split, assume_unique=True)
            rl=[]
            # loop over all considered surrogate model in pool
            for j, model in enumerate(self.model_pool):
                acc_predictor = get_acc_predictor(model, train_data[trn_split], train_target[trn_split])                
                result = acc_predictor.predict(train_data[tst_split])
                rl.append(result)
                
                rmse, rho, tau = utils.get_correlation(result, train_target[tst_split])

                kendall_tau[i, j] = tau

            all_predict_result.append(rl)
            
        winner = int(np.argmax(np.mean(kendall_tau, axis=0) - np.std(kendall_tau, axis=0)))
        print("winner model = {}, tau = {}".format(self.model_pool[winner],
                                                   np.mean(kendall_tau, axis=0)[winner]))
        self.winner = self.model_pool[winner]
        # re-fit the winner model with entire data

        # acc_predictor = get_acc_predictor(self.model_pool[winner], train_data, train_target)
        # self.model = acc_predictor

    def predict(self, test_data):
        


        return self.model.predict(test_data)
