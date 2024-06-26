def get_acc_predictor(model, inputs, targets):

    if model == 'rbf':
        from acc_predictor.rbf import RBF
        acc_predictor = RBF()
        acc_predictor.fit(inputs, targets)

    elif model == 'carts':
        from acc_predictor.carts import CART
        acc_predictor = CART(n_tree=5000)
        acc_predictor.fit(inputs, targets)

    elif model == 'gp':
        from acc_predictor.gp import GP
        acc_predictor = GP()
        acc_predictor.fit(inputs, targets)

    elif model == 'mlp':
        from acc_predictor.mlp import MLP
        acc_predictor = MLP(n_feature=inputs.shape[1])
        acc_predictor.fit(x=inputs, y=targets)

    elif model == 'as':
        from acc_predictor.adaptive_switching import AdaptiveSwitching
        acc_predictor = AdaptiveSwitching()
        acc_predictor.fit(inputs, targets)

    else:
        raise NotImplementedError

    return acc_predictor

