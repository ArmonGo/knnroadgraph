import logging

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm


class BaseAlgorithm:
    @classmethod
    def gridsearch(cls, graph, train_mask, val_mask, hypergrid, eval_func=None):
        if eval_func is None:
            eval_func = mean_absolute_error
        best_error = None
        best_model = None
        paramgrid = ParameterGrid(hypergrid)
        for combination in tqdm(paramgrid):
            model = cls(**combination)
            model.train(graph, train_mask)
            y_pred = model.predict(graph, val_mask)
            error = eval_func(graph.get_y(val_mask), y_pred)
            if best_error is None or error < best_error:
                logging.debug(
                    f"GridSearch improved error to {error} with {combination}"
                )
                best_error = error
                best_model = model
        return best_model, best_error

    def __init__(self, **params):
        for param in params:
            assert param in self.param_names()
        for param in self.param_names():
            assert param in params
        self.params = params

    def param_names(self):
        return []

    def train(self, graph, mask):
        raise NotImplementedError()

    def predict(self, graph, mask):
        raise NotImplementedError()
