import logging

import numpy as np
from mgwr.gwr import GWR as Mod_GWR
from mgwr.sel_bw import Sel_BW
from pykrige.ok import OrdinaryKriging

from .base import BaseAlgorithm


class Kriging(BaseAlgorithm):
    def param_names(self):
        return ["num_lags", "model_type"]

    def train(self, graph, mask):
        self.model = OrdinaryKriging(
            graph.get_X(mask)[:, 0],
            graph.get_X(mask)[:, 1],
            graph.get_y(mask),
            nlags=self.params["num_lags"],
            variogram_model=self.params["model_type"],
            verbose=False,
        )

    def predict(self, graph, mask):
        z, var = self.model.execute(
            "points", graph.get_X(mask)[:, 0], graph.get_X(mask)[:, 1]
        )
        return z


class GWR(BaseAlgorithm):
    def param_names(self):
        return ["num_intercepts", "bandwidth", "model_type"]

    def train(self, graph, mask):
        self.train_mask = mask
        self.coords_train = graph.get_X(mask)
        self.y_train = graph.get_y(mask).reshape((-1, 1))
        self.X_train = np.ones((len(mask), self.params["num_intercepts"]))
        self.gwr_selector = Sel_BW(self.coords_train, self.y_train, self.X_train)
        if not self.params["bandwidth"]:
            try:
                self.params["bandwidth"] = self.gwr_selector.search()
            except ValueError:
                logging.warning("GWR failed BW search")
                self.params["bandwidth"] = 1.0

    def _fresh_gwr(self):
        model = Mod_GWR(
            self.coords_train,
            self.y_train,
            self.X_train,
            self.params["bandwidth"],
            fixed=False,
            kernel=self.params["model_type"],
            constant=False,
        )
        gwr_results = model.fit()
        scale = gwr_results.scale
        residuals = gwr_results.resid_response
        return model, scale, residuals

    def predict(self, graph, mask):
        pred_X = np.ones((len(mask), self.params["num_intercepts"]))
        pred_c = graph.get_X(mask)
        all_preds = []
        try:
            if len(mask) <= len(self.train_mask):
                model, scale, residuals = self._fresh_gwr()
                pred_results = model.predict(pred_c, pred_X)
                return np.array([p[0] for p in pred_results.predictions])
            logging.warning(
                "Requested predictions are longer than train, looping manually"
            )
            for co, x in zip(pred_c, pred_X):
                model, scale, residuals = self._fresh_gwr()
                pred_results = model.predict(co.reshape((1, -1)), x.reshape((1, -1)))
                all_preds.append(pred_results.predictions[0][0])
            return np.array(all_preds)
        except Exception:
            logging.warning("GWR failed")
        return np.full((len(pred_c),), np.mean(self.y_train))
