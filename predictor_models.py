from lifelines import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# How to use the code
#
# setup the pandas dataframe that will contain the data.
# pandas dataframe, with columns: 'ncycles' (maximum between the number of cycles at which death occured,
#                                            or number of cycles the battery was ran at) , 'censoring' (0 if no death, 1 if death recorded)
#                                            any number of  columns for other predictors
#                                            and
#                              with rows: one per battery
#
# Then run the following lines:
# import predictor_models as pm
# predictor = pm.predictor_models(yourdf)
# results = predictor.extract_models_mae()
# print(results)


class predictor_models:
    """
    params
    :indata pandas dataframe, with columns: 'ncycles' (maximum between the number of cycles at which death occured,
                                            or number of cycles the battery was ran at) , 'censoring' (0 if no death, 1 if death recorded)
                                            any number of  columns for other predictors
                                            and
                              with rows: one per battery
    """

    def __init__(self, indata):
        self.indata = indata
        self.train_data, self.valid_data = train_test_split(self.indata, test_size=0.9)
        self.create_folds()

        self.xtensor = torch.tensor(
            indata.drop(["ncycles", "censoring"], axis=1).values, dtype=torch.float32
        )
        self.ytensor = torch.tensor(indata["ncycles"].values, dtype=torch.float32)
        self.ctensor = torch.tensor(indata["censoring"].values, dtype=torch.float32)

        self.modeldict = {
            "weibul": self.fit_weibull,
            "gengamma": self.fit_gengamma,
            "lognorm": self.fit_lognorm,
            "neuralnet1layer": self.fit_neuralnet1,
            "neuralnet2layer": self.fit_neuralnet2,
            "neuralnet2layer_lnorm_lrt_loss": self.fit_neuralnet2_lnorm_lrt_loss
        }

        self.lossdict = {
            "mse": self.mse,
            "lnorm_lrt": self.lnorm_lrt
        }

    def mse(self, modeled=0, ytrue=0, censoring=0):
        ypred = modeled.squeeze()
        uncensored = censoring == 1
        censored = censoring == 0

        uncensored_loss = (
            torch.mean((ypred[uncensored] - ytrue[uncensored]) ** 2)
            if uncensored.sum() > 0
            else torch.tensor(0.0)
        )

        overshoot_mask = censored & (ypred > ytrue)
        censored_loss = (
            torch.mean((ypred[overshoot_mask] - ytrue[overshoot_mask]) ** 2)
            if overshoot_mask.sum() > 0
            else torch.tensor(0.0)
        )

        return uncensored_loss + censored_loss

    def lnorm_lrt(self, modeled={}, ytrue = None, censoring=0, eps = 1e-8):
        mu = modeled["mu"]
        sigma = modeled["sigma"]

        log_t = torch.log(ytrue.clamp_min(eps))
        z = (log_t + mu) / sigma

        # log density of T: Normal log-pdf of log t, minus log t (Jacobian)
        log_f = -0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) - torch.log(sigma)- log_t

        # log survival: log(1 - Phi(z)) = log Phi(-z), stable in the tail
        log_S = torch.special.log_ndtr(-z)

        ll = censoring * log_f + (1 - censoring) * log_S
        return -ll.mean()

    def create_folds(self, nfolds=20):
        rng = np.random.default_rng(42)
        folds = np.array_split(rng.permutation(len(self.indata)), nfolds)

        self.folds = [
            (folds[i], np.concatenate([folds[j] for j in range(nfolds) if j != i]))
            for i in range(nfolds)
        ]

    def fit_weibull(self):
        mae = 0
        for ff in self.folds:
            ttrain = self.indata.iloc[ff[1]]
            tval = self.indata.iloc[ff[0]]
            ctval = tval[tval["censoring"] == 1]
            if ctval.empty:
                continue

            model = WeibullAFTFitter().fit(
                ttrain,
                duration_col="ncycles",
                event_col="censoring",
            )
            y_pred = model.predict_percentile(tval[tval["censoring"] == 1])
            y_true = ctval["ncycles"]

            mae += np.sum(abs(y_pred - y_true))

        self.weibul_model = WeibullAFTFitter().fit(
            self.indata,
            duration_col="ncycles",
            event_col="censoring",
        )

        return mae / np.sum(self.indata["censoring"])

    def fit_gengamma(self):
        mae = 0
        for ff in self.folds:
            ttrain = self.indata.iloc[ff[1]]
            tval = self.indata.iloc[ff[0]]
            ctval = tval[tval["censoring"] == 1]
            if ctval.empty:
                continue

            fitter = GeneralizedGammaRegressionFitter()
            fitter._scipy_fit_method = "SLSQP"
            model = fitter.fit(
                ttrain,
                duration_col="ncycles",
                event_col="censoring",
            )
            y_pred = model.predict_percentile(tval[tval["censoring"] == 1])
            y_true = ctval["ncycles"]

            mae += np.sum(abs(y_pred - y_true))

        final_fitter = GeneralizedGammaRegressionFitter()
        final_fitter._scipy_fit_method = "SLSQP"
        self.gengamma_model = final_fitter.fit(
            self.indata,
            duration_col="ncycles",
            event_col="censoring",
        )

        return mae / np.sum(self.indata["censoring"])

    def fit_lognorm(self):
        mae = 0
        for ff in self.folds:
            ttrain = self.indata.iloc[ff[1]]
            tval = self.indata.iloc[ff[0]]
            ctval = tval[tval["censoring"] == 1]
            if ctval.empty:
                continue

            model = LogNormalAFTFitter().fit(
                ttrain,
                duration_col="ncycles",
                event_col="censoring",
            )
            y_pred = model.predict_percentile(tval[tval["censoring"] == 1])
            y_true = ctval["ncycles"]

            mae += np.sum(abs(y_pred - y_true))

        self.lognorm_model = LogNormalAFTFitter().fit(
            self.indata,
            duration_col="ncycles",
            event_col="censoring",
        )

        return mae / np.sum(self.indata["censoring"])

    def train_net(self, model, xx, yy, cc, loss_fx="mse"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lr = 0.001
        epochs = 10

        xx = xx.to(device)
        yy = yy.to(device)
        cc = cc.to(device)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.02)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            modeled = model(xx)
            loss = self.lossdict[loss_fx](modeled = modeled, ytrue = yy, censoring  = cc)
            loss.backward()
            optimizer.step()

        return model

    def fit_neuralnet1(self):
        input_dim = self.xtensor.shape[1]

        class nnregr(nn.Module):
            def __init__(self, input_dim, hidden_dim=1):
                super(nnregr, self).__init__()
                self.lin1 = nn.Linear(input_dim, hidden_dim)

            def forward(self, x):
                return F.relu(self.lin1(x))

        device = "cuda" if torch.cuda.is_available() else "cpu"

        ypred = torch.empty(0)
        ytrue = torch.empty(0)
        cens = torch.empty(0)

        for ff in self.folds:
            tx = self.xtensor[ff[1]]
            tc = self.ctensor[ff[1]]
            ty = self.ytensor[ff[1]]

            vx = self.xtensor[ff[0]]
            vc = self.ctensor[ff[0]]
            vy = self.ytensor[ff[0]]

            model = nnregr(input_dim)

            trained_m = self.train_net(model, tx, ty, tc, loss_fx="mse")

            trained_m.eval()
            with torch.no_grad():
                ypred = torch.cat((ypred, trained_m(vx.to(device)).to("cpu")), 0)
                ytrue = torch.cat((ytrue, vy), 0)
                cens = torch.cat((cens, vc), 0)

        with torch.no_grad():
            mask = cens == 1
            return torch.mean(abs(ypred[mask] - ytrue[mask]))

    def fit_neuralnet2(self):
        input_dim = self.xtensor.shape[1]
        hidden_dim = np.floor(input_dim / 4).astype(int)
        if hidden_dim < 2:
            hidden_dim = 2

        class nnregr(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super(nnregr, self).__init__()
                self.lin1 = nn.Linear(input_dim, hidden_dim)
                self.lin2 = nn.Linear(hidden_dim, 1)
                self.drop = nn.Dropout(0.2)

            def forward(self, x):
                x = F.relu(self.lin1(x))
                x = self.drop(x)
                return F.relu(self.lin2(x))

        device = "cuda" if torch.cuda.is_available() else "cpu"

        ypred = torch.empty(0)
        ytrue = torch.empty(0)
        cens = torch.empty(0)

        for ff in self.folds:
            tx = self.xtensor[ff[1]]
            tc = self.ctensor[ff[1]]
            ty = self.ytensor[ff[1]]

            vx = self.xtensor[ff[0]]
            vc = self.ctensor[ff[0]]
            vy = self.ytensor[ff[0]]

            model = nnregr(input_dim, hidden_dim)

            trained_m = self.train_net(model, tx, ty, tc, loss_fx="mse")

            trained_m.eval()
            with torch.no_grad():
                ypred = torch.cat((ypred, trained_m(vx.to(device)).to("cpu")), 0)
                ytrue = torch.cat((ytrue, vy), 0)
                cens = torch.cat((cens, vc), 0)

        with torch.no_grad():
            mask = cens == 1
            return torch.mean(abs(ypred[mask] - ytrue[mask]))

    def fit_neuralnet2_lnorm_lrt_loss(self):
        input_dim = self.xtensor.shape[1]
        hidden_dim = np.floor(input_dim / 4).astype(int)
        if hidden_dim < 2:
            hidden_dim = 2

        class nnregr(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super(nnregr, self).__init__()
                self.lin1 = nn.Linear(input_dim, hidden_dim)
                self.lin2 = nn.Linear(hidden_dim, 2)
                self.drop = nn.Dropout(0.2)

            def forward(self, x):
                x = F.relu(self.lin1(x))
                x = self.drop(x)
                x = self.lin2(x)
                return {'mu':x[0], 'sigma':F.softplus(x[1]) + 1e-6}  # Ensure sigma is positive

        device = "cuda" if torch.cuda.is_available() else "cpu"

        ypred = torch.empty(0)
        ytrue = torch.empty(0)
        cens = torch.empty(0)

        for ff in self.folds:
            tx = self.xtensor[ff[1]]
            tc = self.ctensor[ff[1]]
            ty = self.ytensor[ff[1]]

            vx = self.xtensor[ff[0]]
            vc = self.ctensor[ff[0]]
            vy = self.ytensor[ff[0]]

            model = nnregr(input_dim, hidden_dim)

            trained_m = self.train_net(model, tx, ty, tc, loss_fx="lnorm_lrt")

            trained_m.eval()
            with torch.no_grad():
                params = trained_m(vx.to(device)).to("cpu")
                npred = torch.exp((params['sima']**2)/2)/params["mu"]
                ypred = torch.cat((ypred, npred ), 0)
                ytrue = torch.cat((ytrue, vy), 0)

        with torch.no_grad():
            return torch.mean(abs(ypred - ytrue))

    def extract_models_mae(
        self,
        model_list=[
            "weibul",
            "gengamma",
            "lognorm",
            "neuralnet1layer",
            "neuralnet2layer",
            "neuralnet2layer_lnorm_lrt_loss",
        ],
    ):
        """
        :param model_list: It does not, by default, run the neural nets.
        :return:
        """
        maedf = pd.DataFrame({"model": model_list, "mae": [None] * len(model_list)})

        for mm in model_list:
            maedf.loc[maedf["model"] == mm, "mae"] = self.modeldict[mm]()

        return maedf
