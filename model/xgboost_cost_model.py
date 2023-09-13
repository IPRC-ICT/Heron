"""XGBoost as cost model"""

import time

import numpy as np


from tvm.autotvm import feature
from tvm.autotvm.utils import get_rank
from tvm.autotvm.tuner.metric import max_curve, recall_curve, cover_curve
from tvm.autotvm.tuner.model_based_tuner import FeatureCache
xgb = None

class XGBoostCostModel:
    def __init__(self, task, config, num_threads=None, log_interval=25):
        global xgb
        try:
            if xgb is None:
                xgb = __import__("xgboost")
        except ImportError:
            raise ImportError(
                "XGBoost is required for XGBoostCostModel. "
                "Please install its python package first. "
                "Help: (https://xgboost.readthedocs.io/en/latest/) "
            )

        self.task = task
        self.target = task.target
        self.loss_type = config.loss_type
        self.num_threads = num_threads
        self.log_interval = log_interval

        if self.loss_type == "reg":
            self.xgb_params = {
                "max_depth": 7,
                "gamma": 0.00001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.00,
                "alpha": 0,
                "objective": "reg:linear",
            }
        elif self.loss_type == "rank":
            self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.00,
                "alpha": 0,
                "objective": "rank:pairwise",
            }
        else:
            raise RuntimeError("Invalid loss type: " + loss_type)

        self.xgb_params["verbosity"] = 0
        if num_threads:
            self.xgb_params["nthread"] = num_threads
        self.bst = None

        self._sample_size = 0


    def train(self, perf_buffer):
        x_train = np.array(perf_buffer.data_x)
        y_train = np.array(perf_buffer.data_y)
        self.fit(x_train, y_train)

    def fit(self, x_train, y_train, plan_size = 30):
        tic = time.time()
        y_train = y_train / (max(y_train) + 1e-8)

        valid_index = y_train > 1e-6
        index = np.random.permutation(len(x_train))
        dtrain = xgb.DMatrix(x_train[index], y_train[index])
        self._sample_size = len(x_train)

        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=8000,
            callbacks=[
                custom_callback(
                    stopping_rounds=20,
                    metric="tr-a-recall@%d" % plan_size,
                    evals=[(dtrain, "tr")],
                    maximize=True,
                    fevals=[
                        xgb_average_recalln_curve_score(plan_size),
                    ],
                    verbose_eval=self.log_interval,
                )
            ],
        )


    def predict(self, samples, output_margin=False):
        feas = np.array([s.point for s in samples])
        dtest = xgb.DMatrix(feas)
        if self.bst == None:
            return np.ones(len(samples))
        return self.bst.predict(dtest, output_margin=output_margin)

def custom_callback(
    stopping_rounds, metric, fevals, evals=(), log_file=None, maximize=False, verbose_eval=True
):
    """callback function for xgboost to support multiple custom evaluation functions"""
    # pylint: disable=import-outside-toplevel
    from xgboost.core import EarlyStopException
    from xgboost.callback import _fmt_metric

    try:
        from xgboost.training import aggcv
    except ImportError:
        from xgboost.callback import _aggcv as aggcv

    state = {}
    metric_shortname = metric.split("-")[1]

    def init(env):
        """internal function"""
        bst = env.model

        state["maximize_score"] = maximize
        state["best_iteration"] = 0
        if maximize:
            state["best_score"] = float("-inf")
        else:
            state["best_score"] = float("inf")

        if bst is not None:
            if bst.attr("best_score") is not None:
                state["best_score"] = float(bst.attr("best_score"))
                state["best_iteration"] = int(bst.attr("best_iteration"))
                state["best_msg"] = bst.attr("best_msg")
            else:
                bst.set_attr(best_iteration=str(state["best_iteration"]))
                bst.set_attr(best_score=str(state["best_score"]))
        else:
            assert env.cvfolds is not None

    def callback(env):
        """internal function"""
        if not state:
            init(env)

        bst = env.model
        i = env.iteration
        cvfolds = env.cvfolds

        res_dict = {}

        ##### evaluation #####
        if cvfolds is not None:
            for feval in fevals:
                tmp = aggcv([f.eval(i, feval) for f in cvfolds])
                for k, mean, std in tmp:
                    res_dict[k] = [mean, std]
        else:
            for feval in fevals:
                bst_eval = bst.eval_set(evals, i, feval)
                res = [x.split(":") for x in bst_eval.split()]
                for kv in res[1:]:
                    res_dict[kv[0]] = [float(kv[1])]

        eval_res = []
        keys = list(res_dict.keys())
        keys.sort(key=lambda x: x if metric_shortname not in x else "a" + x)
        for key in keys:
            v = res_dict[key]
            eval_res.append([key] + v)

        ##### print eval result #####
        infos = ["XGB iter: %3d" % i]
        for item in eval_res:
            if "null" in item[0]:
                continue
            infos.append("%s: %.6f" % (item[0], item[1]))

        if log_file:
            with open(log_file, "a") as fout:
                fout.write("\t".join(infos) + "\n")

        ##### choose score and do early stopping #####
        score = None
        for item in eval_res:
            if item[0] == metric:
                score = item[1]
                break
        assert score is not None

        best_score = state["best_score"]
        best_iteration = state["best_iteration"]
        maximize_score = state["maximize_score"]
        if (maximize_score and score > best_score) or (not maximize_score and score < best_score):
            msg = "[%d] %s" % (env.iteration, "\t".join([_fmt_metric(x) for x in eval_res]))
            state["best_msg"] = msg
            state["best_score"] = score
            state["best_iteration"] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(
                    best_score=str(state["best_score"]),
                    best_iteration=str(state["best_iteration"]),
                    best_msg=state["best_msg"],
                )
        elif env.iteration - best_iteration >= stopping_rounds:
            best_msg = state["best_msg"]
            raise EarlyStopException(best_iteration)
      # print(state)

    return callback


# feval wrapper for xgboost
def xgb_max_curve_score(N):
    """evaluate max curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        scores = labels[trials]
        curve = max_curve(scores)
        return "Smax@%d" % N, curve[N] / np.max(labels)

    return feval


def xgb_recalln_curve_score(N):
    """evaluate recall-n curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "recall@%d" % N, curve[N]

    return feval


def xgb_average_recalln_curve_score(N):
    """evaluate average recall-n curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "a-recall@%d" % N, np.sum(curve[:N]) / N

    return feval


def xgb_recallk_curve_score(N, topk):
    """evaluate recall-k curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks, topk)
        return "recall@%d" % topk, curve[N]

    return feval


def xgb_cover_curve_score(N):
    """evaluate cover curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = cover_curve(ranks)
        return "cover@%d" % N, curve[N]

    return feval


def xgb_null_score(_):
    """empty score function for xgb"""

    def feval(__, ___):
        return "null", 0

    return feval
