import functools
from typing import List

import ipdb
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression


def area_under_accuracy_coverage_curve(u, a):
    # area under the rejection-VALUE curve, where VALUE could be accuracy, etc.
    df = pd.DataFrame({"u": u, "a": a}).sort_values("u", ascending=True)
    df["amean"] = df["a"].expanding().mean()
    return metrics.auc(np.linspace(0, 1, len(df)), df["amean"])


def area_under_roc(u, a):
    fpr, tpr, thresholds = metrics.roc_curve(a.astype(int), -u, pos_label=1)
    return metrics.auc(fpr, tpr)


class PCA_LR:
    def __init__(self) -> None:
        self.pca = None
        self.clf = None
        self.columns = None

    def fit(self, neg_cdf, xs, y="acc", n_components=10, regression=False):
        orig_neg_cdf = neg_cdf.copy()
        self.columns = xs
        neg_cdf = neg_cdf.dropna(subset=xs + [y], how="any")
        X = neg_cdf.reindex(columns=xs).values
        # clf = LogisticRegression(random_state=42, penalty='l1', solver='liblinear').fit(X, y)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X)
        X = self.pca.transform(X)
        if regression:
            self.clf = LinearRegression().fit(X, 1 - neg_cdf[y].values)

        else:
            self.clf = LogisticRegression(random_state=42).fit(X, 1 - neg_cdf[y].values)
        neg_conf = self.transform(orig_neg_cdf).reindex(neg_cdf.index)
        _auroc = Summarizer.compute_metric(
            neg_conf, orig_neg_cdf[y].reindex(neg_cdf.index).values.astype(int), "auroc"
        )
        assert _auroc > 0.5, f"{_auroc} <= 0.5"
        print(f"{_auroc:.3f} > 0.5")
        return self

    def transform(self, neg_cdf):
        idx = neg_cdf.index
        neg_cdf = neg_cdf.dropna(subset=self.columns, how="any")
        X = neg_cdf.reindex(columns=self.columns).values
        X = self.pca.transform(X)
        if isinstance(self.clf, LinearRegression):
            ret = self.clf.predict(X)
        else:
            ret = self.clf.predict_proba(X)[:, 1]
        return pd.Series(ret, index=idx)


class Summarizer:
    def __init__(self, uqs, acc, lengths=None) -> None:
        self.uqs = uqs
        self.acc = acc
        self.lengths = lengths
        self.num_gens = min(
            [_["neg_ic"].shape[1] for _ in uqs.values() if _["neg_ic"] is not None]
            + [acc["ia"].shape[1]]
        )

        self.mem = {}

    @classmethod
    def compute_metric(cls, u, a, metric):
        if metric == "auarc":
            return area_under_accuracy_coverage_curve(u, a)
        elif metric == "auroc":
            return area_under_roc(u, a)
        raise NotImplementedError()

    def _summarize_one_exp(
        self,
        uqs,
        acc: pd.Series,
        metric: str = "auarc",
        breakdown: np.ndarray = None,
        breakdown_by: List = None,
    ):
        assert all([len(_) == len(acc) for _ in uqs.values()])
        # area under accuracy-coverage curve
        df = {"acc": acc, "breakdown": breakdown}
        df.update(uqs)
        df = pd.DataFrame(df)
        if (
            len(
                df.dropna(subset=[_ for _ in df.columns if _ != "breakdown"], how="any")
            )
            == 0
        ):
            ipdb.set_trace()
        assert (
            df.dropna(subset=["acc"]).drop("breakdown", axis=1).count().nunique() == 1
        ), df.dropna(subset=["acc"]).drop("breakdown", axis=1).count()
        df = df.dropna(subset=[_ for _ in df.columns if _ != "breakdown"], how="any")

        def _make_one_ser(tdf):
            ret = pd.Series(
                {
                    "acc": tdf["acc"].mean(),
                    "_cnt": len(tdf),
                }
            )
            if ret["acc"] == 0 or ret["acc"] == 1:
                return ret
            ret["oracle"] = self.compute_metric(-tdf["acc"], tdf["acc"], metric)
            ret["blind"] = tdf["acc"].mean()
            for name, _ in uqs.items():
                ret[f"{name}"] = self.compute_metric(tdf[name], tdf["acc"], metric)
            return ret

        ret = {"main": _make_one_ser(df)}
        if breakdown is not None:
            tres = {"overall": ret["main"]}
            for i, min_len in enumerate(breakdown_by[:-1]):
                max_len = breakdown_by[i + 1]
                tdf = df[(df["breakdown"] > min_len) & (df["breakdown"] <= max_len)]
                tres[f"({min_len},{max_len}]"] = _make_one_ser(tdf)
            ret["breakdown"] = pd.DataFrame(tres)
        return ret

    @functools.lru_cache(maxsize=1000)
    def list_uq_perfs(self, y, x, metric: str = "auarc", keep_all_rows=False):
        assert (
            x in {"u", "neg_ic", "neg_mlgc"}
        )  # uncertainty, negative confidence (individual), negative confidence for most likely generation
        assert y in {"ea", "ia", "mlga"}  # expected accuracy, individual accuracy
        assert metric in {"auarc", "auroc", "pickmax"}
        assert not (metric == "auroc" and y == "ea")
        assert not (x == "neg_ic" and y == "ea")
        summ = None
        if metric == "pickmax":
            assert x == "neg_ic" and y == "ia"
            summ = self._maximize_acc().mean()
        elif y == "ea":
            assert x == "u" and metric == "auarc"
            summ = self._summarize_one_exp(
                {k: v[x] for k, v in self.uqs.items()}, self.acc[y], metric
            )["main"]
        elif y == "ia":
            assert x in {"neg_ic", "u"}
            summ = []
            for ith_gen in range(self.num_gens):
                summ.append(
                    self._summarize_one_exp(
                        {
                            k: v[x] if x == "u" else v[x][ith_gen]
                            for k, v in self.uqs.items()
                        },
                        self.acc[y].loc[:, ith_gen],
                        metric,
                    )["main"]
                )
            summ = sum(summ) / len(summ)
        elif y == "mlga":
            assert x in {"u", "neg_mlgc"}
            summ = self._summarize_one_exp(
                {k: v[x] for k, v in self.uqs.items()}, self.acc[y], metric
            )["main"]
        if summ is None:
            raise ValueError(f"{y}~{x} is not valid")
        summ = summ.sort_values()
        if metric != "pickmax" and not keep_all_rows:
            summ = summ.drop(["_cnt", "acc"])
            assert summ["oracle"] == summ.max()
            summ = summ.drop(["oracle", "blind"])
        return summ  # .idxmax()

    def _maximize_acc(self):
        acc = self.acc["ia"]
        # maximize accuracy, by choosing the prediction with the highest confidence
        ret = {}
        for uq_name, uq in self.uqs.items():
            assert uq["neg_ic"].shape[1] <= acc.shape[1]
            uq = uq["neg_ic"].reindex(acc.index).values  # [:, :acc.shape[1]]
            idx = np.argmin(uq, axis=1)
            ret[uq_name] = pd.Series(
                [row[_] for row, _ in zip(acc.values, idx)], acc.index
            )
        return pd.DataFrame(ret).dropna(how="any")


if __name__ == "__main__":
    pass
