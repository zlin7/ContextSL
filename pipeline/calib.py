import os
from importlib import reload
from typing import Dict, List, Optional, Tuple

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import persist_to_disk as ptd
import simple_disk_queue as sdq
import torch
import tqdm
from sklearn.calibration import CalibratedClassifierCV

import _settings
import pipeline.summ
from pipeline.eval_uq import area_under_accuracy_coverage_curve, area_under_roc


class Uncal(object):
    def __init__(self) -> None:
        pass

    def fit(self, y_score, y):
        pass

    def predict_proba(self, y_score):
        assert y_score.shape[1] == 1

        return np.concatenate([1 - y_score, y_score], axis=1)


class CalibrationEval:
    def __init__(self):
        pass

    @classmethod
    def get_bins(cls, bins):
        if isinstance(bins, int):
            bins = list(np.arange(bins + 1) / bins)
        return bins

    @classmethod
    def assign_bin_old(cls, sorted_ser, bins):
        import bisect

        ret = pd.DataFrame(sorted_ser)
        bin_assign = pd.Series(0, index=sorted_ser.index)
        locs = [bisect.bisect(sorted_ser, b) for b in bins]
        locs[0], locs[-1] = 0, len(ret)
        for i, loc in enumerate(locs[:-1]):
            bin_assign.iloc[loc : locs[i + 1]] = i
        ret["bin"] = bin_assign
        return ret

    @classmethod
    def assign_bin(cls, sorted_ser, bins, adaptive=False):
        ret = pd.DataFrame(sorted_ser)
        if adaptive:
            assert isinstance(bins, int)
            step = len(sorted_ser) // bins
            nvals = [step for _ in range(bins)]
            for _ in range(len(sorted_ser) % bins):
                nvals[-_ - 1] += 1
            ret["bin"] = [ith for ith, val in enumerate(nvals) for _ in range(val)]
            nvals = list(np.asarray(nvals).cumsum())
            bins = [ret.iloc[0]["conf"]]
            for iloc in nvals:
                bins.append(ret.iloc[iloc - 1]["conf"])
                if iloc != nvals[-1]:
                    bins[-1] = 0.5 * bins[-1] + 0.5 * ret.iloc[iloc]["conf"]
        else:
            bins = cls.get_bins(bins)
            import bisect

            bin_assign = pd.Series(0, index=sorted_ser.index)
            locs = [bisect.bisect(sorted_ser, b) for b in bins]
            locs[0], locs[-1] = 0, len(ret)
            for i, loc in enumerate(locs[:-1]):
                bin_assign.iloc[loc : locs[i + 1]] = i
            ret["bin"] = bin_assign
        return ret["bin"], bins

    @classmethod
    def _ECE_loss(cls, summ):
        w = summ["cnt"] / summ["cnt"].sum()
        loss = np.average((summ["conf"] - summ["acc"]).abs(), weights=w)
        # print(loss, w)
        return loss

    @classmethod
    def ECE_confidence(cls, preds, label, bins=15, adaptive=False, return_bins=False):
        df = (
            pd.DataFrame(
                {"conf": preds.max(1), "truth": label, "pred": np.argmax(preds, 1)}
            )
            .sort_values(["conf"])
            .reset_index()
        )
        df["acc"] = (df["truth"] == df["pred"]).astype(int)
        df["bin"], bin_boundary = cls.assign_bin(df["conf"], bins, adaptive=adaptive)
        # df['bin1'] = cls.assign_bin_old(df['conf'], cls.get_bins(bins))['bin']
        # assert df['bin'].eq(df['bin1']).all()
        summ = pd.DataFrame(df.groupby("bin")[["acc", "conf"]].mean())  # .fillna(0.)
        summ["cnt"] = df.groupby("bin").size()
        summ = summ.reset_index()
        # summ['bin'] /= bins
        if return_bins:
            return (
                summ,
                cls._ECE_loss(summ),
                np.mean(np.square(df["conf"].values - df["acc"].values)),
                bin_boundary,
            )
        return (
            summ,
            cls._ECE_loss(summ),
            np.mean(np.square(df["conf"].values - df["acc"].values)),
        )

    @classmethod
    def ECE_class(
        cls, preds, label, bins=15, threshold=0.0, adaptive=False, return_bins=False
    ):
        K = preds.shape[1]
        summs = []
        class_losses = {}
        bin_boundaries = {}
        for k in range(K):
            msk = preds[:, k] >= threshold
            if msk.sum() == 0:
                continue
            df = (
                pd.DataFrame({"conf": preds[msk, k], "truth": label[msk]})
                .sort_values(["conf"])
                .reset_index()
            )
            df["acc"] = (df["truth"] == k).astype(int)
            df["bin"], bin_boundaries[k] = cls.assign_bin(
                df["conf"], bins, adaptive=adaptive
            )
            # df['bin1'] = cls.assign_bin_old(df['conf'], cls.get_bins(bins))['bin']
            # assert df['bin'].eq(df['bin1']).all()
            summ = pd.DataFrame(
                df.groupby("bin")[["acc", "conf"]].mean()
            )  # .reindex(range(bins))#.fillna(0.)
            summ["cnt"] = df.groupby("bin").size()
            summ["k"] = k
            summs.append(summ.reset_index())
            class_losses[k] = cls._ECE_loss(summs[-1])
        class_losses = pd.Series(class_losses)
        class_losses["avg"], class_losses["sum"] = (
            class_losses.mean(),
            class_losses.sum(),
        )
        summs = pd.concat(summs, ignore_index=True)
        # summs['bin'] /= bins
        if return_bins:
            return summs, class_losses, bin_boundaries
        return summs, class_losses

    @classmethod
    def _plot_bars(
        cls,
        df,
        ax,
        nbins,
        title="",
        plot_cnt=False,
        legend_name="Observed Acc",
        _min=10,
        cnt_ax=None,
        legend=True,
    ):
        df = df.copy().sort_values("bin", ascending=True)
        df["gap"] = df["acc"] - df["conf"]
        nbins = np.asarray(nbins)
        # xs = df['bin'] + width * 0.5
        # width = 1. / nbins
        df["left"] = nbins[df["bin"].values]
        df["width"] = (nbins[1:] - nbins[:-1])[df["bin"].values]
        df = df[df["cnt"] > _min]
        # ipdb.set_trace()
        gap_plt = ax.bar(
            df["left"] + 0.3 * df["width"],
            df["gap"].abs(),
            bottom=df.reindex(columns=["acc", "conf"]).min(1),
            width=df["width"] * 0.4,
            label="Gap",
            color="blue",
            zorder=10,
            align="edge",
        )
        acc_plt = ax.bar(
            df["left"],
            df["acc"],
            bottom=0,
            width=df["width"],
            label=legend_name,
            color="green",
            edgecolor="black",
            zorder=0,
            alpha=0.3,
            align="edge",
        )

        ax.plot([0, 1], [0, 1], c="red", linestyle="dashed")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        df["lb"] = nbins[df["bin"].values]
        df["ub"] = nbins[df["bin"].values + 1]

        handles = [gap_plt, acc_plt]
        if plot_cnt:
            # raise NotImplementedError()
            ax2 = cnt_ax or ax.twinx()
            # print(123213123)
            stride = 1.0 / len(nbins)
            # ser = df.set_index('bin')['cnt']
            # ax2.bar(0.5*stride + np.asarray(ser.index), ser.values, width=0.5*stride, alpha=0.2)
            cnt_plt = ax2.plot(
                (df["lb"] + df["ub"]) / 2,
                df["cnt"].fillna(0),
                color="blue",
                alpha=0.5,
                label="Count/bin",
            )
            ax2.set_yscale("log")
            handles.append(cnt_plt[0])

        if legend:
            ax.legend(handles=handles, loc="upper left")
        ax.set_title(title)
        return ax


def eval_routine(df, plot=False, nbins=15, prob_col="prob"):
    assert all([_ in df.columns for _ in [prob_col, "acc", "split"]])

    df = df[df["split"] == "test"]
    # convert df['prob'] into a 2-column matrix
    tP_curr = np.column_stack((1 - df[prob_col], df[prob_col]))
    tY = df["acc"].astype(int).values

    thres = 0.01
    overall_summ, overall_loss, brier_top1 = CalibrationEval.ECE_confidence(
        tP_curr, tY, bins=nbins
    )
    _, overall_loss_adapt, _ = CalibrationEval.ECE_confidence(
        tP_curr, tY, bins=nbins, adaptive=True
    )
    acc = (np.argmax(tP_curr, 1) == tY).mean()

    _eps = 1e-3 / tP_curr.shape[1]
    nll1 = torch.nn.NLLLoss()(
        torch.log(torch.tensor(np.clip(tP_curr, _eps, 1 - _eps))), torch.tensor(tY)
    ).item()

    _one_hot_Y = np.zeros(tP_curr.shape)
    _one_hot_Y[np.arange(len(tY)), tY] = 1
    _sqs = np.square(tP_curr - _one_hot_Y)
    _msk = tP_curr > thres
    return pd.Series(
        {
            "ece": overall_loss * 100,
            "ece_adapt": overall_loss_adapt * 100,
            "acc": acc * 100,
            "brier": np.mean(_sqs),
            "briert": np.mean(_sqs * _msk),
            "AUARC": area_under_accuracy_coverage_curve(-tP_curr[:, 1], tY),
            "AUROC": area_under_roc(-tP_curr[:, 1], tY),
            "NLLTorch": nll1,
            "cnt": len(tY),
        }
    )


def get_calib_obj(method, cv=3, **kwargs):
    if method is None:
        return Uncal()
    if method == "defaultSVC":
        return CalibratedClassifierCV(None, cv=cv)
    raise ValueError(f"Unknown method {method}")


def calibrate_conf(
    summ_obj,
    conf_name,
    setting=None,
    calib_seed=7,
    calib_size=0.2,
    calib_method="default",
):
    if setting == "mlga~neg_mlgc":
        confs = -summ_obj.uqs[conf_name]["neg_mlgc"]
        acc = summ_obj.acc["mlga"]
    elif setting == "ia~neg_ic":
        confs = -summ_obj.uqs[conf_name]["neg_ic"].stack()
        acc = (
            summ_obj.acc["ia"].stack().reindex(confs.index)
        )  # in case num_gens is different
    else:
        raise ValueError(f"Unknown setting {setting}")
    df = (
        pd.DataFrame({"conf": confs, "acc": acc, "split": "test"})
        .dropna(subset=["acc"])
        .sort_index()
    )
    assert df.count().min() == len(df)
    if 0 < calib_size < 1:
        calib_size = int(len(df) * calib_size)
    rs = np.random.RandomState(calib_seed)
    calib_index = rs.choice(df.index, calib_size, replace=False)
    df.loc[calib_index, "split"] = "cal"
    calib_method = get_calib_obj(calib_method)
    calib_method.fit(
        np.expand_dims(df.loc[calib_index, "conf"].values, 1),
        df.loc[calib_index, "acc"].values,
    )
    df["prob"] = calib_method.predict_proba(np.expand_dims(df["conf"].values, 1))[:, 1]
    return df


@ptd.persistf(groupby=["method"], switch_kwarg="cache")
def cached_calibrated(
    path,
    clean=True,
    seed=0,
    setting="mlga~neg_mlgc",
    curve="auroc",
    num_gens=5,
    acc_name: str = "gpt",
    split: str = "test",
    cal_size: int = 1000,
    *,
    method="attnnll@10",
    cal2_size=0.2,
    calib_method="default",
    calib_seed=None,
):
    assert not acc_name.endswith(
        "|acc"
    ), f"acc_name should not end with |acc, got {acc_name}"
    if calib_seed is None:
        calib_seed = seed
    old_path = path
    path = f"{os.path.normpath(_settings.GENERATION_FOLDER)}{path}"
    assert pipeline.summ._clean_path(path) == old_path
    obj = pipeline.summ.UQ_summ(path, clean, split, cal_size, seed)
    acc_name = f"{acc_name}|acc"
    summ_obj = obj.summ(
        setting,
        curve=curve,
        acc_name=acc_name,
        num_gens=num_gens,
        uq_names=[method],
        get_eval_obj=True,
    )
    return calibrate_conf(
        summ_obj,
        method,
        setting,
        calib_size=cal2_size,
        calib_seed=calib_seed,
        calib_method=calib_method,
    )
