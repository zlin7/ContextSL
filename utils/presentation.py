import glob
import os
from importlib import reload
from typing import Union

import ipdb
import pandas as pd
from scipy import stats


def g_ALLMETHODS():
    return [
        "blind",
        "acc",
        "oracle",
        "degree|agreement_w",
        "self_prob",
        "nll|unnorm",
        "nll|norm",
        "sar",
        "attnnll@10",
        "attnnll_nexttoken@10",
        "semanticEntropy|norm",
        "semanticEntropy|unnorm",
        "semanticEntropyFROMattnnll@10",
    ]


def g_METHOD_MAPs():
    return {
        "semanticEntropy|unnorm": "SE",
        "semanticEntropy|norm": "SE(norm)",
        "semanticEntropyFROMattnnll@10": "SE+CSL",
        "self_prob": "P(true)",
        "sar": "TokenSAR",
        "nll|unnorm": "SL",
        "nll|norm": "SL(norm)",
        "blind": "Random",
        "oracle": "Upper Bound",
        "acc": "Base Accuracy",
        "attnnll@10": "CSL",
        "attnnll_nexttoken@10": "CSL-Next",
        "degree|agreement_w": "Deg(E)",
    }


def g_ALLDATASETS():
    ret = []
    for data in [
        "triviaqa_new",
        "coqa_new",
        "nq_open_new",
    ]:
        for model in [
            "llama2-13b",
            "mistral-7b",
            "gemma-7b",
        ]:
            ret.append(f"{data}({model})")
    return ret


def g_DATASET_MAPs():
    ret = {
        k: k.replace("_new", "")
        .replace("nq_open(", "nq(")
        .replace("triviaqa(", "trivia(")
        .replace("coqa(", "coqa(")
        .replace("-13b", "")
        .replace("-7b", "")
        for k in g_ALLDATASETS()
    }
    return ret


def g_INVALID_METHODS():
    return [
        "oracle",
        "acc",
    ]


def g_METHOD_ORDER():
    _map = g_METHOD_MAPs()
    return {_map[_]: i for i, _ in enumerate(g_ALLMETHODS())}


def filter(df, eq=None):
    if eq is not None:
        for key, val in eq.items():
            df = df[df[key] == val].drop(columns=key)
    return df


def _default_formatter(mean, std):
    if pd.isnull(std):
        return f"{mean:.1f}"
    return f"{mean:.2f}$\pm${std:.2f}"


def summarize_mean_std_pval(
    values_df,
    paired=False,
    higher_better=True,
    target: float = None,
    twosided=False,
    invalid_methods=None,
    all_methods=None,
) -> pd.DataFrame:
    if invalid_methods is None:
        invalid_methods = g_INVALID_METHODS()
    if all_methods is None:
        all_methods = g_ALLMETHODS()
    # values_df[col] is a bunch of random values to compare
    values_df = values_df.copy()
    if target is not None:
        values_df -= target
    if not higher_better:
        values_df = -values_df

    summ = values_df.describe().reindex(["count", "mean", "std"]).T
    summ = summ.reindex([_ for _ in all_methods if _ in summ.index])
    summ["count"] = summ["count"].astype(int)
    summ = summ.sort_values("mean", ascending=False)
    msk = summ["count"] == summ["count"].max()
    summ.loc[summ.index[msk], "rank"] = summ["mean"][msk].rank()
    for best_method in summ["rank"].sort_values(ascending=False).index:
        if best_method not in invalid_methods:
            break
    for method in summ.index[msk]:
        if target is not None:
            assert not paired
            summ.loc[method, "pval"] = stats.ttest_1samp(
                values_df[method], 0, alternative="two-sided" if twosided else "less"
            ).pvalue
        else:
            pval_compute = stats.ttest_rel if paired else stats.ttest_ind
            summ.loc[method, "pval"] = pval_compute(
                values_df[method],
                values_df[best_method],
                alternative="two-sided" if twosided else "less",
            ).pvalue
            if pd.isnull(summ.loc[method, "pval"]) and method == best_method:
                summ.loc[method, "pval"] = 1.0
    if not higher_better:
        summ["mean"] = -summ["mean"]
    if target is not None:
        summ["mean"] += target
    return summ


def create_printable_ser_util(
    summ, formatter=_default_formatter, scale=1, pval=0.01, invalid_methods=None
):
    if invalid_methods is None:
        invalid_methods = g_INVALID_METHODS()
    ret = {}
    mask = {}
    best_method = summ["rank"][~summ.index.isin(invalid_methods)].idxmax()
    num_with_reps = summ["std"].notnull().sum()
    assert num_with_reps == 0 or num_with_reps == summ["mean"].count()
    for method in summ.index:
        assert isinstance(method, str)
        ret[method] = formatter(
            summ.loc[method, "mean"] * scale, summ.loc[method, "std"] * scale
        )
        if num_with_reps == 0:
            mask[method] = (
                1 if summ.loc[method, "mean"] == summ.loc[best_method, "mean"] else 0
            )
        else:
            if pval is None:
                # check if the means are within 1 std of each other
                mask[method] = (
                    1
                    if abs(summ.loc[method, "mean"] - summ.loc[best_method, "mean"])
                    < summ.loc[best_method, "std"]
                    and method not in invalid_methods
                    else 0
                )
            else:
                mask[method] = (
                    1
                    if summ.loc[method, "pval"] > pval and method not in invalid_methods
                    else 0
                )
    return pd.Series(ret).astype(str).reindex(summ.index), pd.Series(mask).astype(
        int
    ).reindex(summ.index)


def create_printable_ser_2(
    summ, formatter=_default_formatter, scale=1, pval=0.01, invalid_methods=None
):
    df, mask = create_printable_ser_util(summ, formatter, scale, pval, invalid_methods)
    _, mask2 = create_printable_ser_util(
        summ, formatter, scale, pval, ["oracle", "acc"]
    )
    mask2[mask2 > 0] = 2
    # assert mask.count() == mask2.count()
    # take the maximum between the two masks
    mask = mask + mask2
    return df, mask


def create_printable_df(
    summs,
    create_ser_fn=create_printable_ser_util,
    formatter=_default_formatter,
    scale: Union[float, int, dict] = 1,
    pval=0.01,
):
    METHOD_ORDER = g_METHOD_ORDER()
    METHOD_MAPs = g_METHOD_MAPs()
    DATASET_MAPs = g_DATASET_MAPs()
    ret = {}
    mask = {}
    for dataset, summ in summs.items():
        if dataset.startswith("coqa("):
            continue
        # summ is indexed by methods, and has columns: count, mean, std, rank, pval
        _scale = scale[dataset] if isinstance(scale, dict) else scale
        # ipdb.set_trace()
        ret[dataset], mask[dataset] = create_ser_fn(summ, formatter, _scale, pval=pval)
    # print(ret.keys())
    ret, mask = pd.DataFrame(ret), pd.DataFrame(mask)
    ret = ret.reindex([_ for _ in g_ALLMETHODS() if _ in ret.index])
    mask = mask.reindex([_ for _ in g_ALLMETHODS() if _ in mask.index])

    ret.index = ret.index.map(METHOD_MAPs)
    mask.index = mask.index.map(METHOD_MAPs)
    sidx = ret.index[ret.index.map(METHOD_ORDER).argsort()]

    ret.columns = ret.columns.map(DATASET_MAPs)
    mask.columns = mask.columns.map(DATASET_MAPs)
    return ret.reindex(sidx), mask.reindex(sidx)


# =====================================================Latex Handling


class LatexPrinter:
    _MIDRULE = "\\midrule"
    _BOTTOM = "\\bottomrule"

    def __init__(self, mask_format=None, fill_nan="\\textendash", pad=True) -> None:
        if mask_format is None:

            def mask_format(s, flag):
                if flag == 0:
                    return s
                assert flag == 1
                return "\\textbf{%s}" % s

        self.mask_format = mask_format
        self.fill_nan = fill_nan
        self.pad = pad

    def _get_formatted_cells(self, df, mask_df):
        _fmt = (
            lambda s, flag: self.fill_nan if pd.isnull(s) else self.mask_format(s, flag)
        )
        assert (
            df.dtypes == "O"
        ).all(), "Expect the dataframe to be full of strings only."
        assert df.shape == mask_df.shape
        strs = []
        for idx in df.index:
            strs.append(
                [str(idx)]
                + [_fmt(df.loc[idx, c], mask_df.loc[idx, c]) for c in df.columns]
            )
        return strs

    def _compute_column_widths(self, cells):
        formattable_cells = [line for line in cells if isinstance(line, list)]
        if not self.pad:
            return [1] * len(formattable_cells[0])
        return [
            max([len(_[j]) for _ in formattable_cells])
            for j in range(len(formattable_cells[0]))
        ]

    def _prints_df_helper(
        self, df, mask_df, table_name="", skip_header=False, colwidths=None
    ):
        # Can repeatedly call this function with different formatter and masks
        lines = [[table_name] + list(map(str, df.columns))]
        lines.extend(self._get_formatted_cells(df, mask_df))
        new_lines = []
        if colwidths is None:
            colwidths = self._compute_column_widths(lines)
        for i, line in enumerate(lines):
            if i == 0 and skip_header:
                continue
            new_lines.append(
                " & ".join([_.rjust(colwidths[j]) for j, _ in enumerate(line)]) + "\\\\"
            )
            if i == 0:
                new_lines.append(self._MIDRULE)
        return new_lines

    def print_df(self, df, mask_df, table_name="", skip_header=False, add_line=None):
        lines = self._prints_df_helper(df, mask_df, table_name, skip_header)
        if add_line is not None:
            for i, l in enumerate(lines):
                print(l)
                if i in add_line:
                    print("\\midrule")
        else:
            print("\n".join(lines))

    def _prints_dfs(
        self, dfs, mask_dfs, column_names=None, row_names=None, multirow=False
    ):
        assert len(dfs) == len(mask_dfs)
        assert all([len(dfs1) == len(mask_dfs[i]) for i, dfs1 in enumerate(dfs)])
        # M x N
        M, N = len(dfs), len(dfs[0])
        if column_names is None:
            column_names = [""] * (N)
        if row_names is None:
            row_names = [""] * (M)

        middle_cells = [
            [self._get_formatted_cells(dfs[i][j], mask_dfs[i][j]) for j in range(N)]
            for i in range(M)
        ]

        strs = []
        if N > 1:
            strs.append(
                " & "
                + " & ".join(
                    [
                        "\\multicolumn{%d}{|c}{%s}"
                        % (dfs[0][j].shape[1], column_names[j])
                        for j in range(N)
                    ]
                )
                + "\\\\"
            )
            # strs.append(self._MIDRULE)
        strs.append(
            [""] * (2 if M > 1 and multirow else 1)
            + [str(_) for j in range(N) for _ in dfs[0][j].columns]
        )
        strs.append(self._MIDRULE)
        for i in range(M):
            middle_cells_i = middle_cells[i]
            if M > 1 and not multirow:
                strs.extend([f"{row_names[i]}\\\\", self._MIDRULE])
            for ii in range(len(middle_cells_i[0])):
                curr_line = []
                if M > 1 and multirow:
                    curr_line.append(
                        "\multirow{%d}{*}{%s}" % (len(dfs[i][0]), row_names[i])
                        if ii == 0
                        else ""
                    )
                for j in range(N):
                    curr_line.extend(middle_cells_i[j][ii][min(1, j) :])
                strs.append(curr_line)
            if i < M - 1:
                strs.append(self._MIDRULE)
        colwidths = self._compute_column_widths(strs)
        new_lines = []
        for i, line in enumerate(strs):
            if isinstance(line, list):
                new_lines.append(
                    " & ".join([_.rjust(colwidths[j]) for j, _ in enumerate(line)])
                    + "\\\\"
                )
            else:
                new_lines.append(line)
        new_lines.append(self._BOTTOM)
        return new_lines

    def print_dfs(
        self, dfs, mask_dfs, column_names=None, row_names=None, multirow=False
    ):
        print(
            "\n".join(
                self._prints_dfs(
                    dfs, mask_dfs, column_names, row_names, multirow=multirow
                )
            )
        )

    @classmethod
    def test(cls):
        test_df = pd.DataFrame([["1", "2"], ["3", "4"]], index=["asdasdasa", "ad"])
        test_mask = pd.DataFrame(0, index=test_df.index, columns=test_df.columns)
        o = cls(pad=True)
        o.print_dfs(
            [[test_df, test_df]],
            [[test_mask, test_mask]],
            column_names=["Datafram1", "dataframe2"],
        )
        o.print_dfs([[test_df], [test_df]], [[test_mask], [test_mask]])
        o.print_dfs([[test_df], [test_df]], [[test_mask], [test_mask]], multirow=True)


def printable_df_to_latex(
    df,
    mask_df,
    mask_format=None,
    table_name="",
    fill_nan="\\textendash",
    skip_header=False,
    pad=True,
    **kwargs,
):
    LatexPrinter(mask_format, fill_nan=fill_nan, pad=pad).print_df(
        df, mask_df, table_name=table_name, skip_header=skip_header, **kwargs
    )
    return


def compute_cdf(x):
    import numpy as np

    count, bins_count = np.histogram(x, bins=100)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return bins_count[1:], cdf
