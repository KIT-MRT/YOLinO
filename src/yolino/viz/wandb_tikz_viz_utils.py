# Copyright 2023 Karlsruhe Institute of Technology, Institute for Measurement
# and Control Systems
#
# This file is part of YOLinO.
#
# YOLinO is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# YOLinO is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# YOLinO. If not, see <https://www.gnu.org/licenses/>.
#
# ---------------------------------------------------------------------------- #
# ----------------------------- COPYRIGHT ------------------------------------ #
# ---------------------------------------------------------------------------- #
import locale
import math
from copy import copy

import numpy as np
import pandas as pd
import tikzplotlib
from matplotlib import pyplot as plt
from pandas import MultiIndex
from tabulate import tabulate
from tqdm import tqdm
from yolino.utils.logger import Log
from yolino.viz.translation import local_german_wording, SHORT, LONG


def is_ok(run, ignore_running=False, subset_filters={}):
    is_running = run.state == "running"
    if is_running and not ignore_running:
        Log.warning(f"{run.name} is still running...")
        return False, is_running

    if "malconfig" in run.tags:
        Log.warning(f"Remove {run.id} because malconfig")
        return False, is_running

    for key in subset_filters:
        if key == "ID" and run.id in subset_filters["ID"]["value"]:
            Log.warning(f"Remove {run.id} because of filter: ID")
            return False, is_running

        if key in run.config:
            filter_vals = np.asarray(subset_filters[key]["value"])
            if subset_filters[key]["equal"] and run.config[key] not in filter_vals:
                Log.warning(f"Remove {run.id} because of {key}!={filter_vals} (={run.config[key]})")
                return False, is_running
            elif not subset_filters[key]["equal"] and run.config[key] in filter_vals:
                Log.warning(f"Remove {run.id} because of {key}={run.config[key]} (={filter_vals})")
                return False, is_running

    return True, is_running


def gather_runs(run_ids=[], sweep_ids=[], use_all_runs=True, ignore_running=False, subset_filters={}, is_argo=True,
                **kwargs):
    import wandb
    Log.warning("Connect to wandb...")
    api = wandb.Api(timeout=19)

    runs = {}
    definition_params = []
    experiment = None
    runs2sweep = {}
    set_has_running = False
    for i, sweep_id in enumerate(sweep_ids):
        if is_argo:
            sweep = api.sweep(f"annkit/argo_po_8p_dn19/{sweep_id}")
        else:
            sweep = api.sweep(f"annkit/tus_po_8p_dn19/{sweep_id}")

        if use_all_runs:
            for v in tqdm(sweep.runs, desc=f"Sweep {sweep_id}"):
                ok, is_running = is_ok(v, ignore_running, subset_filters)
                set_has_running = set_has_running or is_running
                if not ok:
                    continue

                runs.update({v.name: v})
            runs2sweep.update({v.name: f"$r_{i + 1}$" for v in sweep.runs})

            if len(runs) == 0:
                Log.error(f"Sweep {sweep_id} does not contain runs.")

        if experiment is None:
            experiment = "ex_" + sweep.name
        else:
            if experiment != "ex_" + sweep.name:
                Log.error(f"Sweeps have different names! {experiment} and ex_{sweep.name}")
                # experiment = os.path.commonprefix([experiment, sweep.name])
                experiment = experiment + "-ex_" + "_".join([s for s in sweep.name.split("_") if s not in experiment])

        definition_params.append(list(sweep.config["parameters"].keys()))

    definition_params = np.unique(definition_params)
    for run_id in tqdm(run_ids, desc="Runs"):
        try:
            if is_argo:
                run_object = api.run(f"annkit/argo_po_8p_dn19/{run_id}")
            else:
                run_object = api.run(f"annkit/tus_po_8p_dn19/{run_id}")
        except wandb.errors.CommError as ex:
            Log.error(ex)
            continue

        ok, is_running = is_ok(run_object, ignore_running, subset_filters=subset_filters)
        set_has_running = set_has_running or is_running
        if not ok:
            continue

        runs[run_object.name] = run_object

    return definition_params, experiment, runs, runs2sweep, set_has_running


def finish(output, clear=True, width=1, height=0.7, commit_params={}, extra_axis_params=None):
    if not output.endswith(".tex"):
        output += ".tex"

    # plt.draw()
    Log.warning("Export to file://%s_.png" % output, level=1)
    plt.savefig("%s_.png" % output, bbox_inches='tight')

    Log.warning("Export to file://%s" % output, level=1)

    # FIXME: nan can not be plotted
    tikzplotlib.save(output, axis_width=str(width) + "\\textwidth", axis_height=str(height) + "\\textwidth",
                     extra_axis_parameters=extra_axis_params)

    with open(output, "a") as f:
        f.write(f"% {commit_params}\n")

    # plt.show()
    if clear:
        plt.close("all")
        plt.clf()


def handle(value, key, german_wording=None, justification=None, short=True, force_str=False, rm_latex=False):
    if key == "darknet_cfg":
        value = locale.atoi(value[-5])

    if german_wording is None:
        german_wording = local_german_wording[SHORT if short else LONG]

    if type(value) == list:
        if str(value) in german_wording:
            value = german_wording[value]
        else:
            value = ",".join([str(v) for v in value])

    if type(value) == str:
        if value in german_wording:
            value = german_wording[value]
        else:
            for v in german_wording:
                value = value.replace(v, german_wording[v])

        if justification is not None:
            value = value.rjust(justification)

        value = value.replace("[", "")
        value = value.replace("]", "")
        if rm_latex:
            value = remove_latex_cmd(value)
    elif type(value) == np.float64:
        if value > 100:
            Log.error(f"We found a bad value with {key}={value}")
            value = math.inf
        value = float(value)
    elif type(value) == bool:
        value = german_wording[str(value)]

    if force_str:
        value = str(value)
    return value


def remove_latex_cmd(value):
    value = value.replace("\\ac", "")
    value = value.replace("\\gls", "")
    value = value.replace("{", "")
    value = value.replace("}", "")
    value = value.replace("$", "")
    value = value.replace(',', '')
    value = value.replace('\\', '')
    value = value.replace("$", "")
    value = value.replace("^", "")
    value = value.replace("/", "")
    value = value.replace("?", "")
    return value


def handle_df(key, df, german_wording=None, justification=None, short=True):
    return df.apply(handle, key=key, german_wording=german_wording, justification=justification, short=short,
                    force_str=True)


def get_str(rank=0, len_rank=0, val=None, short_float=True, exp_float=False):
    locale.setlocale(locale.LC_ALL, 'de_DE.utf8')
    if type(val) == float:
        if short_float:
            if short_float == 1:
                val = locale.format_string("%.1f", val)
            elif short_float == 2:
                val = locale.format_string("%.2f", val)
        elif exp_float:
            val = "$10^{%d}$" % int(math.log10(val))
        else:
            val = locale.format_string("%f", val)
    return rank_str(rank, len_rank, val)


def rank_str(rank, len_rank, val):
    if rank == 0 and len_rank > 2:
        return "\\first{" + val + "}"
    elif rank == 1 and len_rank > 3:
        return "\snd{" + val + "}"
    elif rank == 2 and len_rank > 5:
        return "\\third{" + val + "}"
    else:
        return "" + val


def float2_format(val):
    locale.setlocale(locale.LC_ALL, 'de_DE.utf8')
    if val == math.inf:
        return "$\\infty$"

    if type(val) == int:
        return locale.format_string("%d", val)

    return locale.format_string("%.2f", val)


def float1_format(val):
    locale.setlocale(locale.LC_ALL, 'de_DE.utf8')
    if val == math.inf:
        return "$\\infty$"

    if type(val) == int:
        return locale.format_string("%d", val)

    return locale.format_string("%.1f", val)


def get_str_series(rank=0, len_rank=1, df: pd.DataFrame = None, format=float2_format):
    locale.setlocale(locale.LC_ALL, 'de_DE.utf8')
    df_str = df.apply(format)
    return rank_str(rank, len_rank, df_str)


def get_str_df(rank=0, len_rank=0, df: pd.DataFrame = None, format=float2_format):
    locale.setlocale(locale.LC_ALL, 'de_DE.utf8')
    df_str = df.applymap(format)
    return rank_str(rank, len_rank, df_str)


def print_table_from_pandas(colalign, def_columns, df_runs, eval_columns, path, params={}, group_by=None,
                            german_wording=None, comment_params={}, add_hline=[], overwrite_index=True,
                            has_running=False):
    assert not df_runs.duplicated().any()

    all_columns = [*def_columns, *eval_columns]

    df_runs: pd.DataFrame

    if type(df_runs.columns) == MultiIndex:
        multiindex_columns = [german_wording["_".join(d) if d[1] != "" else d[0]]
                              for d in df_runs.columns.values
                              if d[0] in all_columns]

    df_runs = df_runs.rename(columns=german_wording)
    if overwrite_index:
        df_runs = df_runs.set_index(german_wording[def_columns[0]])

    try:
        all_de_columns = [german_wording[k] for k in all_columns if german_wording[k] in df_runs.columns]
    except KeyError as er:
        Log.error("Did you put all your variables in translation.py?")
        raise KeyError(str(type(er)).replace("<class", "").replace(">", "").replace("'", "") + " with " + str(er))

    Log.warning(f"Write latex file to file://{path}")
    with open(path, "w") as f:

        f.write(f"% {params}\n")
        for row in range(len(df_runs)):
            row_name = df_runs.iloc[row]['name']
            row_id = df_runs.iloc[row]['ID']
            row_index_value = df_runs.index[row]
            if type(df_runs.columns) == MultiIndex:
                row_values = df_runs.iloc[row][[german_wording[d] for d in def_columns]]
                row_values = {k[0]: v for k, v in row_values.items()}
            else:
                row_values = dict(
                    df_runs.iloc[row][[german_wording[d] for d in def_columns if german_wording[d] in df_runs.columns]])
            row_values[german_wording[def_columns[0]]] = row_index_value
            row_values_str = "; ".join([f"{k}: {v}" for k, v in row_values.items()])
            f.write(f"% {row_name} - {row_id} - {row_values_str if len(row_values) > 0 else ''} " + "\n")

        if comment_params["yolino"]["has_uncommitted"] or has_running or len(df_runs) == 0:
            f.write("\\tofix")
        else:
            f.write("\\todoi")
        f.write("{" + ("experiment: %s\\\\" % str(comment_params['experiment'])) \
                + ("running: %s; " % str(has_running)) \
                + ("dirty: %s; " % str(comment_params['yolino']["has_uncommitted"])) \
                + ("commit: %s; " % str(comment_params['yolino']["short_commit"])) \
                + "}\n")

        if len(df_runs) == 0:
            return

        is_long = len(df_runs) > 100
        if is_long:
            f.write("\\begin{longtable}{" + colalign + "}\n")
        else:
            f.write("\\begin{tabular}{" + colalign + "}\n")

        if len(group_by) == 0:
            if type(df_runs.columns) == MultiIndex:
                lines = tabulate(df_runs[all_de_columns], headers=multiindex_columns, tablefmt="latex_raw")
            else:
                lines = tabulate(df_runs[all_de_columns],
                                 headers=[german_wording[def_columns[0]],
                                          *all_de_columns] if overwrite_index else all_de_columns,
                                 tablefmt="latex_raw", disable_numparse=True)
            lines = lines.splitlines(keepends=True)

            f.writelines(lines[1:4])
            if is_long:
                # \endhead % Kopf in longtable ist zu Ende
                f.write("\\endhead\n")
                f.write("\\hline\n\\endfoot\n")

            if len(add_hline) > 0:
                for l_idx in range(len(add_hline)):
                    if l_idx == 0:
                        f.writelines(lines[4:add_hline[0] + 4])
                    else:
                        f.writelines(lines[4 + add_hline[l_idx - 1]:add_hline[l_idx] + 4])

                    f.writelines("\\hline\n")

                f.writelines(lines[add_hline[l_idx] + 4:])
            else:
                f.writelines(lines[4:])
        else:
            for i, data in enumerate(df_runs.groupby(by=[german_wording[i] for i in group_by])):
                name, group = data
                lines = tabulate(group[all_de_columns],
                                 headers=[german_wording[def_columns[0]], *all_de_columns],
                                 tablefmt="latex_raw", disable_numparse=True)
                lines = lines.splitlines(keepends=True)

                if i == 0:

                    f.writelines(lines[1:4])
                    if is_long:
                        f.write("\\endhead\n")
                        f.write("\\hline\n\\endfoot\n")

                f.writelines(lines[4:-2])
                f.writelines("\\hline\n")
            if is_long:
                f.write("\\caption{\\longtablecaption}")
                f.write("\\label{\\longtablelabel}")
                f.write("\\end{longtable}\n")
            else:
                f.write("\\end{tabular}\n")


def df_2_ranked_str_df(df_runs, eval_columns, group_by=[], round_1_columns=[], descending_columns=[],
                       no_sort_columns=[]):
    # special def column handling
    if "learning_rate" in df_runs.columns:
        df_runs["learning_rate"] = df_runs["learning_rate"].apply(lambda e: locale.format_string("%.e", e))

    df_runs_str = copy(df_runs).astype(str)

    # rank columns
    if len(group_by) > 0:
        groups = df_runs.groupby(by=group_by)
    else:
        groups = df_runs
    ranking = groups[eval_columns].rank(ascending=True, method="dense", na_option="bottom").astype(int)
    ranking[descending_columns] = groups[descending_columns].rank(ascending=False, method="dense",
                                                                  na_option="bottom").astype(int)
    for rank in range(ranking.max().max()):
        df_runs_str[ranking == rank + 1] = get_str_df(rank=rank, len_rank=ranking.max().max(),
                                                      df=df_runs.loc[:, eval_columns])
    for c in round_1_columns:
        for rank in range(ranking.max().max()):
            is_this_rank = ranking[c] == rank + 1
            if sum(is_this_rank) < 1:
                continue
            if c in no_sort_columns:
                no_sort_columns.remove(c)
                df_runs_str.loc[is_this_rank, c] = get_str_series(df=df_runs.loc[is_this_rank, c],
                                                                  format=float1_format)
            else:
                df_runs_str.loc[is_this_rank, c] = get_str_series(rank=rank, len_rank=ranking.max().max(),
                                                                  df=df_runs.loc[is_this_rank, c],
                                                                  format=float1_format)

    df_runs_str[no_sort_columns] = get_str_df(df=df_runs[no_sort_columns])

    return df_runs_str


marker_for_lines = [*([""] * 7), '.', '*', '+', 'o']
marker = ("*", "+", ".", "o")

linestyle_tuple = {
    'solid': 'solid',  # Same as (0, ()) or '-'
    'dotted': 'dotted',  # Same as (0, (1, 1)) or ':'
    'dashed': 'dashed',  # Same as '--'
    'dashdot': 'dashdot',  # Same as '-.'
    # 'loosely dotted':        (0, (1, 10)),
    # 'dotted':                (0, (1, 1)),
    # 'densely dotted':        (0, (1, 1)),
    # 'long dash with offset': (5, (10, 3)),
    # 'loosely dashed':        (0, (5, 10)),
    # 'dashed':                (0, (5, 5)),
    'densely dashed': (0, (5, 1)),

    # 'loosely dashdotted':    (0, (3, 10, 1, 10)),
    # 'dashdotted': (0, (3, 5, 1, 5)),
    'densely dashdotted': (0, (3, 1, 1, 1)),

    # 'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
    # 'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
    'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1)),
    'solid1': 'solid',
    'solid2': 'solid',
    'solid3': 'solid',
    'solid4': 'solid',
    'solid5': 'solid',
    'solid6': 'solid',
    'solid7': 'solid'}
#
