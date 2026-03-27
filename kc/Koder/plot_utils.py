import os, tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import io
import uuid
import numpy as np
import pandas as pd

import state as _state
from config import (
    TIME_UNIT_FACTORS,
    normalize_time_unit,
    unit_suffix,
    time_axis_from_seconds,
    hours_to_unit,
)
from data_utils import (
    resolve_plot_titles,
    sanitize_groups,
    average_group_signals,
    _pick_curve_point_for_level,
    estimate_x_hours_from_y,
    estimate_y_from_x_hours,
    parse_concentration_from_group_name,
)


def _store_plot_figure(fig, filename_prefix):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    plot_id = uuid.uuid4().hex
    _state._plot_images[plot_id] = {
        "bytes": buf.getvalue(),
        "download_name": f"{filename_prefix}_{plot_id[:8]}.png",
    }
    # Keep memory bounded.
    while len(_state._plot_images) > 200:
        oldest_id = next(iter(_state._plot_images))
        _state._plot_images.pop(oldest_id, None)
    return plot_id


def generate_global_fit_plot_image(global_fit_result, selected_wells, time_unit="hours", show_residuals=False):
    x = np.array(global_fit_result.get("x", []), dtype=float)
    if len(x) == 0:
        raise ValueError("Global fit saknar x-axel.")
    raw = global_fit_result.get("raw_data", {})
    pred = global_fit_result.get("model_predictions", {})
    resid = global_fit_result.get("residuals", {})
    wells = [w for w in selected_wells if w in raw and w in pred]
    if not wells:
        raise ValueError("Global fit saknar giltiga well-kurvor.")

    palette = [
        "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
        "#0891B2", "#BE123C", "#4F46E5", "#0F766E", "#EA580C",
    ]

    if show_residuals:
        fig, (ax, ax_r) = plt.subplots(
            2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.4]}
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 5.4))
        ax_r = None

    for i, well in enumerate(wells):
        color = palette[i % len(palette)]
        y = np.array(raw[well], dtype=float)
        yhat = np.array(pred[well], dtype=float)
        ax.scatter(x, y, s=9, color=color, alpha=0.35, label=(f"{well} data"))
        ax.plot(x, yhat, color=color, linewidth=2.0, label=(f"{well} fit"))
        if ax_r is not None:
            rr = np.array(resid.get(well, []), dtype=float)
            if len(rr) == len(x):
                ax_r.plot(x, rr, color=color, linewidth=1.2, alpha=0.9, label=well)

    bp = global_fit_result.get("best_params", {})
    default_title = (
        "Global fitting overlay\n"
        f"kn_plus={bp.get('kn_plus', np.nan):.3g}, k2_plus={bp.get('k2_plus', np.nan):.3g}, "
        f"RMSE={global_fit_result.get('fit_error', np.nan):.3g}"
    )
    default_x = f"Time ({unit_suffix(time_unit)})"
    default_y = "Normalized fluorescence (0-1)" if global_fit_result.get("normalized") else "Fluorescence (a.u.)"
    x_label, y_label, title_label = resolve_plot_titles(
        global_fit_result.get("custom_titles", {}),
        default_x,
        default_y,
        default_title,
    )
    ax.set_title(title_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    if ax_r is not None:
        ax_r.axhline(0.0, color="#6B7280", linewidth=1.0, linestyle="--")
        ax_r.set_ylabel("Residual")
        ax_r.set_xlabel(x_label)
        ax_r.grid(True, linestyle="--", linewidth=0.5, alpha=0.8)
    else:
        ax.set_xlabel(x_label)

    fig.tight_layout()
    return _store_plot_figure(fig, "global_fit")


def build_thalf_plot_image(
    session_data,
    selected_wells,
    assignments,
    scale="log",
    x_axis_attr="conc",
    y_axis_attr="half_time",
):
    rows = []
    for well in selected_wells:
        if well not in session_data["well_halftime"]:
            continue
        halftime = session_data["well_halftime"][well]
        if halftime is None:
            continue
        if well not in assignments:
            continue

        rows.append(
            {
                "Well": well,
                "Group": assignments[well]["group"],
                "conc_uM": assignments[well]["conc"],
                "half_time": halftime,
                "attrs": assignments[well].get("attrs", {}),
            }
        )

    if not rows:
        raise ValueError("Inga wells med både giltig halftime och grupp+koncentration.")

    df = pd.DataFrame(rows)
    all_groups = sorted(df["Group"].unique())
    palette = [
        "#3B82F6",
        "#F59E0B",
        "#10B981",
        "#EF4444",
        "#8B5CF6",
        "#06B6D4",
        "#84CC16",
        "#F97316",
        "#EC4899",
        "#6366F1",
    ]

    def axis_value(row, axis_name):
        if axis_name == "conc":
            return row.get("conc_uM")
        if axis_name == "half_time":
            return row.get("half_time")
        attrs = row.get("attrs", {})
        if isinstance(attrs, dict):
            return attrs.get(axis_name)
        return None

    def axis_label(axis_name):
        if axis_name == "conc":
            return "Concentration (µM)"
        if axis_name == "half_time":
            return f"Half-time ({unit_suffix(session_data.get('time_unit', 'hours'))})"
        return axis_name

    fig, ax = plt.subplots(figsize=(8, 5))
    color_idx = 0

    for group_name in all_groups:
        group_df = df[df["Group"] == group_name].copy()

        rep_counts = (
            group_df[group_df["conc_uM"] != 0]
            .groupby("conc_uM")
            .size()
        )

        if not rep_counts.empty:
            target_reps = int(rep_counts.min())
            zero_rows = group_df[group_df["conc_uM"] == 0]
            if not zero_rows.empty:
                group_df = pd.concat(
                    [group_df[group_df["conc_uM"] != 0], zero_rows.iloc[:target_reps]]
                )

        group_df["x_axis_value"] = group_df.apply(lambda r: axis_value(r, x_axis_attr), axis=1)
        group_df["y_axis_value"] = group_df.apply(lambda r: axis_value(r, y_axis_attr), axis=1)
        group_df = group_df.dropna(subset=["x_axis_value", "y_axis_value"])
        if group_df.empty:
            continue

        summary = (
            group_df.groupby("x_axis_value")["y_axis_value"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("x_axis_value")
        )

        if scale == "log":
            summary = summary[summary["x_axis_value"] > 0]
        if summary.empty:
            continue

        color = palette[color_idx % len(palette)]
        color_idx += 1

        for _, row in summary.iterrows():
            std_value = row["std"]
            yerr = 0 if pd.isna(std_value) else std_value
            ax.errorbar(
                row["x_axis_value"],
                row["mean"],
                yerr=yerr,
                fmt="o",
                capsize=4,
                elinewidth=1.5,
                markersize=7,
                color=color,
            )

        ax.plot(
            summary["x_axis_value"],
            summary["mean"],
            color=color,
            linewidth=1.3,
            alpha=0.8,
            label=group_name,
        )

    x_label_name = axis_label(x_axis_attr)
    y_label_name = axis_label(y_axis_attr)
    default_title = f"{y_label_name} vs {'log(' + x_label_name + ')' if scale == 'log' else x_label_name}"
    default_x = x_label_name
    default_y = y_label_name
    x_label, y_label, title_label = resolve_plot_titles(
        session_data.get("custom_titles", {}),
        default_x,
        default_y,
        default_title,
    )

    if scale == "log":
        ax.set_xscale("log")
        if x_axis_attr == "conc":
            unique_concs = sorted(df[df["conc_uM"] > 0]["conc_uM"].unique())
            if unique_concs:
                ax.set_xticks(unique_concs)
                ax.set_xticklabels([f"{c:g}" for c in unique_concs])
                ax.xaxis.set_minor_locator(
                    ticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1)
                )
        ax.set_title(title_label)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    else:
        ax.set_title(title_label)
        ax.grid(True, linestyle="--", linewidth=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="upper left", fontsize=8, title="Group")
    fig.tight_layout()

    return _store_plot_figure(fig, "thalf")


def generate_plot_image(
    time_sec,
    wells_dict,
    selected_wells,
    normalized=False,
    x_from=None,
    x_to=None,
    groups=None,
    time_unit="hours",
    custom_titles=None,
):
    time_h = time_axis_from_seconds(time_sec, time_unit)

    if x_from is not None and x_to is not None and x_from > x_to:
        raise ValueError("'from x' måste vara mindre än eller lika med 'to x'.")

    mask = np.ones_like(time_h, dtype=bool)
    if x_from is not None:
        mask &= time_h >= x_from
    if x_to is not None:
        mask &= time_h <= x_to

    if not np.any(mask):
        raise ValueError("Valt x-intervall innehåller inga datapunkter.")

    time_h = time_h[mask]

    groups = sanitize_groups(groups or {}, selected_wells)
    has_groups = len(groups) > 0
    well_to_group = {}
    for group_name, wells in groups.items():
        for well in wells:
            well_to_group[well] = group_name

    palette = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#999999",
        "#8B4513",
    ]

    color_index = 0
    group_color = {}
    for group_name in sorted(groups.keys()):
        group_color[group_name] = palette[color_index % len(palette)]
        color_index += 1

    individual_color = {}
    shown_group_labels = set()

    fig, ax = plt.subplots(figsize=(8, 5))
    for well in selected_wells:
        if well not in wells_dict:
            continue

        y = np.array(wells_dict[well], dtype=float)
        if len(y) != len(mask):
            continue
        y = y[mask]
        if len(y) != len(time_h):
            continue

        if normalized:
            min_val = np.min(y)
            max_val = np.max(y)
            if max_val - min_val == 0:
                continue
            y = (y - min_val) / (max_val - min_val)

        if well in well_to_group:
            group_name = well_to_group[well]
            color = group_color[group_name]
            if group_name not in shown_group_labels:
                label = group_name
                shown_group_labels.add(group_name)
            else:
                label = None
        else:
            if well not in individual_color:
                individual_color[well] = palette[color_index % len(palette)]
                color_index += 1
            color = individual_color[well]
            label = well

        # If groups are used, grouped wells are described by group labels instead of well IDs.
        if has_groups and well in well_to_group:
            ax.plot(time_h, y, linewidth=1.6, alpha=0.9, color=color, label=label)
        else:
            ax.plot(time_h, y, linewidth=1.6, alpha=0.9, color=color, label=well if not has_groups else label)

    default_x = f"Time ({unit_suffix(time_unit)})"
    if normalized:
        default_y = "Normalized fluorescence (0-1)"
        default_title = "Normalized aggregation curve"
    else:
        default_y = "Fluorescence (a.u.)"
        default_title = "Aggregation curve"
    x_label, y_label, title_label = resolve_plot_titles(custom_titles, default_x, default_y, default_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title_label)

    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    return _store_plot_figure(fig, "plot")


def generate_representative_group_plot_image(
    time_sec,
    wells_dict,
    group_name,
    group_wells,
    representative_wells,
    time_unit="hours",
):
    time_axis = time_axis_from_seconds(time_sec, time_unit)
    rep_set = set(representative_wells or [])

    fig, ax = plt.subplots(figsize=(8, 4.6))
    any_curve = False

    # Draw non-representative curves first in light gray.
    for well in group_wells:
        signal = np.array((wells_dict or {}).get(well, []), dtype=float)
        if len(signal) != len(time_axis):
            continue
        any_curve = True
        if well in rep_set:
            continue
        ax.plot(time_axis, signal, color="#94A3B8", linewidth=1.3, alpha=0.45)

    # Draw representative curves on top with clear color.
    rep_palette = ["#DC2626", "#2563EB", "#059669", "#D97706", "#7C3AED"]
    rep_i = 0
    for well in group_wells:
        if well not in rep_set:
            continue
        signal = np.array((wells_dict or {}).get(well, []), dtype=float)
        if len(signal) != len(time_axis):
            continue
        any_curve = True
        color = rep_palette[rep_i % len(rep_palette)]
        rep_i += 1
        ax.plot(
            time_axis,
            signal,
            color=color,
            linewidth=2.3,
            alpha=0.95,
            label=f"Representative: {well}",
        )

    if not any_curve:
        plt.close(fig)
        raise ValueError(f"No plottable curves in group '{group_name}'.")

    ax.set_xlabel(f"Time ({unit_suffix(time_unit)})")
    ax.set_ylabel("Fluorescence (a.u.)")
    ax.set_title(f"{group_name} - Representative curve control")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    if rep_set:
        ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return _store_plot_figure(fig, f"rep_control_{group_name}")


def generate_representative_control_plot(
    time_sec,
    wells_dict,
    group_name,
    group_wells,
    representative_wells,
    alternative_well=None,
    time_unit="hours",
):
    time_axis = time_axis_from_seconds(time_sec, time_unit)
    rep_set = set(representative_wells or [])
    primary_rep = representative_wells[0] if representative_wells else None
    diverse_reps = list(representative_wells[1:]) if representative_wells else []
    rep_color = "#0072B2"          # primary representative (blue)
    diverse_palette = ["#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#7C3AED"]
    diverse_color_map = {}
    for i, w in enumerate(diverse_reps):
        diverse_color_map[str(w)] = diverse_palette[i % len(diverse_palette)]
    alt_color = "#E69F00"          # selected alternative (orange)
    other_color = "#94A3B8"        # muted gray

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    any_curve = False
    payload_curves = []

    for well in group_wells:
        signal = np.array((wells_dict or {}).get(well, []), dtype=float)
        if len(signal) != len(time_axis):
            continue
        any_curve = True
        y_list = [float(v) for v in signal.tolist()]
        x_list = [float(v) for v in time_axis.tolist()]
        payload_curves.append(
            {
                "well": str(well),
                "x": x_list,
                "y": y_list,
                "is_model_selected": bool(well in rep_set),
            }
        )
        if alternative_well and str(well) == str(alternative_well):
            ax.plot(time_axis, signal, color=alt_color, linewidth=2.5, alpha=0.98, label=f"Alternative: {well}")
        elif primary_rep is not None and str(well) == str(primary_rep):
            ax.plot(time_axis, signal, color=rep_color, linewidth=2.2, alpha=0.95, label=f"Model selected: {well}")
        elif str(well) in diverse_color_map:
            ax.plot(
                time_axis,
                signal,
                color=diverse_color_map[str(well)],
                linewidth=2.1,
                alpha=0.94,
                label=f"Diverse selected: {well}",
            )
        else:
            ax.plot(time_axis, signal, color=other_color, linewidth=1.3, alpha=0.45)

    if not any_curve:
        plt.close(fig)
        raise ValueError(f"No plottable curves in group '{group_name}'.")

    ax.set_xlabel(f"Time ({unit_suffix(time_unit)})")
    ax.set_ylabel("Fluorescence (a.u.)")
    ax.set_title(f"{group_name} - Representative curve control")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    if rep_set:
        ax.legend(loc="upper left", fontsize=8)

    # Stable axis geometry for click-to-curve mapping in browser.
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.12, top=0.88)
    ax_pos = ax.get_position()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    plot_meta = {
        "ax_left": float(ax_pos.x0),
        "ax_right": float(ax_pos.x1),
        "ax_bottom": float(ax_pos.y0),
        "ax_top": float(ax_pos.y1),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
    }
    plot_id = _store_plot_figure(fig, f"rep_control_{group_name}")
    payload = {
        "group_name": str(group_name),
        "primary_representative": (str(primary_rep) if primary_rep is not None else None),
        "diverse_representatives": [str(w) for w in diverse_reps],
        "model_representatives": sorted(list(rep_set)),
        "curves": payload_curves,
        "rep_color": rep_color,
        "diverse_color_map": diverse_color_map,
        "alternative_color": alt_color,
    }
    return plot_id, plot_meta, payload


def generate_single_well_plot(
    time_sec,
    well,
    signal,
    t_half=None,
    submitted_t_half=None,
    include_submitted_marker=True,
    time_unit="hours",
    show_halftime_dot=True,
    baseline_pred=None,
    plateau_pred=None,
    show_baseline_dot=False,
    show_plateau_dot=False,
    custom_titles=None,
):
    time_h = time_axis_from_seconds(time_sec, time_unit)
    y = np.array(signal, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_h, y, linewidth=2.0, alpha=0.95, color="#3B82F6", label=well)

    # Show a subtle "start y" reference at the lowest y-value on the curve.
    if len(time_h) > 0 and len(y) == len(time_h):
        min_idx = int(np.argmin(y))
        min_x = float(time_h[min_idx])
        min_y = float(y[min_idx])
        ax.annotate(
            f"start y: {min_y:.0f}",
            xy=(min_x, min_y),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=8,
            color="#6B7280",
            arrowprops=dict(
                arrowstyle="->",
                color="#9CA3AF",
                lw=0.8,
                shrinkA=0,
                shrinkB=0,
            ),
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc=(1, 1, 1, 0.55),
                ec=(0, 0, 0, 0),
            ),
        )

    # Mark calculated halftime on the curve if available.
    if show_halftime_dot and t_half is not None and len(time_h) > 1 and len(y) == len(time_h):
        t_half_plot = hours_to_unit(t_half, time_unit)
        if time_h[0] <= t_half_plot <= time_h[-1]:
            y_calc = float(np.interp(float(t_half_plot), time_h, y))
            ax.scatter(
                [float(t_half_plot)],
                [y_calc],
                s=70,
                color="#EF4444",
                edgecolors="white",
                linewidths=1.0,
                zorder=5,
                label="Calculated t1/2",
            )

    if show_baseline_dot:
        point = _pick_curve_point_for_level(time_h, y, baseline_pred, prefer_tail=False)
        if point is not None:
            ax.scatter(
                [point["x"]],
                [point["y"]],
                s=70,
                color="#10B981",
                edgecolors="white",
                linewidths=1.0,
                zorder=6,
                label="Predicted baseline",
            )

    if show_plateau_dot:
        point = _pick_curve_point_for_level(time_h, y, plateau_pred, prefer_tail=True)
        if point is not None:
            ax.scatter(
                [point["x"]],
                [point["y"]],
                s=70,
                color="#F59E0B",
                edgecolors="white",
                linewidths=1.0,
                zorder=6,
                label="Predicted plateau",
            )

    # Mark user-submitted halftime on the curve if available.
    if (
        include_submitted_marker
        and submitted_t_half is not None
        and len(time_h) > 1
        and len(y) == len(time_h)
    ):
        submitted_plot = hours_to_unit(submitted_t_half, time_unit)
        if time_h[0] <= submitted_plot <= time_h[-1]:
            y_sub = float(np.interp(float(submitted_plot), time_h, y))
            ax.scatter(
                [float(submitted_plot)],
                [y_sub],
                s=70,
                color="#10B981",
                edgecolors="white",
                linewidths=1.0,
                zorder=6,
                label="Submitted t1/2",
            )

    x_label, y_label, title_label = resolve_plot_titles(
        custom_titles,
        f"Time ({unit_suffix(time_unit)})",
        "Fluorescence (a.u.)",
        f"Aggregation curve - {well}",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title_label)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=9)
    # Keep axis box stable so client-side overlay mapping stays aligned.
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.12, top=0.88)

    # Export exact axis geometry + limits so browser dot maps exactly to the curve.
    ax_pos = ax.get_position()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    plot_meta = {
        "ax_left": float(ax_pos.x0),
        "ax_right": float(ax_pos.x1),
        "ax_bottom": float(ax_pos.y0),
        "ax_top": float(ax_pos.y1),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
    }

    return _store_plot_figure(fig, f"well_{well}"), plot_meta


def generate_sigmoid_control_plot(
    time_sec,
    well,
    signal,
    baseline_pred=None,
    plateau_pred=None,
    submitted_baseline_x=None,
    submitted_plateau_x=None,
    time_unit="hours",
    custom_titles=None,
):
    time_h = time_axis_from_seconds(time_sec, time_unit)
    y = np.array(signal, dtype=float)
    if len(time_h) == 0:
        time_h = np.array([0.0])
        y = np.array([0.0])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_h, y, linewidth=2.0, alpha=0.95, color="#3B82F6", label=well)

    baseline_point = _pick_curve_point_for_level(time_h, y, baseline_pred, prefer_tail=False)
    plateau_point = _pick_curve_point_for_level(time_h, y, plateau_pred, prefer_tail=True)

    if baseline_point is not None:
        ax.scatter(
            [baseline_point["x"]],
            [baseline_point["y"]],
            s=72,
            color="#10B981",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
            label="Predicted baseline",
        )
    if plateau_point is not None:
        ax.scatter(
            [plateau_point["x"]],
            [plateau_point["y"]],
            s=72,
            color="#F59E0B",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
            label="Predicted plateau",
        )

    submitted_baseline_point = None
    submitted_plateau_point = None
    if submitted_baseline_x is not None:
        submitted_baseline_plot = hours_to_unit(float(submitted_baseline_x), time_unit)
        submitted_baseline_y = float(np.interp(float(submitted_baseline_plot), time_h, y))
        submitted_baseline_point = {
            "x": float(submitted_baseline_plot),
            "y": submitted_baseline_y,
        }
        ax.scatter(
            [submitted_baseline_point["x"]],
            [submitted_baseline_point["y"]],
            s=70,
            color="#06B6D4",
            edgecolors="white",
            linewidths=1.0,
            zorder=6,
            label="Submitted baseline",
        )
    if submitted_plateau_x is not None:
        submitted_plateau_plot = hours_to_unit(float(submitted_plateau_x), time_unit)
        submitted_plateau_y = float(np.interp(float(submitted_plateau_plot), time_h, y))
        submitted_plateau_point = {
            "x": float(submitted_plateau_plot),
            "y": submitted_plateau_y,
        }
        ax.scatter(
            [submitted_plateau_point["x"]],
            [submitted_plateau_point["y"]],
            s=70,
            color="#A855F7",
            edgecolors="white",
            linewidths=1.0,
            zorder=6,
            label="Submitted plateau",
        )

    x_label, y_label, title_label = resolve_plot_titles(
        custom_titles,
        f"Time ({unit_suffix(time_unit)})",
        "Fluorescence (a.u.)",
        f"Sigmoidal fitting control - {well}",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title_label)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=9)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.12, top=0.88)

    ax_pos = ax.get_position()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    plot_meta = {
        "ax_left": float(ax_pos.x0),
        "ax_right": float(ax_pos.x1),
        "ax_bottom": float(ax_pos.y0),
        "ax_top": float(ax_pos.y1),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
    }

    return _store_plot_figure(fig, f"sigmoid_{well}"), plot_meta, {
        "baseline_point": baseline_point,
        "plateau_point": plateau_point,
        "submitted_baseline_point": submitted_baseline_point,
        "submitted_plateau_point": submitted_plateau_point,
    }


def generate_group_vs_control_plot(
    time_sec,
    wells_dict,
    control_wells,
    group_wells,
    normalized=False,
    x_from=None,
    x_to=None,
    control_color="#000000",
    group_color="#E69F00",
    time_unit="hours",
    custom_titles=None,
    group_name="",
):
    time_h = time_axis_from_seconds(time_sec, time_unit)

    mask = np.ones_like(time_h, dtype=bool)
    if x_from is not None:
        mask &= time_h >= x_from
    if x_to is not None:
        mask &= time_h <= x_to
    if not np.any(mask):
        raise ValueError("No data points in selected x range.")
    time_h = time_h[mask]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    def _plot_wells(well_list, color, legend_label):
        shown = False
        for well in well_list:
            if well not in wells_dict:
                continue
            y = np.array(wells_dict[well], dtype=float)
            if len(y) != len(mask):
                continue
            y = y[mask]
            if normalized:
                mn, mx = np.min(y), np.max(y)
                if mx - mn == 0:
                    continue
                y = (y - mn) / (mx - mn)
            lbl = legend_label if not shown else None
            ax.plot(time_h, y, linewidth=1.4, alpha=0.8, color=color, label=lbl)
            shown = True

    _plot_wells(control_wells, control_color, "Control")
    _plot_wells(group_wells, group_color, group_name or "Group")

    suffix = unit_suffix(time_unit)
    default_x = f"Time ({suffix})"
    default_y = "Normalized fluorescence (0-1)" if normalized else "Fluorescence (a.u.)"
    default_title = group_name or "Group vs Control"
    x_label, y_label, title_label = resolve_plot_titles(custom_titles, default_x, default_y, default_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title_label)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    prefix = f"gvc_{group_name[:12]}" if group_name else "gvc_group"
    return _store_plot_figure(fig, prefix)
