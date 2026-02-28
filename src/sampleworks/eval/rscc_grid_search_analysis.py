import marimo


__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # RSCC Analysis for Grid Search Results

    This notebook calculates the Real Space Correlation Coefficient (RSCC) between computed maps
    from refined structures and reference (ground truth) maps for all experiments in the grid
    search results.

    ## Workflow:
    1. Scan the `grid_search_results` directory for completed experiments
    2. For each experiment with a `refined.cif`, compute the electron density map
    3. Compare against the corresponding base map and calculate RSCC
    4. Aggregate and visualize results by ensemble size, guidance weight, and scaler type
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import copy
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch

    return Path, copy, np, pd, plt, sns, torch


@app.cell
def _():
    from importlib.resources import files

    from atomworks.io.parser import parse
    from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (
        XMap,
    )
    from sampleworks.eval.constants import DEFAULT_SELECTION_PADDING, OCCUPANCY_LEVELS
    from sampleworks.eval.eval_dataclasses import ProteinConfig
    from sampleworks.eval.grid_search_eval_utils import scan_grid_search_results
    from sampleworks.eval.metrics import rscc
    from sampleworks.eval.structure_utils import (
        get_asym_unit_from_structure,
        get_reference_structure_coords,
    )
    from sampleworks.utils.density_utils import compute_density_from_atomarray
    from sampleworks.utils.guidance_constants import GuidanceType

    DEFAULT_PROTEIN_CONFIGS_CSV = files("sampleworks.data") / "protein_configs.csv"

    return (
        DEFAULT_PROTEIN_CONFIGS_CSV,
        DEFAULT_SELECTION_PADDING,
        GuidanceType,
        OCCUPANCY_LEVELS,
        ProteinConfig,
        XMap,
        compute_density_from_atomarray,
        get_asym_unit_from_structure,
        get_reference_structure_coords,
        parse,
        rscc,
        scan_grid_search_results,
    )


@app.cell
def _(DEFAULT_PROTEIN_CONFIGS_CSV, Path, ProteinConfig):
    WORKSPACE_ROOT = Path("/home/kchrispens/sampleworks")
    GRID_SEARCH_DIR = WORKSPACE_ROOT / "grid_search_results"

    protein_configs = ProteinConfig.from_csv(WORKSPACE_ROOT, DEFAULT_PROTEIN_CONFIGS_CSV)

    print(f"Grid search directory: {GRID_SEARCH_DIR}")
    print(f"Proteins configured: {list(protein_configs.keys())}")
    return GRID_SEARCH_DIR, WORKSPACE_ROOT, protein_configs


@app.cell
def _(GRID_SEARCH_DIR, scan_grid_search_results):
    all_experiments = scan_grid_search_results(GRID_SEARCH_DIR)
    print(f"Found {len(all_experiments)} experiments with refined.cif files")

    if all_experiments:
        all_experiments.summarize()
    return (all_experiments,)


@app.cell
def _(get_reference_structure_coords, protein_configs):
    ref_coords = {}
    for _protein_key, _protein_config in protein_configs.items():
        _protein_ref_coords = get_reference_structure_coords(_protein_config, _protein_key)
        if _protein_ref_coords is not None:
            ref_coords[_protein_key] = _protein_ref_coords

    print(f"Loaded reference coordinates for {len(ref_coords)} proteins")
    return (ref_coords,)


@app.cell
def _(
    DEFAULT_SELECTION_PADDING,
    all_experiments,
    compute_density_from_atomarray,
    copy,
    get_asym_unit_from_structure,
    np,
    parse,
    protein_configs,
    ref_coords,
    rscc,
    torch,
):
    print("Calculating RSCC values for all experiments...")
    print("Note: RSCC is computed on the region around altloc residues (defined by selection)")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_device}")

    results = []
    _base_map_cache = {}

    for _i, _exp in enumerate(all_experiments):
        if _exp.protein not in protein_configs:
            print(f"Skipping protein with no configuration: {_exp.protein}")
            continue

        _protein_config = protein_configs[_exp.protein]

        if _exp.protein not in ref_coords:
            print(
                f"Skipping {_exp.protein_dir_name}: no reference structure available "
                f"for {_exp.protein}"
            )
            continue

        _selection_coords = ref_coords[_exp.protein]
        _base_map_path = _protein_config.get_base_map_path_for_occupancy(_exp.occ_a)
        if _base_map_path is None:
            print(
                f"Skipping {_exp.protein_dir_name}: base map for occupancy {_exp.occ_a} not found"
            )
            continue

        try:
            if (_exp.protein, _exp.occ_a) not in _base_map_cache:
                _base_xmap = _protein_config.load_map(_base_map_path)
                if _base_xmap is None:
                    raise ValueError(f"Failed to load base map from {_base_map_path}")
                _extracted_base = _base_xmap.extract(
                    _selection_coords, padding=DEFAULT_SELECTION_PADDING
                )
                _base_map_cache[(_exp.protein, _exp.occ_a)] = (_base_xmap, _extracted_base)
            else:
                _base_xmap, _extracted_base = _base_map_cache[(_exp.protein, _exp.occ_a)]

            if _extracted_base is None or _extracted_base.array.size == 0:
                raise ValueError(f"Extracted base map from {_base_map_path} is empty")

            _structure = parse(_exp.refined_cif_path, ccd_mirror_path=None)
            _atom_array = get_asym_unit_from_structure(_structure)
            _computed_density, _ = compute_density_from_atomarray(
                _atom_array, xmap=_base_xmap, em_mode=False, device=_device
            )

            _computed_xmap = copy.deepcopy(_base_xmap)
            _computed_xmap.array = _computed_density.cpu().numpy().squeeze()
            _extracted_computed = _computed_xmap.extract(
                _selection_coords, padding=DEFAULT_SELECTION_PADDING
            )

            if _extracted_computed is None or _extracted_computed.array.size == 0:
                raise ValueError("Extracted computed map is empty")

            _exp.rscc = rscc(_extracted_base.array, _extracted_computed.array)
            _exp.base_map_path = _base_map_path

        except Exception as _e:
            import traceback

            print(f"ERROR processing {_exp.exp_dir}: {_e}")
            print(f"  Traceback: {traceback.format_exc()}")
            _exp.error = _e
            _exp.rscc = np.nan
            _exp.base_map_path = _base_map_path

        results.append(_exp)
        if (_i + 1) % 10 == 0 or _i == 0:
            print(
                f"  [{_i + 1}/{len(all_experiments)}] {_exp.protein_dir_name} / "
                f"{_exp.model} / {_exp.scaler} / ens{_exp.ensemble_size}_"
                f"gw{_exp.guidance_weight}: RSCC = {_exp.rscc:.4f}"
            )

    print(f"\nCompleted RSCC calculation for {len(results)} experiments")
    return (results,)


@app.cell
def _(pd, results):
    df = pd.DataFrame([r.__dict__ for r in results])

    drop_cols = [
        "exp_dir",
        "refined_cif_path",
        "base_map_path",
        "error",
        "protein_dir_name",
    ]

    print("Results Summary:")
    print(df.drop(drop_cols, axis=1, errors="ignore").head(20).to_string())

    print("\n\nSummary Statistics by Protein and Scaler:")
    summary = (
        df.groupby(["protein", "scaler"])["rscc"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
    )
    print(summary)
    return (df,)


@app.cell
def _(OCCUPANCY_LEVELS, pd, protein_configs, ref_coords, rscc):
    print("Calculating correlations between base maps and pure conformer maps...")
    print("This shows how well single conformers explain occupancy-mixed data")

    base_pure_correlations = []

    for _protein_key, _protein_config in protein_configs.items():
        if _protein_key not in ref_coords:
            print(f"Skipping {_protein_key}: no reference coordinates available")
            continue

        _selection_coords = ref_coords[_protein_key]

        _map_path_1occA = _protein_config.get_base_map_path_for_occupancy(1.0)
        _map_path_1occB = _protein_config.get_base_map_path_for_occupancy(0.0)
        if _map_path_1occA is None or _map_path_1occB is None:
            print(f"Skipping {_protein_key}: pure conformer maps not found")
            continue

        print(f"\nProcessing {_protein_key} single conformer explanatory power:")
        print(f"  Pure A reference: {_map_path_1occA.name}")
        print(f"  Pure B reference: {_map_path_1occB.name}")

        try:
            _extracted_pure_A = _protein_config.load_map(
                _map_path_1occA, selection_coords=_selection_coords
            )
            _extracted_pure_B = _protein_config.load_map(
                _map_path_1occB, selection_coords=_selection_coords
            )

            for _occ_a in OCCUPANCY_LEVELS:
                try:
                    _base_map_path = _protein_config.get_base_map_path_for_occupancy(_occ_a)
                    if _base_map_path is None:
                        continue

                    print(f"  Processing occ_A={_occ_a}: {_base_map_path.name}")

                    _extracted_base = _protein_config.load_map(
                        _base_map_path, selection_coords=_selection_coords
                    )

                    if (
                        _extracted_base is None
                        or _extracted_pure_A is None
                        or _extracted_pure_B is None
                    ):
                        raise ValueError("One of the extracted maps is empty")

                    _corr_base_vs_pureA = rscc(_extracted_base.array, _extracted_pure_A.array)
                    _corr_base_vs_pureB = rscc(_extracted_base.array, _extracted_pure_B.array)

                    base_pure_correlations.append(
                        {
                            "protein": _protein_key,
                            "occ_a": _occ_a,
                            "base_vs_1occA": _corr_base_vs_pureA,
                            "base_vs_1occB": _corr_base_vs_pureB,
                        }
                    )

                    print(f"    Base map vs pure A: {_corr_base_vs_pureA:.4f}")
                    print(f"    Base map vs pure B: {_corr_base_vs_pureB:.4f}")

                except Exception as _e:
                    import traceback

                    print(f"  Error processing occ_A={_occ_a} for {_protein_key}: {_e}")
                    print(f"  Traceback: {traceback.format_exc()}")

        except Exception as _e:
            import traceback

            print(f"Error calculating correlations for {_protein_key}: {_e}")
            print(f"  Traceback: {traceback.format_exc()}")

    df_base_vs_pure = pd.DataFrame(base_pure_correlations)
    print(
        f"\nCalculated single conformer explanatory power for "
        f"{len(df_base_vs_pure)} occupancy points"
    )

    if not df_base_vs_pure.empty:
        print("\nSummary by protein:")
        for _protein in df_base_vs_pure["protein"].unique():
            _protein_data = df_base_vs_pure[df_base_vs_pure["protein"] == _protein].sort_values(
                "occ_a"
            )
            print(f"\n{_protein}:")
            for _, _row in _protein_data.iterrows():
                print(
                    f"  occ_A={_row['occ_a']:.2f}: "
                    f"vs_pureA={_row['base_vs_1occA']:.4f}, "
                    f"vs_pureB={_row['base_vs_1occB']:.4f}"
                )
    return (df_base_vs_pure,)


@app.cell
def _(df, plt, sns, GuidanceType):
    # Visualization: RSCC by ensemble size and guidance weight
    if df.empty or df["rscc"].isna().all():
        print("No valid RSCC values to plot")
    else:
        _plot_df = df.dropna(subset=["rscc", "ensemble_size", "guidance_weight"])

        if _plot_df.empty:
            print("No valid data for plotting")
        else:
            # Set up the plotting style
            sns.set_theme(context="poster", style="whitegrid")

            # Plot 1: RSCC vs ensemble size, faceted by scaler
            _fig1, _axes1 = plt.subplots(1, 2, figsize=(14, 5))

            for _idx, _scaler in enumerate([GuidanceType.PURE_GUIDANCE, GuidanceType.FK_STEERING]):
                _ax = _axes1[_idx]
                _scaler_df = _plot_df[_plot_df["scaler"] == _scaler]

                if not _scaler_df.empty:
                    for _gw in sorted(_scaler_df["guidance_weight"].unique()):
                        _gw_df = _scaler_df[_scaler_df["guidance_weight"] == _gw]
                        _agg = (
                            _gw_df.groupby("ensemble_size")["rscc"]
                            .agg(["mean", "std"])
                            .reset_index()
                        )
                        _ax.errorbar(
                            _agg["ensemble_size"],
                            _agg["mean"],
                            yerr=_agg["std"],
                            marker="o",
                            label=f"gw={_gw}",
                            capsize=3,
                        )

                _ax.set_xlabel("Ensemble Size", fontsize=12)
                _ax.set_ylabel("RSCC", fontsize=12)
                _ax.set_title(f"{_scaler.value.replace('_', ' ').title()}", fontsize=14)
                _ax.legend()
                _ax.set_xticks([1, 2, 4, 8])

            plt.tight_layout()
            plt.show()

            # Plot 2: Heatmap of RSCC by ensemble size and guidance weight for each scaler
            _fig2, _axes2 = plt.subplots(1, 2, figsize=(14, 5))

            for _idx, _scaler in enumerate([GuidanceType.PURE_GUIDANCE, GuidanceType.FK_STEERING]):
                _ax = _axes2[_idx]
                _scaler_df = _plot_df[_plot_df["scaler"] == _scaler]

                if not _scaler_df.empty:
                    _pivot = _scaler_df.pivot_table(
                        values="rscc",
                        index="ensemble_size",
                        columns="guidance_weight",
                        aggfunc="mean",
                    )

                    sns.heatmap(
                        _pivot,
                        annot=True,
                        fmt=".3f",
                        cmap="RdYlGn",
                        vmin=0,
                        vmax=1,
                        ax=_ax,
                    )
                    _ax.set_title(f"{_scaler.value.replace('_', ' ').title()}", fontsize=14)
                    _ax.set_xlabel("Guidance Weight", fontsize=12)
                    _ax.set_ylabel("Ensemble Size", fontsize=12)

            plt.tight_layout()
            plt.show()
    return


@app.cell
def _(df, plt):
    # Visualization: RSCC by protein and occupancy
    if df.empty or df["rscc"].isna().all():
        print("No valid RSCC values to plot")
    else:
        _plot_df = df.dropna(subset=["rscc", "occ_a"])

        if _plot_df.empty:
            print("No valid data for plotting")
        else:
            # Get unique proteins
            _proteins = sorted(_plot_df["protein"].unique())
            _n_proteins = len(_proteins)

            _fig, _axes = plt.subplots(1, _n_proteins, figsize=(5 * _n_proteins, 5), squeeze=False)
            _axes = _axes.flatten()

            for _idx, _protein in enumerate(_proteins):
                _ax = _axes[_idx]
                _protein_df = _plot_df[_plot_df["protein"] == _protein]

                for _scaler in _protein_df["scaler"].unique():
                    _scaler_df = _protein_df[_protein_df["scaler"] == _scaler]
                    _agg = _scaler_df.groupby("occ_a")["rscc"].agg(["mean", "std"]).reset_index()

                    _ax.errorbar(
                        _agg["occ_a"],
                        _agg["mean"],
                        yerr=_agg["std"],
                        marker="o",
                        label=_scaler.replace("_", " ").title(),
                        capsize=3,
                    )

                _ax.set_xlabel("Conformer A Occupancy", fontsize=12)
                _ax.set_ylabel("RSCC", fontsize=12)
                _ax.set_title(f"{_protein.upper()}", fontsize=14)
                _ax.set_xlim(-0.05, 1.05)
                _ax.set_ylim(0, 1.05)
                _ax.legend()
                _ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])

            plt.tight_layout()
            plt.show()
    return


@app.cell
def _(df, np, plt, sns):
    # Visualization: Compare models (boltz2 vs protenix)
    if df.empty or df["rscc"].isna().all():
        print("No valid RSCC values to plot")
    else:
        _plot_df = df.dropna(subset=["rscc"])

        if _plot_df.empty:
            print("No valid data for plotting")
        else:
            _models = sorted(_plot_df["model"].unique())

            if len(_models) > 1:
                _fig, _ax = plt.subplots(figsize=(10, 6))

                _agg = (
                    _plot_df.groupby(["model", "scaler"])["rscc"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )

                _x_pos = np.arange(len(_agg))
                _labels = [f"{_row['model']}\n{_row['scaler']}" for _, _row in _agg.iterrows()]

                _colors = sns.color_palette("husl", len(_agg))
                _bars = _ax.bar(_x_pos, _agg["mean"], yerr=_agg["std"], capsize=5, color=_colors)

                _ax.set_xticks(_x_pos)
                _ax.set_xticklabels(_labels, rotation=45, ha="right")
                _ax.set_ylabel("RSCC", fontsize=12)
                _ax.set_title("RSCC by Model and Scaler", fontsize=14)
                _ax.set_ylim(0, 1.05)

                # Add count labels
                for _bar, _count in zip(_bars, _agg["count"]):
                    _ax.text(
                        _bar.get_x() + _bar.get_width() / 2,
                        _bar.get_height() + 0.02,
                        f"n={_count}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

                plt.tight_layout()
                plt.show()
            else:
                print(f"Only one model found: {_models}")
    return


@app.cell
def _(df, df_base_vs_pure, plt, sns):
    # Visualization: Guidance RSCC vs Single Conformer
    print("\nPlotting guidance effectiveness vs single conformer explanatory power...")

    if df.empty or df["rscc"].isna().all():
        print("No valid guidance RSCC data to plot")
    elif df_base_vs_pure.empty:
        print("No pure conformer correlation data to plot")
    else:
        # Aggregate guidance RSCC by protein and occupancy
        _plot_df = df.dropna(subset=["rscc", "occ_a"])
        _agg_guidance = (
            _plot_df.groupby(["protein", "occ_a"], as_index=False)
            .agg(
                rscc_mean=("rscc", "mean"),
                rscc_std=("rscc", "std"),
                n=("rscc", "size"),
            )
            .sort_values(["protein", "occ_a"])
        )

        # Get unique proteins that have both guidance and pure correlation data
        _proteins_guidance = set(_agg_guidance["protein"].unique())
        _proteins_pure = set(df_base_vs_pure["protein"].unique())
        _proteins = sorted(_proteins_guidance & _proteins_pure)

        if not _proteins:
            print("No proteins with both guidance and pure correlation data")
        else:
            # Set plotting style
            sns.set_theme(context="paper", style="whitegrid")

            # Define colors
            _colors = {
                "guidance": "#1f77b4",
                "pure_A": "#ff7f0e",
                "pure_B": "#2ca02c",
            }

            # Create one plot per protein
            for _protein in _proteins:
                _fig, _ax = plt.subplots(figsize=(10, 6))

                # Plot guidance RSCC
                _protein_guidance = _agg_guidance[_agg_guidance["protein"] == _protein].sort_values(
                    "occ_a"
                )

                if len(_protein_guidance) > 0:
                    _ax.plot(
                        _protein_guidance["occ_a"],
                        _protein_guidance["rscc_mean"],
                        color=_colors["guidance"],
                        marker="o",
                        linestyle="-",
                        markersize=8,
                        linewidth=2,
                        label="Guided Ensemble Map",
                    )

                    # Add error bars if available
                    _has_error = (_protein_guidance["n"] > 1) & ~_protein_guidance[
                        "rscc_std"
                    ].isna()
                    if _has_error.any():
                        _error_sub = _protein_guidance[_has_error]
                        _ax.errorbar(
                            _error_sub["occ_a"],
                            _error_sub["rscc_mean"],
                            yerr=_error_sub["rscc_std"],
                            fmt="none",
                            color=_colors["guidance"],
                            alpha=0.5,
                            capsize=3,
                        )

                # Plot pure conformer correlations
                _protein_pure = df_base_vs_pure[df_base_vs_pure["protein"] == _protein].sort_values(
                    "occ_a"
                )

                if len(_protein_pure) > 0:
                    _ax.plot(
                        _protein_pure["occ_a"],
                        _protein_pure["base_vs_1occA"],
                        color=_colors["pure_A"],
                        marker="s",
                        linestyle="-",
                        markersize=8,
                        linewidth=2,
                        label="Conformer A Map",
                    )

                    _ax.plot(
                        _protein_pure["occ_a"],
                        _protein_pure["base_vs_1occB"],
                        color=_colors["pure_B"],
                        marker="^",
                        linestyle="-",
                        markersize=8,
                        linewidth=2,
                        label="Conformer B Map",
                    )

                # Formatting
                _ax.set_xlabel("Conformer A Occupancy", fontsize=12, fontweight="bold")
                _ax.set_ylabel("RSCC", fontsize=12, fontweight="bold")
                _ax.set_title(
                    f"{_protein.upper()} - Guidance vs Single Conformer RSCC",
                    fontsize=14,
                    fontweight="bold",
                )
                _ax.set_xlim(-0.05, 1.05)
                _ax.set_ylim(0.0, 1.05)
                _ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                _ax.set_xticklabels(
                    [
                        "0.0\n(pure B)",
                        "0.25",
                        "0.5\n(equal mix)",
                        "0.75",
                        "1.0\n(pure A)",
                    ]
                )
                _ax.legend(
                    fontsize=10,
                    loc="best",
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                )
                _ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

            print(f"Plotted guidance vs pure conformer comparisons for {len(_proteins)} proteins")
    return


@app.cell
def _(GRID_SEARCH_DIR, df, df_base_vs_pure, pd):
    _output_path = GRID_SEARCH_DIR / "rscc_results.csv"

    _export_cols = [
        "protein",
        "occ_a",
        "model",
        "method",
        "scaler",
        "ensemble_size",
        "guidance_weight",
        "gd_steps",
        "rscc",
    ]
    _export_df = df[[c for c in _export_cols if c in df.columns]]

    if not df_base_vs_pure.empty:
        _export_df = pd.merge(
            _export_df,
            df_base_vs_pure[["protein", "occ_a", "base_vs_1occA", "base_vs_1occB"]],
            on=["protein", "occ_a"],
            how="left",
        )
        print("Added base vs pure conformer correlation columns to export")

    _export_df.to_csv(_output_path, index=False)
    print(f"Results exported to: {_output_path}")
    print(f"Exported {len(_export_df)} rows with {len(_export_df.columns)} columns")
    return


if __name__ == "__main__":
    app.run()
