"""
Command-line interface for running HIVDB regression analysis and benchmarks.
"""

import argparse
import os
import warnings

from .DRMcv import DRMcv, run_benchmark
from .data_loading import load_drm_lists_from_csv


def run_paper(args) -> None:
    results_root = "results_for_paper"
    plots_root = "plots_for_paper"
    os.makedirs(results_root, exist_ok=True)
    os.makedirs(plots_root, exist_ok=True)

    mixture_options = [None, 0, 1]
    min_muts_options = [1, 2, 3, 4, 5]

    for mix in mixture_options:
        mix_label = "all" if mix is None else str(mix)
        for mm in min_muts_options:
            subdir = f"mix_{mix_label}_min_{mm}"
            res_dir = os.path.join(results_root, subdir)
            plot_dir = os.path.join(plots_root, subdir)
            os.makedirs(res_dir, exist_ok=True)
            os.makedirs(plot_dir, exist_ok=True)

            drm_lists = load_drm_lists_from_csv(['inputs/default'])
            metadata = {
                'Mixture_limit': mix_label,
                'Min_muts': mm,
                'Paper': True
            }
            print(f"[paper] mixture={mix_label}, min_muts={mm}")
            DRMcv(
                min_muts=mm,
                nfold=args.nfold,
                nrep=args.nrep,
                lars=False,
                drm_lists=drm_lists,
                drugs=args.drugs,
                show_legend=False,
                mixture_limit=mix,
                results_dir=res_dir,
                plots_dir=plot_dir,
                paper_mode=True,
                run_metadata=metadata
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='DRMcv: Cross-validated OLS for HIVDB GenoPheno datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--min_muts', type=int, default=5,
                        help='Minimum number of sequences required to keep a mutation (default: 5).')
    parser.add_argument('--nfold', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--nrep', type=int, default=10,
                        help='Number of CV repetitions (default: 10)')
    parser.add_argument('--no-lars', dest='lars', action='store_false',
                        help='Disable LASSO regression (enabled by default)')
    parser.set_defaults(lars=True)
    parser.add_argument('--drm_lists', nargs='+', default=None,
                        help='Directory path or CSV files with DRM lists (Pos and Mut columns). '
                             'Files must follow pattern: DRM_{drug_or_class}.csv '
                             '(e.g., DRM_3TC.csv, DRM_NRTI.csv). '
                             'Drug-specific lists take precedence over class-wide lists. '
                             'Defaults to inputs/default when omitted.')
    parser.add_argument('--drugs', nargs='+', default=None,
                        help='Specific drugs to analyze (e.g., 3TC ABC LPV DRV)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmarking across default and drm_with_scores DRM sets '
                             'for min_muts values {1,5,10,20} and generate aggregate outputs.')
    parser.add_argument('--paper', action='store_true',
                        help='Generate publication-ready outputs across preset min_muts/mixture grids.')
    parser.add_argument('--no-legend', dest='show_legend', action='store_false',
                        help='Disable legends in coefficient plots.')
    parser.add_argument('--mixtures', type=int, default=None,
                        help='Maximum number of mixture residues to accept (0 excludes mixtures).')
    parser.set_defaults(show_legend=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mixtures is not None and args.mixtures < 0:
        parser.error("--mixtures must be a non-negative integer")

    if args.paper:
        if args.drm_lists:
            warnings.warn("--drm_lists is ignored when --paper is specified.")
        if args.mixtures is not None:
            warnings.warn("--mixtures is ignored when --paper is specified.")
        if args.min_muts != 5:
            warnings.warn("--min_muts is ignored when --paper is specified.")
        run_paper(args)
        return

    if args.benchmark:
        if args.drm_lists:
            warnings.warn("--drm_lists is ignored when --benchmark is specified.")
        run_benchmark(
            lars=args.lars,
            nfold=args.nfold,
            nrep=args.nrep,
            drugs=args.drugs,
            min_muts_grid=[1, 5, 10, 20],
            show_legend=args.show_legend,
            mixture_limit=args.mixtures
        )
        return

    custom_drm_lists = None
    if args.drm_lists:
        custom_drm_lists = load_drm_lists_from_csv(args.drm_lists)

    DRMcv(
        min_muts=args.min_muts,
        nfold=args.nfold,
        nrep=args.nrep,
        lars=args.lars,
        drm_lists=custom_drm_lists,
        drugs=args.drugs,
        show_legend=args.show_legend,
        mixture_limit=args.mixtures
    )


if __name__ == "__main__":
    main()
