import argparse
import subprocess
import sys
from pathlib import Path


def beta_token(beta):
    return str(beta).replace(".", "p")


def run_command(cmd, skip_existing_output=None):
    if skip_existing_output is not None and skip_existing_output.exists():
        print(f"Skip existing: {skip_existing_output}")
        return
    print("Running:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def base_command(args, output_dir):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_nsl_kdd_ocsvm_hpo.py"),
        "--output-dir",
        str(output_dir),
        "--n-repeats",
        str(args.n_repeats),
        "--n-trials",
        str(args.n_trials),
        "--grid-size",
        str(args.grid_size),
        "--max-fit-normals",
        str(args.max_fit_normals),
        "--max-val-samples",
        str(args.max_val_samples),
        "--val-size",
        str(args.val_size),
        "--seed",
        str(args.seed),
        "--performance-only",
    ]
    if args.equal_budget_from_grid:
        cmd.append("--equal-budget-from-grid")
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("NSL-KDD/results_sensitivity_binary_f1"))
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--equal-budget-from-grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-fit-normals", type=int, default=2000)
    parser.add_argument("--max-val-samples", type=int, default=6000)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--kernels", nargs="+", default=["rbf", "matern", "rational_quadratic", "linear"])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--run-kernel-sweep", action="store_true")
    parser.add_argument("--run-beta-sweep", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny budget to validate the experiment wiring.",
    )
    args = parser.parse_args()

    if not args.run_kernel_sweep and not args.run_beta_sweep:
        args.run_kernel_sweep = True
        args.run_beta_sweep = True

    if args.smoke:
        args.output_root = args.output_root / "smoke"
        args.n_repeats = 1
        args.grid_size = 2
        args.n_trials = 4
        args.max_fit_normals = min(args.max_fit_normals, 300)
        args.max_val_samples = min(args.max_val_samples, 600)

    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.run_kernel_sweep:
        for kernel in args.kernels:
            output_dir = args.output_root / "kernel_sweep_f1" / f"bo_kernel_{kernel}"
            cmd = base_command(args, output_dir)
            cmd += [
                "--methods",
                "BO",
                "--validation-metric",
                "f_beta",
                "--validation-beta",
                "1.0",
                "--validation-average",
                "binary",
                "--bo-kernel",
                kernel,
            ]
            sentinel = output_dir / "nsl_kdd_ocsvm_summary.csv" if args.skip_existing else None
            run_command(cmd, sentinel)

    if args.run_beta_sweep:
        for beta in args.betas:
            output_dir = args.output_root / "beta_sweep" / f"beta_{beta_token(beta)}"
            cmd = base_command(args, output_dir)
            cmd += [
                "--validation-metric",
                "f_beta",
                "--validation-beta",
                str(beta),
                "--validation-average",
                "binary",
                "--bo-kernel",
                "rbf",
            ]
            sentinel = output_dir / "nsl_kdd_ocsvm_summary.csv" if args.skip_existing else None
            run_command(cmd, sentinel)


if __name__ == "__main__":
    main()
