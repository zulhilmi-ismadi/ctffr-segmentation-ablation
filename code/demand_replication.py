import sys, subprocess
from pathlib import Path

HERE = Path(__file__).parent
scale, root, outdir = float(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
outdir.mkdir(parents=True, exist_ok=True)
PY = sys.executable

PATCH = (f"import sys; sys.path.insert(0, {str(HERE)!r}); import zerod_ffr; zerod_ffr.K_MURRAY = 562.0 * {scale}; ")

def run(code, log):
    with open(outdir / log, "w") as fh:
        r = subprocess.run([PY, "-c", PATCH + code], stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode: sys.exit(f"stage failed, see {outdir / log}")

run("import severity_sweep as ss; from pathlib import Path; "
    f"ss.__file__ = {str(outdir / 'code' / 'severity_sweep.py')!r}; ss.run(Path({root!r}), 'test')", "1_sweep.log")
(outdir / "results" / "sweep_test.csv").rename(outdir / "sweep_test.csv")
(outdir / "results" / "sweep_test_rejections.csv").rename(outdir / "sweep_test_rejections.csv")
(outdir / "results").rmdir()

run(f"import severity_sweep as ss; ss.select({str(outdir / 'sweep_test.csv')!r})", "2_select.log")
coh = str(outdir / "sweep_test_selected.csv")

run(f"import discrete_arm as d; sys.argv = ['x', {root!r}, '--cohort', {coh!r}, '--out', "
    f"{str(outdir / 'discrete_arm_eligibility.csv')!r}]; d.main()", "3_discrete.log")

run(f"import ablation as A; A.K_MURRAY = zerod_ffr.K_MURRAY; sys.argv = ['x', {root!r}, '--cohort', {coh!r}, "
    f"'--out', {str(outdir / 'ablation.csv')!r}]; A.main()", "4_ablation.log")

r = subprocess.run([PY, str(HERE / "ablation_per_territory.py"), root, "--cohort", coh, "--kscale", str(scale),
                    "--out", str(outdir / "ablation-perterritory.csv")],
                   stdout=open(outdir / "5_perterritory.log", "w"), stderr=subprocess.STDOUT)
print("done" if r.returncode == 0 else "stage 5 failed")
