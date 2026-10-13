# Segmentation error and boundary-condition tuning in computed coronary FFR

Analysis code for the study "Segmentation Error and Boundary-Condition Tuning in Computed Coronary FFR: A Controlled In Silico Study" (F. Yamin, M. A. Mohammed Sapardi, M. K. Tan, X. Wang, M.-Z. Ismadi; submitted to IEEE Journal of Biomedical and Health Informatics).

The code does seven things:
- inserts idealized stenoses into ImageCAS-X coronary trees;
- applies five segmentation errors: missed side branch (T1), vessel break (T2), lesion length (T3), taper (T4) and throat (T5);
- solves a steady reduced-order FFR model under four outlet boundary-condition protocols: A fixed, B re-derived, C globally tuned, D per-territory tuned;
- uses two microvascular bed structures, leaky and discrete;
- computes reclassifications at FFR 0.80, false-pass rates and the noise floors. A false pass is a model that passes the perfusion check (territory residual below 10%) while its FFR is materially wrong (|ΔFFR| > 0.05). The script outputs label this quantity `passes_and_wrong`;
- tunes every error model to the same noisy perfusion targets as correct anatomy (noise-matched comparison);
- repeats Protocols C and D with finer perfusion territories, and tests tube-model Dice and clDice (including the throat error), the pre-tuning perfusion residual and the fitted scaling as error detectors.

## Data

The code reads the public ImageCAS-X dataset (lumen masks, centerlines and surfaces; Zenodo, DOI 10.5281/zenodo.21887809, CC BY 4.0), which is derived from ImageCAS (Zeng et al., Comput. Med. Imaging Graph. 2023). Download and unzip ImageCAS-X. Its root folder is passed to the scripts as `<root>`. No data are included in this repository.

`cohort/cohort.csv` lists the 150 instances of the study:

| Column | Meaning |
|---|---|
| `scan` | ImageCAS scan number |
| `side` | Coronary tree |
| `vessel` | Host vessel |
| `loc` | Lesion position |
| `c_mm` | Lesion centre, in mm from the vessel origin |
| `L_mm` | Lesion length |
| `ds_pct` | Diameter stenosis, % |
| `band` | Baseline-FFR band |

## Requirements

Python 3.14 with numpy 2.5, pandas 3.0, scipy 1.18, statsmodels 0.15, networkx 3.6, pyvista 0.49 and nibabel 5.4 (see `requirements.txt`).

## Code

| File | Purpose |
|---|---|
| `code/zerod_ffr.py` | Reduced-order network model: Poiseuille elements, expansion loss at the stenosis, leaky and discrete beds |
| `code/imagecasx_loader.py` | Builds a centerline tree with radii from an ImageCAS-X case |
| `code/severity_sweep.py` | Eligibility rules, lesion insertion and cohort selection |
| `code/error_types.py` | Segmentation errors T1–T4 |
| `code/discrete_arm.py` | Eligibility of instances for the discrete bed |
| `code/ablation.py` | Protocols A–C for every instance and error type |
| `code/ablation_per_territory.py` | Protocol D |
| `code/negatives.py` | Simulated physiological noise floor on correct anatomy |
| `code/ablation_demand_scale.py`, `code/demand_replication.py` | Demand sensitivity and replication with re-selected cohorts |
| `code/analyse_ablation.py`, `code/summarise_endpoints.py`, `code/grey_zone.py` | Endpoints, Wilson intervals, McNemar, sign and Wilcoxon tests, gray-zone analysis |
| `code/throat_error.py` | Throat error T5: the lesion re-inserted with its throat narrowed or widened by DS ±10 points or half an in-plane voxel |
| `code/throat_run.py` | Protocols A–D for the throat error, with zero-magnitude and monotonicity checks |
| `code/throat_summary.py` | Throat-error endpoints, gray-zone reclassifications and paired sign tests against T1–T4 |
| `code/territory_granularity.py` | Protocols A–D with main-branch or finer perfusion territories |
| `code/territory_granularity_compare.py` | Row-by-row comparison of a territory configuration with the main run |
| `code/territory_granularity_summary.py` | Endpoints per territory configuration, paired against the main run (McNemar) |
| `code/throat_granularity.py` | Throat error with main-branch or finer perfusion territories |
| `code/throat_granularity_summary.py` | Throat-error endpoints at main-branch and finer territories, paired (McNemar) |
| `code/overlap_metrics.py` | Tube-model Dice, clDice and connected components of each corrupted tree against its clean tree |
| `code/overlap_analysis.py` | Overlap metrics against decision changes: medians and AUCs with instance-bootstrap intervals |
| `code/overlap_metrics_throat.py` | Tube-model Dice, clDice and connected components for the half-voxel throat errors |
| `code/overlap_throat_join.py` | Joins the throat-error overlap metrics to their Protocol A and B outcomes and adds them to the T1–T4 table |
| `code/detector_pretuning.py` | Noise floor run that also records the pre-tuning perfusion residual of each draw |
| `code/detector_analysis.py` | Pre-tuning residual and fitted scaling as error detectors: AUC, sensitivity and false-alarm rate |
| `code/noise_matched_run.py` | Error models and correct anatomy tuned under Protocols C and D to the same 20 noisy target sets per instance |
| `code/noise_matched_analysis.py` | Noise-matched false passes, paired excess over correct anatomy and over Protocol B, detector AUCs; patient-level cluster bootstrap |

## Usage

Run from `code/`, writing outputs to `../results/`:

```
python discrete_arm.py <root> --cohort ../cohort/cohort.csv --out ../results/discrete_arm_eligibility.csv
python ablation.py <root> --cohort ../cohort/cohort.csv --out ../results/ablation.csv
python ablation_per_territory.py <root> --cohort ../cohort/cohort.csv --out ../results/ablation-perterritory.csv
python negatives.py <root> --cohort ../cohort/cohort.csv --draws 20 --out ../results/negatives.csv
python summarise_endpoints.py ../results/ablation.csv ../results/ablation-perterritory.csv ../results/discrete_arm_eligibility.csv
python grey_zone.py ../results/ablation.csv ../results/ablation-perterritory.csv ../results/discrete_arm_eligibility.csv ../results/grey-zone
python analyse_ablation.py ../results/ablation.csv --eligibility ../results/discrete_arm_eligibility.csv --negatives ../results/negatives.csv
python ablation_demand_scale.py 0.7 <root> --cohort ../cohort/cohort.csv --out ../results/ablation-demand0.7.csv
python demand_replication.py 2 <root> ../results/demand-x2
python throat_run.py run <root> --cohort ../cohort/cohort.csv --outdir ../results/throat
python throat_run.py validate <root> --outdir ../results/throat --reference-ablation ../results/ablation.csv --reference-perterritory ../results/ablation-perterritory.csv
python throat_summary.py ../results/throat ../results/ablation.csv ../results/ablation-perterritory.csv ../results/discrete_arm_eligibility.csv
python territory_granularity.py <root> --config finer_keep --cohort ../cohort/cohort.csv --outdir ../results/territory_granularity
python territory_granularity_compare.py ../results/territory_granularity main_drop ../results/ablation.csv ../results/ablation-perterritory.csv
python territory_granularity_summary.py ../results/territory_granularity ../results/ablation.csv ../results/ablation-perterritory.csv ../results/discrete_arm_eligibility.csv
python throat_granularity.py <root> --config main_drop --cohort ../cohort/cohort.csv --outdir ../results/throat_granularity
python throat_granularity.py <root> --config finer_keep --cohort ../cohort/cohort.csv --outdir ../results/throat_granularity
python throat_granularity_summary.py ../results/throat_granularity ../results/discrete_arm_eligibility.csv
python overlap_metrics.py <root> ../cohort/cohort.csv ../results/overlap
python overlap_analysis.py ../results/overlap/overlap_metrics.csv ../results/ablation.csv ../results/discrete_arm_eligibility.csv ../results/overlap
python detector_pretuning.py <root> --cohort ../cohort/cohort.csv --draws 20 --out ../results/detector/negatives_pretune.csv
python detector_analysis.py ../results/ablation.csv ../results/ablation_territory.csv ../results/discrete_arm_eligibility.csv ../results/detector/negatives_pretune.csv ../results/detector
python overlap_metrics_throat.py <root> ../cohort/cohort.csv ../results/overlap_throat
python overlap_throat_join.py ../results/overlap_throat/overlap_metrics.csv ../results/throat/ablation_throat.csv ../results/overlap/overlap_joined.csv ../results/discrete_arm_eligibility.csv ../results/overlap_throat
python noise_matched_run.py <root> ../cohort/cohort.csv ../results/discrete_arm_eligibility.csv ../results/detector/negatives_pretune.csv ../results/noise_matched
python noise_matched_analysis.py ../results/noise_matched/matched_rows.csv ../results/discrete_arm_eligibility.csv ../results/noise_matched
```

`territory_granularity.py` takes four configurations: `main_drop` and `main_keep` (main-branch territories) and `finer_keep` and `finer_drop` (each main-branch territory divided again at its first bifurcation). With `keep`, a territory that loses all its vessels stays in the residual; with `drop`, it is left out.

Add `--limit 2` to a run for a quick check. The random seeds are fixed in the code: 20260918 for cohort selection, 20260919 for the noise floor and 20261009 for the bootstrap intervals.

## License

MIT (see `LICENSE`). If you use this code, please cite the paper above.
