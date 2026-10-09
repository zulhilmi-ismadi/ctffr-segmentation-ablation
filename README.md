# Segmentation error and boundary-condition tuning in computed coronary FFR

Analysis code for the study "Topological Segmentation Error and Boundary-Condition Tuning in Computed Coronary FFR: A Controlled In Silico Study" (F. Yamin, M. A. Mohammed Sapardi, M. K. Tan, X. Wang, M.-Z. Ismadi; submitted to IEEE Journal of Biomedical and Health Informatics).

The code does five things:
- inserts idealized stenoses into ImageCAS-X coronary trees;
- applies four segmentation errors: missed side branch (T1), vessel break (T2), lesion length (T3) and taper (T4);
- solves a steady reduced-order FFR model under four outlet boundary-condition protocols: A fixed, B re-derived, C globally tuned, D per-territory tuned;
- uses two microvascular bed structures, leaky and discrete;
- computes decision changes at FFR 0.80, passes-and-wrong rates and the noise floors.

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

Python 3.14 with numpy 2.5, pandas 3.0, scipy 1.18, statsmodels 0.15, networkx 3.6 and pyvista 0.49 (see `requirements.txt`).

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
| `code/analyse_ablation.py`, `code/summarise_endpoints.py`, `code/grey_zone.py` | Endpoints, Wilson intervals, McNemar, sign and Wilcoxon tests, grey-zone analysis |

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
```

Add `--limit 2` to a run for a quick check. The random seeds are fixed in the code: 20260918 for cohort selection and 20260919 for the noise floor.

## License

MIT (see `LICENSE`). If you use this code, please cite the paper above.
