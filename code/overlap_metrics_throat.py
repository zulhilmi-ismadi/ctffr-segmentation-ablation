import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import throat_error as TE
import overlap_metrics as OM
import ablation

TYPES = {k: TE.THROAT_TYPES[k] for k in ("T5_vox_narrow", "T5_vox_wide")}
TE.install(TYPES)
OM.insert = TE.patched_insert
OM.ERROR_TYPES = TYPES
OM.CALIBRE_ONLY = ablation.CALIBRE_ONLY
_RUN = OM.run_instance

def run_instance(args):
    TE.DATA_ROOT = Path(args[0])
    TE.reset()
    return _RUN(args)

OM.run_instance = run_instance

if __name__ == "__main__":
    OM.main()
