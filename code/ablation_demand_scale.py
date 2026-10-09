import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import zerod_ffr
scale = float(sys.argv.pop(1))
zerod_ffr.K_MURRAY = 562.0 * scale
import ablation
ablation.K_MURRAY = zerod_ffr.K_MURRAY
print(f"demand scale {scale}: K_MURRAY = {zerod_ffr.K_MURRAY:.1f} s^-1", flush=True)
ablation.main()
