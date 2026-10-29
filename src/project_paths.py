from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SKAB_ROOT = PACKAGE_ROOT / 'SKAB'
DATA_ROOT = Path(os.environ.get('SKAB_DATA_ROOT', str(SKAB_ROOT / 'data'))).expanduser().resolve()
OUTPUTS_ROOT = Path(os.environ.get('PUMP_OUTPUT_ROOT', str(PACKAGE_ROOT / 'outputs'))).expanduser().resolve()
