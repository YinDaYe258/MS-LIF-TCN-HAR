from __future__ import annotations

import sys

from run_param_matched_lif import main as run_param_matched_main


if __name__ == "__main__":
    if "--protocols" not in sys.argv:
        sys.argv.extend(["--protocols", "ucihar"])
    run_param_matched_main()
