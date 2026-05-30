from __future__ import annotations

import sys
import os
from pathlib import Path

# Support multiple run modes via DOCQA_RUN_MODE env:
# - 'gui' (default): launch the Tkinter chat window
# - 'web' : launch the Flask web UI
# - 'cli' : launch the CLI
run_mode = os.getenv("DOCQA_RUN_MODE", "gui").lower()

if run_mode == "web":
    if __package__:
        from .web_app import run
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from docqa_agent.web_app import run

    if __name__ == "__main__":
        run(host=os.environ.get("DOCQA_WEB_HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 5000)))
else:
    if __package__:
        from .chat_window import main
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from docqa_agent.chat_window import main

    if __name__ == "__main__":
        if run_mode == "cli":
            from .cli import main as cli_main
            cli_main()
        else:
            main()
