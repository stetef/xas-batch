"""``xas-batch-view`` — launch the Streamlit fit viewer.

Thin wrapper so users don't have to remember the ``streamlit run <path>`` incantation.
``--db`` is forwarded to the app via the ``XAS_VIEW_DB`` env var; any remaining args
are passed straight through to Streamlit (e.g. ``--server.port 8502``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xas-batch-view",
        description="Launch the interactive XAS fit viewer (Streamlit + Plotly).",
    )
    parser.add_argument("--db", type=Path, default=None,
                        help="catalog .sqlite to open (default: auto-discover from .env / ./out)")
    args, extra = parser.parse_known_args(argv)

    try:
        from streamlit.web import cli as stcli
    except ModuleNotFoundError:
        print("streamlit is not installed. Install the viewer extra:\n"
              "    uv pip install -e '.[viewer]'   (or: pip install streamlit plotly)",
              file=sys.stderr)
        return 2

    if args.db is not None:
        os.environ["XAS_VIEW_DB"] = str(args.db.expanduser())

    app = Path(__file__).with_name("viewer_app.py")
    sys.argv = ["streamlit", "run", str(app), *extra]
    return stcli.main()  # type: ignore[no-any-return]


if __name__ == "__main__":
    raise SystemExit(main())
