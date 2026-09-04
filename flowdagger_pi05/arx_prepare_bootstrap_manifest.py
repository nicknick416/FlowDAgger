"""Deprecated helper. Online FlowDAgger no longer needs a reviewed manifest."""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "bootstrap manifest is no longer required. "
        "Start ./run_arx_flowdagger_server.sh and collect assisted-success episodes."
    )


if __name__ == "__main__":
    main()
