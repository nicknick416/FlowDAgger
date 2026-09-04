"""Copy the 30 shortest raw expert episodes into the demonstration folder."""
from __future__ import annotations

import argparse
import json

from arx_campaign import add_config_file_arg, preload_campaign_config
from arx_demo_buffer import copy_demonstrations, select_shortest_demonstrations


def main() -> None:
    cfg = preload_campaign_config()
    parser = argparse.ArgumentParser()
    add_config_file_arg(parser)
    parser.add_argument("--raw-root", default=cfg.raw_data_root)
    parser.add_argument("--output-dir", default=cfg.demo_buffer_dir)
    parser.add_argument("--count", type=int, default=cfg.demo_buffer_count)
    args = parser.parse_args()
    rows = select_shortest_demonstrations(args.raw_root, count=args.count)
    if len(rows) < args.count:
        raise RuntimeError(f"need {args.count} demonstrations, found {len(rows)}")
    output = copy_demonstrations(rows, args.output_dir)
    print(json.dumps({
        "output_dir": str(output),
        "count": len(rows),
        "duration_seconds": [row["duration_seconds"] for row in rows],
        "episodes": [row["episode_id"] for row in rows],
    }, indent=2))


if __name__ == "__main__":
    main()
