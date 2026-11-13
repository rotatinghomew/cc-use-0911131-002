"""命令行入口：

    python3 -m brand_ladder_observatory.segment_release fit \
        --input <JSON> --database <DB> --output <JSON>
    python3 -m brand_ladder_observatory.segment_release score \
        --input <JSON> --database <DB> --release <发布编号> --output <JSON>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from brand_ladder_observatory.segment_release.pipeline import (
    SegmentReleaseError,
    fit_release,
    score_records,
)
from brand_ladder_observatory.segment_release.storage import SegmentStore


def _load_input(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_output(path: str | None, result: dict[str, Any]) -> None:
    text = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    if path:
        Path(path).write_text(text + "\n", encoding="utf-8")
    print(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="经营分群发布与沿用工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="拟合并冻结一个分群发布版")
    fit_parser.add_argument("--input", required=True)
    fit_parser.add_argument("--database", required=True)
    fit_parser.add_argument("--output", required=True)

    score_parser = subparsers.add_parser("score", help="用已冻结发布版给新商户归群")
    score_parser.add_argument("--input", required=True)
    score_parser.add_argument("--database", required=True)
    score_parser.add_argument("--release", required=True, dest="release_id")
    score_parser.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    payload = _load_input(args.input)
    store = SegmentStore(args.database)

    try:
        if args.command == "fit":
            result = fit_release(payload, store=store)
        else:
            result = score_records(payload, args.release_id, store=store)
    except (SegmentReleaseError, KeyError) as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - parser.error 已退出

    _write_output(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
