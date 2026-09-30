#!/usr/bin/env python3
"""
Standalone CAPTCHA reader: base64 in, digit string out.

Give it the base64 ``captchaContent`` the service returned and it prints the
numeric string it reads, and nothing else, so it works by hand or in a pipe:

    python captcha_ocr.py iVBORw0KGgoAAAANSUhEUg...
    python captcha_ocr.py --file captcha.txt
    curl -s -X POST .../request-captcha | python captcha_ocr.py --json

The reading itself is captcha_ocr_lab.OcrEngine: the same PNG container check,
the same 16 preprocessing strategies, the same majority vote. Only the input
handling lives here -- the parts a person at a keyboard needs that the harness
never sees, because the harness gets its base64 straight out of JSON. Keeping
one copy of the voting logic matters: a second copy drifts, and a reader that
disagrees with the harness tells you nothing about the harness.

A read either produces the digits or produces nothing. When no strategy
agrees, stdout stays empty and the reason goes to stderr, so a blank read can
never be mistaken for a wrong one, and nothing is ever invented to fill the
gap.

Exit status
    0   a string was read
    1   no string: the image was damaged, or no strategy agreed
    2   the input itself was unusable (bad base64, no input, both sources)

Usage
-----
    python captcha_ocr.py <base64>
    python captcha_ocr.py --file captcha.txt
    python captcha_ocr.py <base64> --json
    python captcha_ocr.py <base64> --expect-len 4
    python captcha_ocr.py <base64> --save captcha.png
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import sys
from pathlib import Path

# Response text may contain non-ASCII, which crashes a cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# The reader lives next to this file. Failing loudly beats failing with a
# bare ModuleNotFoundError halfway through a paste.
try:
    from captcha_ocr_lab import EXPECT_LEN, OcrEngine, SolveResult
except ModuleNotFoundError as exc:  # pragma: no cover - environment problem
    raise SystemExit(
        "captcha_ocr.py reads with the same engine as the harness, so it needs"
        " captcha_ocr_lab.py in the same directory.\n"
        f"Missing module: {exc}") from exc


# ----------------------------------------------------------------- input
def decode_base64(text: str) -> bytes:
    """
    Turn whatever was pasted into image bytes.

    Handles the three shapes a human actually supplies: the bare base64 the
    API returns, a ``data:image/png;base64,...`` URI copied out of a browser,
    and base64 wrapped over several lines by whatever produced it. Whitespace
    is dropped because it is formatting, not data.

    Anything else raises ValueError with the reason. Guessing here would turn
    a typo into a decode of the wrong bytes and then a misleading container
    error much later, so the input is checked once, up front, and rejected.
    """
    s = (text or "").strip()
    if not s:
        raise ValueError(
            "no input: pass the base64 as an argument, use --file, or pipe it in")

    if s[:5].lower() == "data:":
        if "," not in s:
            raise ValueError(
                "looks like a data URI but has no ',' separating the header "
                "from the base64 payload")
        s = s.split(",", 1)[1]

    s = "".join(s.split())
    if not s:
        raise ValueError("input held only a data-URI header, no base64 payload")

    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"not valid base64: {exc}") from exc
    if not raw:
        raise ValueError("base64 decoded to zero bytes")
    return raw


def read_input(args: argparse.Namespace) -> str:
    """
    Fetch the base64 from the argument, --file, or stdin, in that order.

    Two sources at once is refused rather than silently preferring one: when
    piped input and an argument disagree, taking the argument would hide the
    pipe's data and the operator would never know.
    """
    from_arg = args.base64 if args.base64 is not None else None
    from_file = args.file is not None

    if from_arg and from_file:
        raise ValueError(
            "give the base64 either as an argument or with --file, not both")
    if from_file:
        path = Path(args.file)
        if not path.exists():
            raise ValueError(f"no such file: {path}")
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{path} is not text -- it looks like the PNG itself; pass the "
                f"base64, not the decoded image") from exc
    if from_arg is not None:
        return from_arg
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError(
        "no input: pass the base64 as an argument, use --file, or pipe it in")


# ---------------------------------------------------------------- reading
def read_captcha(b64: str, expect_len: int = EXPECT_LEN) -> SolveResult:
    """
    base64 -> digits, or an empty read with the reason attached.

    The container is validated before the model ever sees the pixels: PIL
    opens damaged PNGs anyway, and a model asked to read corrupt pixel data
    will happily invent digits that look plausible. An intact image that no
    strategy can agree on is reported as no answer rather than as a guess.
    """
    raw = decode_base64(b64)
    return OcrEngine(expect_len=expect_len).solve(raw)


def save_image(b64: str, path: Path) -> None:
    """Write the decoded PNG out, so a wrong read can be looked at."""
    path.write_bytes(decode_base64(b64))


def as_json(res: SolveResult) -> dict:
    return {
        "text": res.text,
        "digits": len(res.text),
        "votes": res.votes,
        "strategies": res.total,
        "agreement": round(res.agreed, 4),
        "confidence": round(res.confidence, 4),
        "valid_container": res.valid_container,
        "problems": res.problems,
    }


# -------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="captcha_ocr.py",
        description="Read a CAPTCHA from base64 and print its digit string.",
        epilog="Prints the digits and nothing else, so it can be piped. "
               "Exit 0 only when a string was read.")
    p.add_argument("base64", nargs="?", default=None,
                   help="the base64 captchaContent; omit to read --file or stdin")
    p.add_argument("-f", "--file", default=None,
                   help="read the base64 from this file instead of the argument")
    p.add_argument("--expect-len", type=int, default=EXPECT_LEN,
                   help=f"how many digits the strategies must agree on "
                        f"(default {EXPECT_LEN})")
    p.add_argument("--json", action="store_true",
                   help="print the full read result as JSON, not just the digits")
    p.add_argument("--save", default=None, metavar="PNG",
                   help="also write the decoded image to this path, to see "
                        "what the reader was given")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        b64 = read_input(args)
        if args.save:
            save_image(b64, Path(args.save))
        res = read_captcha(b64, args.expect_len)
    except ValueError as exc:
        # The input was unusable. Nothing was attempted, so say so and stop.
        print(f"captcha_ocr.py: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"captcha_ocr.py: cannot write {args.save}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        # JSON goes to stdout even on a failed read: the diagnostics are the
        # answer when someone is tuning the preprocessing.
        print(json.dumps(as_json(res), ensure_ascii=False))
        return 0 if res.text else 1

    if not res.text:
        # A blank read prints nothing at all. A wrong number would be sent to
        # the service as if it were real; silence is the honest answer, and
        # stderr carries why.
        for problem in res.problems:
            print(f"captcha_ocr.py: {problem}", file=sys.stderr)
        if not res.problems:
            print(f"captcha_ocr.py: no strategy agreed on {args.expect_len} "
                  f"digits across {res.total} attempts "
                  f"({res.votes} voted for any one reading); not guessing",
                  file=sys.stderr)
        return 1

    print(res.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
