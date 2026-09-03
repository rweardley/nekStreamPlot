#!/usr/bin/env python3
"""
Read a CSV file, apply linear scalings to specified fields, and output a new CSV.

Linear scalings are of the form: y = a*x + b

Usage:
    python rescale_csv.py input.csv --scale field_name a b [--scale field_name2 a2 b2 ...]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply linear scalings to CSV fields and output a new CSV."
    )
    parser.add_argument("input_csv", help="Input CSV file to rescale.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <input>_rescaled.csv).",
    )
    parser.add_argument(
        "--scale",
        nargs=3,
        action="append",
        metavar=("FIELD", "a", "b"),
        help="Apply scaling y = a*x + b to a field. Can be used multiple times.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Read the CSV file
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: File '{args.input_csv}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply scalings if specified
    if args.scale:
        for field_name, a_str, b_str in args.scale:
            if field_name not in df.columns:
                print(
                    f"Warning: Field '{field_name}' not found in CSV. Available fields: {', '.join(df.columns)}",
                    file=sys.stderr,
                )
                continue

            try:
                a = float(a_str)
                b = float(b_str)
            except ValueError:
                print(
                    f"Error: Scaling coefficients for '{field_name}' must be floats, got a={a_str}, b={b_str}",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Apply scaling: y = a*x + b
            df[field_name] = a * df[field_name] + b
            print(f"Applied scaling to '{field_name}': y = {a}*x + {b}")

    # Determine output filename
    if args.output is None:
        input_path = Path(args.input_csv)
        args.output = str(input_path.with_stem(input_path.stem + "_rescaled"))

    # Write output CSV
    try:
        df.to_csv(args.output, index=False)
        print(f"Wrote rescaled CSV to '{args.output}'")
    except Exception as e:
        print(f"Error writing CSV: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
