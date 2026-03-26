"""Utilities for filtering peptide sequences and creating train/validation splits."""

import argparse
import os
import random
from typing import Dict, List, Tuple


DEFAULT_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def write_sequences(sequences: List[str], filepath: str) -> None:
    """Write one sequence per line."""
    with open(filepath, "w", encoding="utf-8") as file_out:
        for sequence in sequences:
            file_out.write(sequence + "\n")


def load_sequences(
    file_path: str,
    max_sequences: int = None,
    min_length: int = 1,
    max_length: int = 50,
    allowed_amino_acids: str = DEFAULT_AMINO_ACIDS,
    deduplicate: bool = False,
) -> Tuple[List[str], Dict[str, int]]:
    """Load sequences from a text file and apply basic filtering rules."""
    sequences: List[str] = []
    stats = {
        "lines_read": 0,
        "empty_lines": 0,
        "too_short": 0,
        "too_long": 0,
        "invalid_residue": 0,
        "duplicates_removed": 0,
        "accepted": 0,
    }

    allowed_set = set(allowed_amino_acids) if allowed_amino_acids else None
    seen = set()

    with open(file_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            stats["lines_read"] += 1
            sequence = raw_line.strip().upper()

            if not sequence:
                stats["empty_lines"] += 1
                continue

            if len(sequence) < min_length:
                stats["too_short"] += 1
                continue

            if max_length is not None and len(sequence) > max_length:
                stats["too_long"] += 1
                continue

            if allowed_set is not None and any(residue not in allowed_set for residue in sequence):
                stats["invalid_residue"] += 1
                continue

            if deduplicate:
                if sequence in seen:
                    stats["duplicates_removed"] += 1
                    continue
                seen.add(sequence)

            sequences.append(sequence)
            stats["accepted"] += 1

            if max_sequences is not None and len(sequences) >= max_sequences:
                break

    return sequences, stats


def split_sequences(sequences: List[str], validation_split: float, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split sequences into training and validation sets."""
    shuffled_sequences = list(sequences)
    random.Random(seed).shuffle(shuffled_sequences)
    split_idx = int(len(shuffled_sequences) * (1 - validation_split))
    return shuffled_sequences[:split_idx], shuffled_sequences[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="Peptide sequence preprocessing")

    parser.add_argument("--input_file", type=str, required=True, help="Path to the input sequence file, one sequence per line")
    parser.add_argument("--output_dir", type=str, default="./processed_data", help="Output directory")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--max_sequences", type=int, default=None, help="Maximum number of sequences to load")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum allowed sequence length")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum allowed sequence length")
    parser.add_argument(
        "--allowed_amino_acids",
        type=str,
        default=DEFAULT_AMINO_ACIDS,
        help="Allowed residue alphabet. Use an empty string to disable residue filtering.",
    )
    parser.add_argument("--deduplicate", action="store_true", help="Remove duplicate sequences before splitting")

    args = parser.parse_args()

    if not 0 < args.validation_split < 1:
        raise ValueError("--validation_split must be between 0 and 1")

    if args.min_length < 1:
        raise ValueError("--min_length must be at least 1")

    if args.max_length is not None and args.max_length < args.min_length:
        raise ValueError("--max_length must be greater than or equal to --min_length")

    os.makedirs(args.output_dir, exist_ok=True)

    sequences, stats = load_sequences(
        args.input_file,
        max_sequences=args.max_sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        allowed_amino_acids=args.allowed_amino_acids,
        deduplicate=args.deduplicate,
    )

    if not sequences:
        raise ValueError("No valid sequences remained after filtering")

    train_seqs, val_seqs = split_sequences(sequences, args.validation_split, args.seed)

    train_path = os.path.join(args.output_dir, "train.txt")
    val_path = os.path.join(args.output_dir, "validation.txt")
    write_sequences(train_seqs, train_path)
    write_sequences(val_seqs, val_path)

    print("Preprocessing complete.")
    print(f"Input file: {args.input_file}")
    print(f"Lines read: {stats['lines_read']}")
    print(f"Accepted sequences: {stats['accepted']}")
    print(f"Filtered empty lines: {stats['empty_lines']}")
    print(f"Filtered too short: {stats['too_short']}")
    print(f"Filtered too long: {stats['too_long']}")
    print(f"Filtered invalid residues: {stats['invalid_residue']}")
    print(f"Removed duplicates: {stats['duplicates_removed']}")
    print(f"Train split: {len(train_seqs)} -> {train_path}")
    print(f"Validation split: {len(val_seqs)} -> {val_path}")


if __name__ == "__main__":
    main()
