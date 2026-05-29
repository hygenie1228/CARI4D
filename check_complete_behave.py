#!/usr/bin/env python3

from pathlib import Path


def main() -> None:
    behave_root = Path(__file__).resolve().parent / "experiments" / "behave"

    if not behave_root.exists() or not behave_root.is_dir():
        print(f"[ERROR] Directory not found: {behave_root}")
        return

    sample_dirs = sorted(
        p for p in behave_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )

    if not sample_dirs:
        print(f"[INFO] No sample directories found in: {behave_root}")
        return

    incomplete_samples = []

    for sample_dir in sample_dirs:
        human_params = sample_dir / "human" / "human_params.npz"
        object_params = sample_dir / "object" / "object_params.npz"

        if not (human_params.exists() and object_params.exists()):
            incomplete_samples.append(sample_dir.name)

    total = len(sample_dirs)
    incomplete_count = len(incomplete_samples)
    complete_count = total - incomplete_count

    print(f"Total samples      : {total}")
    print(f"Complete samples   : {complete_count}")
    print(f"Incomplete samples : {incomplete_count}")

    if incomplete_samples:
        print("\nIncomplete sample list:")
        for sample_name in incomplete_samples:
            print(f"- {sample_name}")
    else:
        print("\nAll samples are complete.")


if __name__ == "__main__":
    main()
