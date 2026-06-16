"""
Benchmark: serial vs parallel processing time comparison.
Produces a chart showing speedup — include this output in your notebook.

Usage:
    python src/benchmark.py --input-dir data/raw/dem_tiles \
                            --output-dir data/processed/benchmark \
                            --workers 4
"""
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from batch_processor import process_single_tile, run_parallel_batch
import logging

logger = logging.getLogger(__name__)


def run_serial(input_dir, output_dir):
    """Process tiles one at a time — baseline for comparison."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tiles = list(Path(input_dir).glob("*.tif"))
    results = []
    for tile in tiles:
        result = process_single_tile((str(tile), output_dir))
        results.append(result)
    return results


def run_benchmark(input_dir, output_dir, max_workers=4):
    print("=" * 50)
    print(" Serial vs Parallel Benchmark")
    print("=" * 50)

    print(f"\n[1/2] Running serial (1 worker)...")
    start = time.time()
    serial_results = run_serial(input_dir, output_dir + "_serial")
    serial_time = time.time() - start

    print(f"\n[2/2] Running parallel ({max_workers} workers)...")
    start = time.time()
    parallel_results = run_parallel_batch(
        input_dir, output_dir + "_parallel", max_workers
    )
    parallel_time = time.time() - start

    n_tiles = len(serial_results)
    speedup = serial_time / parallel_time if parallel_time > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"  Tiles processed : {n_tiles}")
    print(f"  Serial time     : {serial_time:.1f}s")
    print(f"  Parallel time   : {parallel_time:.1f}s ({max_workers} workers)")
    print(f"  Speedup         : {speedup:.2f}x")
    print(f"{'=' * 50}\n")

    # Plot
    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        ["Serial\n(1 worker)", f"Parallel\n({max_workers} workers)"],
        [serial_time, parallel_time],
        color=["#c0392b", "#27ae60"],
        width=0.45,
        edgecolor="white",
    )

    for bar, val in zip(bars, [serial_time, parallel_time]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}s",
            ha="center", va="bottom", fontsize=12, fontweight="bold"
        )

    ax.annotate(
        f"{speedup:.1f}x faster",
        xy=(1, parallel_time),
        xytext=(0.5, (serial_time + parallel_time) / 1.8),
        fontsize=12, color="#27ae60", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#27ae60"),
        ha="center"
    )

    ax.set_ylabel("Processing time (seconds)", fontsize=11)
    ax.set_title(
        f"Parallel Processing Speedup — {n_tiles} DEM Tiles\n"
        f"concurrent.futures ProcessPoolExecutor",
        fontsize=12
    )
    ax.set_ylim(0, serial_time * 1.3)
    plt.tight_layout()
    plt.savefig("data/outputs/benchmark_results.png", dpi=150)
    print("Chart saved → data/outputs/benchmark_results.png")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers",    type=int, default=4)
    args = parser.parse_args()
    run_benchmark(args.input_dir, args.output_dir, args.workers)