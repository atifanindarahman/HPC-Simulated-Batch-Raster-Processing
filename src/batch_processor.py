"""
Parallel batch DEM processing pipeline.
It will simulate a HPC job array pattern using Python concurrent.futures.

Usage:
    python src/batch_processor.py --input-dir data/raw/dem_tiles \
                                  --output-dir data/processed \
                                  --workers 4 --merge
"""
import os, time, logging, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_CRS = "EPSG:32613"
OUTPUT_RES = 30  # meters


def process_single_tile(args):
    """Process one DEM tile — simulates one HPC array job."""
    input_path, output_dir = args
    tile_name = Path(input_path).stem
    output_path = Path(output_dir) / f"{tile_name}_processed.tif"
    try:
        logger.info(f"Starting tile: {tile_name}")
        start = time.time()
        with rasterio.open(input_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, TARGET_CRS, src.width, src.height,
                *src.bounds, resolution=OUTPUT_RES)
            kwargs = src.meta.copy()
            kwargs.update({"crs": TARGET_CRS, "transform": transform,
                           "width": width, "height": height, "dtype": "float32"})
            with rasterio.open(output_path, "w", **kwargs) as dst:
                reproject(source=rasterio.band(src, 1),
                          destination=rasterio.band(dst, 1),
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs=TARGET_CRS,
                          resampling=Resampling.bilinear)
        elapsed = time.time() - start
        logger.info(f"Finished {tile_name} in {elapsed:.1f}s")
        return {"tile": tile_name, "status": "success", "time": elapsed}
    except Exception as e:
        logger.error(f"Failed {tile_name}: {e}")
        return {"tile": tile_name, "status": "failed", "error": str(e)}


def merge_tiles(processed_dir, output_path):
    """Merge all processed tiles into a single seamless mosaic."""
    tile_paths = list(Path(processed_dir).glob("*_processed.tif"))
    logger.info(f"Merging {len(tile_paths)} tiles...")
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, out_transform = merge(datasets)
    out_meta = datasets[0].meta.copy()
    out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2],
                     "transform": out_transform})
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    for ds in datasets:
        ds.close()
    logger.info(f"Mosaic saved → {output_path}")


def run_parallel_batch(input_dir, output_dir, n_workers=4):
    """Main batch runner — embarrassingly parallel."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tiles = list(Path(input_dir).glob("*.tif"))
    if not tiles:
        logger.error(f"No .tif files found in {input_dir}")
        return []
    logger.info(f"Found {len(tiles)} tiles · launching {n_workers} workers")
    args = [(str(t), output_dir) for t in tiles]
    results = []
    total_start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_tile, a): a for a in args}
        for future in as_completed(futures):
            results.append(future.result())
    total_time = time.time() - total_start
    success = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Batch complete: {success}/{len(tiles)} tiles | Total: {total_time:.1f}s")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--merge",      action="store_true")
    args = parser.parse_args()
    results = run_parallel_batch(args.input_dir, args.output_dir, args.workers)
    if args.merge:
        merge_tiles(args.output_dir, str(Path(args.output_dir) / "colorado_dem_mosaic.tif"))