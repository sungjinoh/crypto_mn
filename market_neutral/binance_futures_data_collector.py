#!/usr/bin/env python3
"""
Binance PyArrow Compatible Collector

Uses PyArrow for Parquet operations only (compatible with older PyArrow versions)
- Pandas for CSV reading (universal compatibility)
- PyArrow for optimized Parquet I/O
- Proper large integer handling
- Better compression and performance

Author: Data Collection Script
Date: 2025-08-09
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import zipfile
from datetime import datetime, timedelta
import logging
from multiprocess import Pool, cpu_count, current_process
import requests
from tqdm import tqdm
import warnings
import ccxt
import time
import os

warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging with process info
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_top_usdt_futures_symbols(limit=10):
    """Get top USDT futures symbols by 24h volume from Binance"""
    try:
        exchange = ccxt.binance(
            {
                "sandbox": False,
                "options": {"defaultType": "future"},
            }
        )

        print(f"🔍 Fetching top {limit} USDT futures symbols by volume...")
        markets = exchange.load_markets()

        usdt_futures = {
            symbol: market
            for symbol, market in markets.items()
            if (
                market["type"] == "swap"
                and market["active"]
                and market.get("subType") == "linear"
                and symbol.endswith("USDT:USDT")
            )
        }

        print(f"📊 Found {len(usdt_futures)} active USDT perpetual swaps")

        if not usdt_futures:
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

        symbols_list = list(usdt_futures.keys())
        tickers = exchange.fetch_tickers(symbols_list)

        volume_data = []
        for symbol, ticker in tickers.items():
            if ticker.get("quoteVolume") is not None:
                clean_symbol = symbol.split("/")[0] + "USDT"
                volume_data.append((clean_symbol, ticker["quoteVolume"]))

        volume_data.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, volume in volume_data[:limit]]

        print(f"✅ Top {len(top_symbols)} symbols: {top_symbols}")
        return top_symbols

    except Exception as e:
        print(f"❌ Error getting symbols: {e}")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def process_single_file_pyarrow_compatible(task_data: Tuple) -> Dict:
    """
    Process a single file with PyArrow Parquet optimization (compatible version)
    """
    url, symbol, date_str, interval, file_type, output_dir, task_id = task_data

    process = current_process()
    worker_id = process.name

    result = {
        "task_id": task_id,
        "symbol": symbol,
        "date_str": date_str,
        "file_type": file_type,
        "interval": interval,
        "worker_id": worker_id,
        "success": False,
        "error": None,
        "steps_completed": [],
        "file_size_original": 0,
        "file_size_parquet": 0,
        "rows_original": 0,
        "rows_cleaned": 0,
        "processing_time": 0,
        "compression_ratio": 0,
    }

    start_time = time.time()

    try:
        logger.info(
            f"🚀 [{worker_id}] Starting PyArrow task {task_id}: {symbol}-{file_type}-{date_str}"
        )

        # Step 1: Download ZIP file
        logger.info(f"🌐 [{worker_id}] Step 1/5: Downloading")

        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            result["error"] = f"Download failed - Status: {response.status_code}"
            return result

        result["file_size_original"] = len(response.content)
        result["steps_completed"].append("download")
        logger.info(
            f"✅ [{worker_id}] Downloaded: {result['file_size_original']:,} bytes"
        )

        # Step 2: Save and extract ZIP
        logger.info(f"📦 [{worker_id}] Step 2/5: Extract ZIP")

        temp_dir = Path(f"temp_data/{worker_id}/{symbol}")
        temp_dir.mkdir(parents=True, exist_ok=True)

        if file_type == "klines":
            zip_filename = f"{symbol}-{interval}-{date_str}.zip"
        else:
            zip_filename = f"{symbol}-{file_type}-{date_str}.zip"

        zip_path = temp_dir / zip_filename

        with open(zip_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_files = zf.namelist()
            if not csv_files:
                result["error"] = "No files found in ZIP"
                return result

            csv_filename = csv_files[0]
            zf.extract(csv_filename, temp_dir)
            csv_path = temp_dir / csv_filename

        result["steps_completed"].append("extract")
        logger.info(f"✅ [{worker_id}] Extracted: {csv_filename}")

        # Step 3: Process CSV with pandas (compatible with all PyArrow versions)
        logger.info(f"🔧 [{worker_id}] Step 3/5: Processing CSV data")

        if file_type == "klines":
            df = pd.read_csv(csv_path, header=None)

            if len(df.columns) < 12:
                result["error"] = (
                    f"Invalid klines data - expected 12 columns, got {len(df.columns)}"
                )
                return result

            df.columns = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_volume",
                "taker_buy_quote_volume",
                "ignore",
            ]

            result["rows_original"] = len(df)

            # Convert data types for PyArrow compatibility
            df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
            df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")

            numeric_cols = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trades",
                "taker_buy_volume",
                "taker_buy_quote_volume",
                "ignore",
            ]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Clean data
            df = df.dropna(
                subset=["open_time", "close_time", "open", "high", "low", "close"]
            )
            df["interval"] = interval

        elif file_type == "trades":
            df = pd.read_csv(csv_path, header=None)
            df.columns = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"]

            result["rows_original"] = len(df)

            # Convert data types properly for PyArrow (keep large integers as int64)
            df["id"] = pd.to_numeric(df["id"], errors="coerce")
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
            df["quote_qty"] = pd.to_numeric(df["quote_qty"], errors="coerce")
            df["time"] = pd.to_numeric(df["time"], errors="coerce")
            df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)

            # Clean data
            df = df.dropna(subset=["id", "price", "qty", "time"])

            logger.info(
                f"🔧 [{worker_id}] Trades data: {len(df):,} rows with int64 types"
            )

        elif file_type == "fundingRate":
            df = pd.read_csv(csv_path, header=None)
            df.columns = ["calc_time", "funding_interval_hours", "fundingRate"]

            result["rows_original"] = len(df)

            # Convert data types
            df["calc_time"] = pd.to_numeric(df["calc_time"], errors="coerce")
            df["funding_interval_hours"] = pd.to_numeric(
                df["funding_interval_hours"], errors="coerce"
            )
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")

            df = df.dropna()

        else:
            # Fallback for unknown types
            df = pd.read_csv(csv_path, header=None)
            result["rows_original"] = len(df)

        # Add metadata
        df["symbol"] = symbol
        df["date"] = date_str

        result["rows_cleaned"] = len(df)

        if result["rows_cleaned"] == 0:
            result["error"] = "No valid data after cleaning"
            return result

        result["steps_completed"].append("process")
        removed_rows = result["rows_original"] - result["rows_cleaned"]
        logger.info(
            f"✅ [{worker_id}] Processed: {result['rows_cleaned']:,} rows ({removed_rows:,} removed)"
        )

        # Step 4: Save with PyArrow Parquet (optimized I/O)
        logger.info(f"💾 [{worker_id}] Step 4/5: Saving with PyArrow Parquet")

        # Create output directory
        if file_type == "klines":
            save_dir = Path(output_dir) / "klines" / symbol / interval
        else:
            save_dir = Path(output_dir) / file_type / symbol

        save_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = save_dir / f"{date_str}.parquet"

        # Convert to PyArrow Table and save with optimized settings
        try:
            # Create PyArrow table from pandas DataFrame
            table = pa.Table.from_pandas(df, preserve_index=False)

            # Save with optimized PyArrow settings
            pq.write_table(
                table,
                parquet_path,
                compression="snappy",  # Fast compression
                use_dictionary=True,  # Better compression for strings
                row_group_size=50000,  # Optimized row group size
                write_statistics=True,  # Enable statistics for faster queries
                use_deprecated_int96_timestamps=False,  # Use modern timestamp format
            )

            result["file_size_parquet"] = parquet_path.stat().st_size
            result["compression_ratio"] = (
                result["file_size_parquet"] / result["file_size_original"] * 100
            )

            logger.info(f"💾 [{worker_id}] PyArrow Parquet saved: {parquet_path}")

        except Exception as parquet_error:
            logger.warning(
                f"⚠️  [{worker_id}] PyArrow save failed, trying fallback: {parquet_error}"
            )

            # Fallback: Use pandas to_parquet (still uses PyArrow under the hood)
            df.to_parquet(parquet_path, compression="snappy", index=False)
            result["file_size_parquet"] = parquet_path.stat().st_size
            result["compression_ratio"] = (
                result["file_size_parquet"] / result["file_size_original"] * 100
            )

            logger.info(f"💾 [{worker_id}] Fallback Parquet saved: {parquet_path}")

        result["steps_completed"].append("save")

        # Step 5: Cleanup
        logger.info(f"🧹 [{worker_id}] Step 5/5: Cleanup")

        csv_path.unlink()
        zip_path.unlink()

        try:
            temp_dir.rmdir()
            temp_parent = temp_dir.parent
            if temp_parent.name == worker_id:
                temp_parent.rmdir()
        except:
            pass

        result["steps_completed"].append("cleanup")
        result["success"] = True

        logger.info(
            f"✅ [{worker_id}] PyArrow task {task_id} COMPLETE: {result['rows_cleaned']:,} rows"
        )
        logger.info(
            f"📊 [{worker_id}] Compression: {result['file_size_original']:,} → {result['file_size_parquet']:,} bytes ({result['compression_ratio']:.1f}%)"
        )

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"❌ [{worker_id}] PyArrow task {task_id} FAILED: {e}")

    result["processing_time"] = time.time() - start_time
    return result


def generate_tasks(
    symbols: List[str],
    year: int,
    intervals: List[str] = ["1h"],
    include_trades: bool = True,
    include_funding_rate: bool = True,
    download_type: str = "monthly",
) -> List[Tuple]:
    """Generate download tasks with task IDs"""
    BASE_URL = "https://data.binance.vision"
    tasks = []
    task_id = 1

    if download_type == "monthly":
        for month in range(1, 13):
            date_str = f"{year}-{month:02d}"

            for symbol in symbols:
                # Klines tasks
                for interval in intervals:
                    url = f"{BASE_URL}/data/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
                    tasks.append(
                        (url, symbol, date_str, interval, "klines", None, task_id)
                    )
                    task_id += 1

                # Trades tasks
                if include_trades:
                    url = f"{BASE_URL}/data/futures/um/monthly/trades/{symbol}/{symbol}-trades-{date_str}.zip"
                    tasks.append((url, symbol, date_str, None, "trades", None, task_id))
                    task_id += 1

                # Funding rate tasks
                if include_funding_rate:
                    url = f"{BASE_URL}/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{date_str}.zip"
                    tasks.append(
                        (url, symbol, date_str, None, "fundingRate", None, task_id)
                    )
                    task_id += 1

    return tasks


def collect_data_pyarrow_compatible(
    symbols: List[str],
    year: int = 2024,
    intervals: List[str] = ["1h"],
    data_dir: str = "binance_pyarrow_compatible",
    include_trades: bool = True,
    include_funding_rate: bool = True,
    download_type: str = "monthly",
    n_workers: int = 6,
    chunk_size: int = 2,
):
    """
    Collect data with PyArrow Parquet optimization (compatible version)
    """
    print(f"\n{'='*80}")
    print(f"🚀 BINANCE PYARROW COMPATIBLE COLLECTOR")
    print(f"{'='*80}")
    print(f"📊 Symbols: {symbols}")
    print(f"📅 Year: {year}")
    print(f"⏱️  Intervals: {intervals}")
    print(f"📁 Output directory: {data_dir}")
    print(f"👥 Workers: {n_workers}")
    print(f"📦 Chunk size: {chunk_size}")
    print(f"🏹 PyArrow version: {pa.__version__}")
    print(f"💱 Include trades: {include_trades}")
    print(f"💰 Include funding rates: {include_funding_rate}")

    # Generate all tasks
    tasks = generate_tasks(
        symbols=symbols,
        year=year,
        intervals=intervals,
        include_trades=include_trades,
        include_funding_rate=include_funding_rate,
        download_type=download_type,
    )

    # Add output directory to each task
    tasks = [
        (url, symbol, date_str, interval, file_type, data_dir, task_id)
        for url, symbol, date_str, interval, file_type, _, task_id in tasks
    ]

    total_tasks = len(tasks)
    print(f"\n📋 Generated {total_tasks} tasks to process with PyArrow")

    # Track results
    successful_tasks = 0
    failed_tasks = 0
    total_original_size = 0
    total_parquet_size = 0
    total_rows_processed = 0
    total_processing_time = 0

    start_time = time.time()

    # Process with multiprocessing
    with Pool(processes=n_workers) as pool:
        print(f"\n🔄 Starting PyArrow compatible multiprocess execution...")

        with tqdm(
            total=total_tasks,
            desc="PyArrow processing",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:

            for result in pool.imap_unordered(
                process_single_file_pyarrow_compatible, tasks, chunksize=chunk_size
            ):
                if result["success"]:
                    successful_tasks += 1
                    total_original_size += result["file_size_original"]
                    total_parquet_size += result["file_size_parquet"]
                    total_rows_processed += result["rows_cleaned"]
                    total_processing_time += result["processing_time"]

                    logger.info(
                        f"🎉 [{result['worker_id']}] PyArrow SUCCESS #{successful_tasks}: {result['symbol']}-{result['file_type']}-{result['date_str']} ({result['processing_time']:.1f}s, {result['compression_ratio']:.1f}% compression)"
                    )

                else:
                    failed_tasks += 1
                    logger.error(
                        f"💥 [{result['worker_id']}] PyArrow FAILED #{failed_tasks}: {result['symbol']}-{result['file_type']}-{result['date_str']} - {result['error']}"
                    )

                pbar.update(1)

                # Update progress bar with stats
                success_rate = (
                    successful_tasks / (successful_tasks + failed_tasks) * 100
                )
                avg_compression = (
                    total_parquet_size / total_original_size * 100
                    if total_original_size > 0
                    else 0
                )
                pbar.set_description(
                    f"PyArrow processing (✅{successful_tasks} ❌{failed_tasks} - {success_rate:.1f}% - {avg_compression:.1f}% compression)"
                )

    # Final summary
    total_time = time.time() - start_time
    avg_compression = (
        total_parquet_size / total_original_size * 100 if total_original_size > 0 else 0
    )
    avg_processing_time = (
        total_processing_time / successful_tasks if successful_tasks > 0 else 0
    )

    print(f"\n{'='*80}")
    print(f"🎉 PYARROW COMPATIBLE COLLECTION COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Successful tasks: {successful_tasks}")
    print(f"❌ Failed tasks: {failed_tasks}")
    print(f"📊 Success rate: {successful_tasks/total_tasks*100:.1f}%")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"⚡ Average processing time per task: {avg_processing_time:.1f} seconds")
    print(f"📈 Total rows processed: {total_rows_processed:,}")
    print(
        f"💾 PyArrow compression: {total_original_size:,} → {total_parquet_size:,} bytes ({avg_compression:.1f}%)"
    )
    print(
        f"🚀 Processing speed: {total_rows_processed/(total_time*1000):.1f}K rows/second"
    )
    print(f"📁 Data saved to: {data_dir}")

    # Clean up temp directories
    temp_base = Path("temp_data")
    if temp_base.exists():
        import shutil

        shutil.rmtree(temp_base)
        print(f"🧹 Cleaned up temporary directories")

    return {
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
        "total_time": total_time,
        "total_rows": total_rows_processed,
        "compression_ratio": avg_compression,
        "processing_speed": total_rows_processed / (total_time * 1000),  # K rows/second
    }


if __name__ == "__main__":
    print("🏹 PyArrow compatible collector ready!")
    print(f"📦 PyArrow version: {pa.__version__}")

    # Get top symbols
    symbols = get_top_usdt_futures_symbols(limit=100)

    # Process data with PyArrow Parquet optimization
    results = collect_data_pyarrow_compatible(
        symbols=symbols,
        year=2023,
        intervals=["1m"],
        data_dir="binance_futures_data",
        include_trades=False,
        include_funding_rate=True,
        n_workers=2,
        chunk_size=2,
    )

    print(f"\n🎯 PyArrow Compatible Results:")
    print(f"🚀 Processing speed: {results['processing_speed']:.1f}K rows/second")
    print(f"💾 Compression ratio: {results['compression_ratio']:.1f}%")
    print(f"⏱️  Total time: {results['total_time']/60:.1f} minutes")
