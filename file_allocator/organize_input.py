#!/usr/bin/env python3
"""
Organize flat parquet files into symbol-based directory structure.

Scans input directory for parquet files named like:
  - symbol_future_*.parquet
  - symbol_call_*.parquet
  - symbol_put_*.parquet
  - symbol_index_*.parquet

Creates organized structure:
  input_organized/
    NIFTY/
      nifty_future_*.parquet
      nifty_call_*.parquet
      nifty_put_*.parquet
    SENSEX/
      sensex_future_*.parquet
      ...
"""

import shutil
from pathlib import Path
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def extract_symbol_from_filename(filename: str) -> str | None:
    """
    Extract symbol from parquet filename.
    
    Expected formats:
      - symbol_future_DDMMYYYY.parquet
      - symbol_call_DDMMYYYY.parquet
      - symbol_put_DDMMYYYY.parquet
      - symbol_index_DDMMYYYY.parquet
    
    Returns:
        Symbol in uppercase or None if pattern doesn't match
    """
    stem = filename.lower()
    
    # Remove .parquet extension
    if stem.endswith('.parquet'):
        stem = stem[:-8]
    
    # Split by underscore
    parts = stem.split('_')
    if len(parts) < 2:
        return None
    
    # First part should be symbol
    symbol = parts[0].strip()
    
    # Validate second part is a known type
    if len(parts) >= 2 and parts[1] in ('future', 'call', 'put', 'index', 'cash', 'spot'):
        return symbol.upper()
    
    return None


def get_file_type(filename: str) -> str | None:
    """
    Extract file type (future, call, put, index, etc.) from filename.
    
    Returns:
        File type or None if not recognized
    """
    stem = filename.lower()
    if stem.endswith('.parquet'):
        stem = stem[:-8]
    
    parts = stem.split('_')
    if len(parts) >= 2:
        file_type = parts[1]
        if file_type in ('future', 'call', 'put', 'index', 'cash', 'spot'):
            return file_type
    return None


def organize_files(input_dir: Path, output_dir: Path, copy_mode: bool = True):
    """
    Organize parquet files into symbol-based directories with type subdirectories.
    
    Structure:
      output_dir/
        nifty/
          futures/
            nifty_future_*.parquet
          options/
            nifty_call_*.parquet
            nifty_put_*.parquet
    
    Args:
        input_dir: Source directory with flat parquet files
        output_dir: Destination directory for organized structure
        copy_mode: If True, copy files; if False, move files (default: True/copy)
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Find all parquet files
    parquet_files = list(input_dir.glob("*.parquet"))
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {input_dir}")
        return
    
    logger.info(f"Found {len(parquet_files)} parquet files in {input_dir}")
    
    # Group files by symbol and type
    symbol_type_files = {}
    unrecognized = []
    
    for file_path in parquet_files:
        symbol = extract_symbol_from_filename(file_path.name)
        file_type = get_file_type(file_path.name)
        
        if symbol and file_type:
            # Use lowercase symbol for folder names
            symbol_lower = symbol.lower()
            
            if symbol_lower not in symbol_type_files:
                symbol_type_files[symbol_lower] = {}
            
            if file_type not in symbol_type_files[symbol_lower]:
                symbol_type_files[symbol_lower][file_type] = []
            
            symbol_type_files[symbol_lower][file_type].append(file_path)
        else:
            unrecognized.append(file_path)
    
    if unrecognized:
        logger.warning(f"Could not extract symbol/type from {len(unrecognized)} files:")
        for f in unrecognized[:5]:  # Show first 5
            logger.warning(f"  - {f.name}")
        if len(unrecognized) > 5:
            logger.warning(f"  ... and {len(unrecognized) - 5} more")
    
    # Create organized structure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    operation = "Copying" if copy_mode else "Moving"
    
    for symbol_lower in sorted(symbol_type_files.keys()):
        symbol_dir = output_dir / symbol_lower
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"{operation} files for {symbol_lower}...")
        
        for file_type, files in sorted(symbol_type_files[symbol_lower].items()):
            # Determine subdirectory based on file type
            if file_type == 'future':
                subdir = symbol_dir / 'futures'
            elif file_type in ('call', 'put'):
                subdir = symbol_dir / 'options'
            elif file_type in ('index', 'cash', 'spot'):
                subdir = symbol_dir / 'index'
            else:
                subdir = symbol_dir
            
            subdir.mkdir(parents=True, exist_ok=True)
            
            for file_path in files:
                dest_path = subdir / file_path.name
                
                try:
                    if copy_mode:
                        shutil.copy2(file_path, dest_path)
                    else:
                        shutil.move(str(file_path), str(dest_path))
                    total_processed += 1
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
    
    # Summary
    logger.info("="*60)
    logger.info("ORGANIZATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Symbols detected: {len(symbol_type_files)}")
    logger.info(f"Files processed: {total_processed}/{len(parquet_files)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)
    logger.info("\nOrganized structure:")
    for symbol_lower in sorted(symbol_type_files.keys()):
        logger.info(f"  {symbol_lower}/")
        for file_type in sorted(symbol_type_files[symbol_lower].keys()):
            file_count = len(symbol_type_files[symbol_lower][file_type])
            if file_type == 'future':
                logger.info(f"    futures/  ({file_count} files)")
            elif file_type in ('call', 'put'):
                logger.info(f"    options/  ({file_count} {file_type} files)")
            elif file_type in ('index', 'cash', 'spot'):
                logger.info(f"    index/  ({file_count} files)")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Organize flat parquet files into symbol-based directories"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent / "input",
        help="Input directory with flat parquet files (default: ./input)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "input_organized",
        help="Output directory for organized structure (default: ./input_organized)"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them (default: copy)"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting file organization...")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Mode: {'MOVE' if args.move else 'COPY'}")
    logger.info("="*60)
    
    organize_files(args.input_dir, args.output_dir, copy_mode=not args.move)


if __name__ == "__main__":
    main()

