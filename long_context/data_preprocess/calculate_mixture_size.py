#!/usr/bin/env python3
"""
Calculate the total number of tokens and samples in a data mixture folder.
Follows symlinks and recursively processes all .bin files in the directory tree.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict


def calculate_mixture_stats(folder_path: str, seq_len: int) -> Tuple[int, int, Dict[str, Dict]]:
    """
    Calculate total tokens and samples in a data mixture folder.
    
    Args:
        folder_path: Path to the data mixture folder
        seq_len: Sequence length for calculating samples
        
    Returns:
        Tuple of (total_tokens, total_samples, group_stats)
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    total_tokens = 0
    total_samples = 0
    group_stats = {}
    
    # Walk through the directory tree, following symlinks
    for root, dirs, files in os.walk(folder, followlinks=True):
        root_path = Path(root)
        
        for file in files:
            if file.endswith('.bin'):
                file_path = root_path / file
                try:
                    # Get file size in bytes
                    file_size = file_path.stat().st_size
                    
                    # Calculate tokens (assuming 4 bytes per token)
                    tokens_in_file = file_size // 4
                    samples_in_file = tokens_in_file // seq_len
                    
                    total_tokens += tokens_in_file
                    total_samples += samples_in_file
                    
                    # Determine group (direct subfolder of the main folder)
                    relative_path = file_path.relative_to(folder)
                    group_name = relative_path.parts[0] if relative_path.parts else "root"
                    
                    # Initialize group stats if not exists
                    if group_name not in group_stats:
                        group_stats[group_name] = {
                            'tokens': 0,
                            'samples': 0,
                            'files': 0
                        }
                    
                    # Add to group stats
                    group_stats[group_name]['tokens'] += tokens_in_file
                    group_stats[group_name]['samples'] += samples_in_file
                    group_stats[group_name]['files'] += 1
                    
                    print(f"  {file_path.relative_to(folder)}: {tokens_in_file:,} tokens, {samples_in_file:,} samples")
                    
                except (OSError, IOError) as e:
                    print(f"Warning: Could not read {file_path}: {e}")
    
    return total_tokens, total_samples, group_stats


def main():
    parser = argparse.ArgumentParser(
        description="Calculate tokens and samples in a data mixture folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calculate_mixture_size.py /path/to/mixture 16384
  python calculate_mixture_size.py ./mixture_16k_64 8192
        """
    )
    
    parser.add_argument(
        "folder",
        help="Path to the data mixture folder"
    )
    
    parser.add_argument(
        "seq_len",
        type=int,
        help="Sequence length for calculating samples"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Analyzing data mixture in: {args.folder}")
        print(f"Sequence length: {args.seq_len}")
        print("-" * 60)
        
        total_tokens, total_samples, group_stats = calculate_mixture_stats(args.folder, args.seq_len)
        
        print("-" * 60)
        print(f"GROUP STATISTICS:")
        print("-" * 60)
        
        # Sort groups by token count (descending)
        sorted_groups = sorted(group_stats.items(), key=lambda x: x[1]['tokens'], reverse=True)
        
        for group_name, stats in sorted_groups:
            token_ratio = (stats['tokens'] / total_tokens * 100) if total_tokens > 0 else 0
            print(f"  {group_name}:")
            print(f"    Tokens: {stats['tokens']:,} ({token_ratio:.1f}%)")
            print(f"    Samples: {stats['samples']:,}")
            print(f"    Files: {stats['files']}")
            if stats['samples'] > 0:
                avg_tokens_per_sample = stats['tokens'] / stats['samples']
                print(f"    Avg tokens per sample: {avg_tokens_per_sample:,.1f}")
            print()
        
        print("-" * 60)
        print(f"SUMMARY:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Tokens per sample: {total_tokens / total_samples if total_samples > 0 else 0:,.1f}")
        print(f"  Number of groups: {len(group_stats)}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()