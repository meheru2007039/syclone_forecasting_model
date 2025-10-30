"""
Utility script to analyze storm data and generate train/val/test split.

This script helps you:
1. Understand the data structure and available storms
2. Generate a random storm-level split
3. Save the split configuration for reuse

Usage:
    python generate_storm_split.py

Output:
    - Prints statistics about all available storms
    - Creates storm_split.json with the split configuration
"""

import json
from pathlib import Path
import sys

# Import utilities from notebook.py
from notebook import get_all_storms, create_random_storm_split


def analyze_dataset(root_dir, years):
    """
    Analyze the dataset and print detailed statistics.
    """
    print("="*80)
    print("DATASET ANALYSIS")
    print("="*80)

    all_storms = get_all_storms(root_dir, years)

    if not all_storms:
        print(f"ERROR: No storms found in {root_dir}")
        return None

    # Overall statistics
    total_storms = len(all_storms)
    total_timesteps = sum(info['num_timesteps'] for info in all_storms.values())
    total_samples = sum(info['num_samples'] for info in all_storms.values())

    print(f"\nOverall Statistics:")
    print(f"  Total storms: {total_storms}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Total samples (consecutive pairs): {total_samples}")
    print(f"  Average timesteps per storm: {total_timesteps / total_storms:.1f}")
    print(f"  Average samples per storm: {total_samples / total_storms:.1f}")

    # Per-year breakdown
    print(f"\nPer-year breakdown:")
    for year in years:
        year_storms = {name: info for name, info in all_storms.items()
                       if info['year'] == year}
        if year_storms:
            year_samples = sum(info['num_samples'] for info in year_storms.values())
            print(f"  {year}: {len(year_storms)} storms, {year_samples} samples")

    # Distribution of storm sizes
    storm_sizes = [info['num_timesteps'] for info in all_storms.values()]
    storm_sizes.sort()

    print(f"\nStorm size distribution (number of timesteps):")
    print(f"  Min: {min(storm_sizes)}")
    print(f"  Max: {max(storm_sizes)}")
    print(f"  Median: {storm_sizes[len(storm_sizes)//2]}")

    # Show some example storms
    print(f"\nExample storms (first 10):")
    for i, (name, info) in enumerate(list(all_storms.items())[:10]):
        print(f"  {name}: {info['num_timesteps']} timesteps, "
              f"{info['num_samples']} samples, year {info['year']}")

    return all_storms


def main():
    # Configuration - ADJUST THESE PATHS TO MATCH YOUR SETUP
    config = {
        'root_dir': r'/kaggle/input/setcd-dataset',
        'train_years': ['2005_0', '2016_0', '2022_0'],
        'test_storm': ['2022349N13068'],  # Keep test set as-is
        'num_val_storms': 5,  # Number of storms for validation
        'random_seed': 42,  # For reproducible splits
        'min_timesteps': 5,  # Minimum timesteps per storm
    }

    print("Configuration:")
    print(json.dumps(config, indent=2))
    print()

    # Analyze dataset
    all_storms = analyze_dataset(config['root_dir'], config['train_years'])

    if all_storms is None:
        print("\nCannot proceed without valid data.")
        sys.exit(1)

    # Create split
    print("\n" + "="*80)
    print("GENERATING RANDOM STORM SPLIT")
    print("="*80)

    try:
        storm_split = create_random_storm_split(
            root_dir=config['root_dir'],
            years=config['train_years'],
            num_val_storms=config['num_val_storms'],
            test_storms=config['test_storm'],
            seed=config['random_seed'],
            min_timesteps=config['min_timesteps']
        )

        # Save to file
        output_file = 'storm_split.json'
        with open(output_file, 'w') as f:
            json.dump({
                'config': config,
                'val_storms': storm_split['val_storms'],
                'test_storms': storm_split['test_storms'],
                'num_train_storms': len(storm_split['train_storms']),
                'num_val_storms': len(storm_split['val_storms']),
                'num_test_storms': len(storm_split['test_storms']),
            }, f, indent=2)

        print(f"\n" + "="*80)
        print(f"Split configuration saved to: {output_file}")
        print("="*80)

        print("\nTo use this split in your training, copy these storm lists:")
        print(f"\nval_storm = {storm_split['val_storms']}")
        print(f"test_storm = {storm_split['test_storms']}")

    except Exception as e:
        print(f"\nERROR generating split: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
