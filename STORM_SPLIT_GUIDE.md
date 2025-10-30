# Storm-Level Data Split Guide

## Overview

This document explains the improved data splitting strategy for the cyclone forecasting model.

## Understanding "Samples" vs "Storms"

### What is a Sample?

A **sample** is a consecutive pair of timesteps from the same storm:
- **Input**: Current timestep (GRIDSAT image + ERA5 atmospheric data)
- **Target**: Next timestep (GRIDSAT image)

### Example

If a storm has 10 timesteps:
- Number of samples = 9 (pairs: 0→1, 1→2, 2→3, ..., 8→9)

### Dataset Size Calculation

```
Total samples = Sum of (num_timesteps - 1) across all storms
```

For example:
- 210 storms × 40 timesteps average = ~8400 samples
- NOT 210 storms × 70 (incorrect interpretation)

## Old vs New Split Strategy

### Old Strategy (Fixed Storms)
```python
test_storm = ['2022349N13068']    # 1 storm, 9 samples
val_storm = ['2022345N17125']     # 1 storm, 8 samples
```

**Problem**: Validation set too small (only 8 samples!)

### New Strategy (Random Storm-Level Split)
```python
test_storm = ['2022349N13068']    # 1 storm (kept as-is)
val_storm = [5 randomly selected storms]  # ~200-400 samples
```

**Benefits**:
- ✅ Much larger validation set
- ✅ Random sampling reduces bias
- ✅ Reproducible (fixed seed)
- ✅ Storm-level split (no data leakage)

## Implementation Details

### New Configuration Parameters

```python
config = {
    'test_storm': ['2022349N13068'],  # Keep test set as-is
    'num_val_storms': 5,              # Number of storms for validation
    'random_seed': 42,                # For reproducible splits
    'min_timesteps': 5,               # Minimum timesteps per storm
}
```

### How It Works

1. **Scan all storms** across all year folders
2. **Filter** storms with fewer than `min_timesteps` timesteps
3. **Exclude** test storms from selection
4. **Randomly select** `num_val_storms` storms for validation
5. **Remaining storms** go to training set

### Reproducibility

The split is deterministic when using the same:
- `random_seed`
- `train_years`
- `min_timesteps`
- `test_storm`

## Usage

### Method 1: Automatic Split (Recommended)

Just run the training script - it will automatically create a random split:

```bash
python notebook.py
```

The script will:
1. Scan all available storms
2. Print split statistics
3. Create train/val/test dataloaders

### Method 2: Analyze and Generate Split First

To understand your data before training:

```bash
python generate_storm_split.py
```

This will:
1. Print detailed dataset statistics
2. Show storm distribution
3. Generate and save `storm_split.json`
4. Print the validation storm list

### Method 3: Use Pre-generated Split

If you want to use a specific split across runs:

1. Generate split once:
   ```bash
   python generate_storm_split.py
   ```

2. Copy the validation storms from output

3. Manually set in `notebook.py`:
   ```python
   config = {
       'val_storm': ['storm1', 'storm2', 'storm3', 'storm4', 'storm5'],
       # Comment out 'num_val_storms' to use manual split
   }
   ```

## Expected Results

### Before (Old Split)
```
Dataset sizes:
  Training: 8562 samples
  Validation: 8 samples      ← Too small!
  Test: 9 samples
```

### After (New Split with 5 storms)
```
Dataset sizes:
  Training: ~8100-8300 samples
  Validation: ~250-450 samples  ← Much better!
  Test: 9 samples
```

Actual numbers depend on:
- Which storms are randomly selected
- Number of timesteps per storm

## Customization

### Change Number of Validation Storms

```python
config = {
    'num_val_storms': 10,  # Use 10 storms instead of 5
}
```

### Change Random Seed

```python
config = {
    'random_seed': 123,  # Different seed = different random storms
}
```

### Filter Small Storms

```python
config = {
    'min_timesteps': 10,  # Only use storms with 10+ timesteps
}
```

## Data Structure

Your data should be organized as:

```
root_dir/
  2005_0/
    2005_0/              # Nested year folder
      storm1/            # Storm folder (e.g., '2005123N45678')
        timestep1/       # Timestep folder
          GRIDSAT_data.npy
          ERA5_data.npy
        timestep2/
          GRIDSAT_data.npy
          ERA5_data.npy
        ...
      storm2/
        ...
  2016_0/
    ...
  2022_0/
    ...
```

## Troubleshooting

### "No storms found"

Check that:
- `root_dir` path is correct
- `train_years` matches your folder names
- Nested folder structure is correct

### "Not enough storms for split"

Reduce `num_val_storms` or `min_timesteps`:

```python
config = {
    'num_val_storms': 3,   # Use fewer storms
    'min_timesteps': 3,    # Allow smaller storms
}
```

### Validation set still too small

Increase `num_val_storms`:

```python
config = {
    'num_val_storms': 10,  # Use more storms
}
```

## Technical Details

### Functions Added

1. **`get_all_storms(root_dir, years)`**
   - Scans directory and collects all storm names
   - Returns dictionary with storm metadata

2. **`create_random_storm_split(root_dir, years, ...)`**
   - Creates random storm-level split
   - Prints detailed statistics
   - Returns train/val/test storm lists

### Files Modified

- `notebook.py`:
  - Added utility functions (lines 25-148)
  - Updated `main()` to use random split (lines 2042-2057)

### Files Created

- `generate_storm_split.py`: Standalone analysis tool
- `STORM_SPLIT_GUIDE.md`: This documentation

## References

- Original issue: Validation set has only 8 samples
- Solution: Random storm-level split with 5 validation storms
- Test set kept unchanged for consistency
