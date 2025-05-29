#!/usr/bin/env python3
"""
CLI script to visualize attribution weights from training logs.
Supports both local JSON files and parquet rollout data.

Usage:
    python visualize_attribution.py attribution_logs/step_100_attribution.json
    python visualize_attribution.py data_rollout/step_5/*.parquet --mode parquet
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import math

try:
    import pandas as pd
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


def rgb_to_ansi(r: int, g: int, b: int) -> str:
    """Convert RGB values to ANSI escape code."""
    return f"\033[38;2;{r};{g};{b}m"


def reset_color() -> str:
    """ANSI reset code."""
    return "\033[0m"


def attribution_to_color(attribution: float, min_attr: float, max_attr: float) -> tuple[int, int, int]:
    """
    Convert attribution weight to RGB color using a blue->white->red color ramp.
    
    Args:
        attribution: The attribution weight
        min_attr: Minimum attribution in the sequence
        max_attr: Maximum attribution in the sequence
        
    Returns:
        RGB tuple (r, g, b)
    """
    # Normalize attribution to [0, 1]
    if max_attr == min_attr:
        normalized = 0.5  # If all same, use middle color
    else:
        normalized = (attribution - min_attr) / (max_attr - min_attr)
    
    # Blue -> White -> Red color ramp
    if normalized < 0.5:
        # Blue to White: (0,0,255) -> (255,255,255)
        t = normalized * 2  # [0, 1]
        r = int(255 * t)
        g = int(255 * t)
        b = 255
    else:
        # White to Red: (255,255,255) -> (255,0,0)
        t = (normalized - 0.5) * 2  # [0, 1]
        r = 255
        g = int(255 * (1 - t))
        b = int(255 * (1 - t))
    
    return r, g, b


def visualize_attribution_sequence(tokens: List[Dict[str, Any]], title: str = "") -> None:
    """
    Visualize a sequence of tokens with attribution weights.
    
    Args:
        tokens: List of token dictionaries with 'token', 'attribution' keys
        title: Optional title for the visualization
    """
    if not tokens:
        print("No tokens to visualize.")
        return
    
    # Extract attribution values for normalization
    attributions = [t["attribution"] for t in tokens]
    min_attr = min(attributions)
    max_attr = max(attributions)
    
    print(f"\n{'='*60}")
    if title:
        print(f"📊 {title}")
    print(f"Attribution range: {min_attr:.4f} to {max_attr:.4f}")
    print(f"{'='*60}")
    
    # Print color legend
    print("\n🎨 Color Legend:")
    legend_steps = 5
    for i in range(legend_steps):
        val = i / (legend_steps - 1)
        attr_val = min_attr + val * (max_attr - min_attr)
        r, g, b = attribution_to_color(attr_val, min_attr, max_attr)
        color_code = rgb_to_ansi(r, g, b)
        print(f"{color_code}■{reset_color()} {attr_val:.4f}", end="  ")
    print(f"  (Low {rgb_to_ansi(0,0,255)}Blue{reset_color()} ← → {rgb_to_ansi(255,0,0)}Red{reset_color()} High)")
    
    print(f"\n📝 Response with attribution:")
    print("-" * 40)
    
    # Visualize tokens
    current_line = ""
    for i, token_data in enumerate(tokens):
        token = token_data["token"]
        attribution = token_data["attribution"]
        
        # Get color for this token
        r, g, b = attribution_to_color(attribution, min_attr, max_attr)
        color_code = rgb_to_ansi(r, g, b)
        
        # Add token to current line
        colored_token = f"{color_code}{token}{reset_color()}"
        
        # Check if we need to wrap (roughly 80 chars)
        if len(current_line) + len(token) > 80 and current_line:
            print(current_line)
            current_line = colored_token
        else:
            current_line += colored_token
    
    # Print remaining line
    if current_line:
        print(current_line)
    
    print("\n" + "-" * 40)
    
    # Print detailed attribution values
    print(f"\n🔍 Detailed Attribution Values:")
    for i, token_data in enumerate(tokens):
        token = token_data["token"]
        attribution = token_data["attribution"]
        # Use a simple background color for readability
        if attribution > (min_attr + max_attr) / 2:
            bg_color = f"\033[48;2;255;200;200m"  # Light red background
        else:
            bg_color = f"\033[48;2;200;200;255m"  # Light blue background
        
        print(f"{i:3d}: {bg_color}{token:20s}{reset_color()} {attribution:8.4f}")


def load_attribution_from_json(file_path: Path) -> List[Dict[str, Any]]:
    """Load attribution data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_attribution_from_parquet(file_paths: List[Path]) -> List[Dict[str, Any]]:
    """Load attribution data from parquet files (if available)."""
    if not PARQUET_AVAILABLE:
        print("❌ Error: pyarrow and pandas are required to read parquet files.")
        print("Install with: pip install pyarrow pandas")
        sys.exit(1)
    
    print("⚠️  Note: Attribution visualization from parquet files is not yet implemented.")
    print("This would require the inference pipeline to store attribution data in parquet files.")
    print("For now, use the JSON files from attribution_logs/ directory.")
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Visualize attribution weights from training logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize from JSON attribution log
  python visualize_attribution.py attribution_logs/step_100_attribution.json
  
  # Show specific sample from the log
  python visualize_attribution.py attribution_logs/step_100_attribution.json --sample 0
  
  # Future: Visualize from parquet (not yet implemented)
  python visualize_attribution.py data_rollout/step_5/*.parquet --mode parquet
        """
    )
    
    parser.add_argument("files", nargs="+", help="Path(s) to attribution files")
    parser.add_argument("--mode", choices=["json", "parquet"], default="json", 
                       help="Input file format (default: json)")
    parser.add_argument("--sample", type=int, default=None,
                       help="Show specific sample index (default: show all)")
    parser.add_argument("--max-samples", type=int, default=5,
                       help="Maximum number of samples to show (default: 5)")
    
    args = parser.parse_args()
    
    # Convert file paths
    file_paths = [Path(f) for f in args.files]
    
    # Load data based on mode
    if args.mode == "json":
        all_data = []
        for file_path in file_paths:
            if not file_path.exists():
                print(f"❌ Error: File {file_path} does not exist.")
                continue
            data = load_attribution_from_json(file_path)
            all_data.extend(data)
    elif args.mode == "parquet":
        all_data = load_attribution_from_parquet(file_paths)
    
    if not all_data:
        print("❌ No attribution data found.")
        sys.exit(1)
    
    print(f"📈 Found {len(all_data)} attribution samples")
    
    # Filter to specific sample if requested
    if args.sample is not None:
        if args.sample >= len(all_data):
            print(f"❌ Error: Sample index {args.sample} out of range (0-{len(all_data)-1})")
            sys.exit(1)
        all_data = [all_data[args.sample]]
    else:
        # Limit to max_samples
        all_data = all_data[:args.max_samples]
    
    # Visualize each sample
    for i, sample in enumerate(all_data):
        step = sample.get("step", "unknown")
        sample_idx = sample.get("sample_idx", i)
        title = f"Step {step}, Sample {sample_idx}"
        
        if "tokens" in sample and sample["tokens"]:
            visualize_attribution_sequence(sample["tokens"], title)
        else:
            print(f"⚠️  Sample {i} has no token attribution data")
        
        # Add separator between samples
        if i < len(all_data) - 1:
            input("\nPress Enter to continue to next sample...")


if __name__ == "__main__":
    main() 