#!/usr/bin/env python3
"""
Benchmark script to measure retargeting time per frame.

This script retargets motion frames and reports detailed timing statistics
including average, min, max, and standard deviation of retargeting time per frame.
"""

import argparse
import pathlib
import os
import time
import statistics

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print
from rich.console import Console
from rich.table import Table

console = Console()


def benchmark_retarget(
    smplx_file,
    robot="unitree_g1",
    num_frames=None,
    warmup_frames=5,
    disable_viewer=True,
):
    """
    Benchmark retargeting performance.
    
    Args:
        smplx_file: Path to SMPLX motion file
        robot: Robot type to retarget to
        num_frames: Number of frames to benchmark (None = all frames)
        warmup_frames: Number of warmup frames to skip in timing
        disable_viewer: If True, skip visualization (faster, pure retargeting time)
    """
    HERE = pathlib.Path(__file__).parent
    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    # Load SMPLX trajectory
    print(f"[cyan]Loading SMPLX file: {smplx_file}[/cyan]")
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
    
    total_frames = len(smplx_data_frames)
    if num_frames is None:
        num_frames = total_frames
    num_frames = min(num_frames, total_frames)
    
    print(f"[green]Loaded {total_frames} frames (will benchmark {num_frames} frames)[/green]")
    print(f"[green]Target FPS: {aligned_fps}[/green]")
    
    # Initialize the retargeting system
    print(f"[cyan]Initializing retargeting system for robot: {robot}[/cyan]")
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
        aligned_fps=aligned_fps,
    )
    
    # Timing statistics
    retarget_times = []
    total_start_time = time.time()
    
    # Warmup frames (not counted in statistics)
    print(f"[yellow]Warming up with {warmup_frames} frames...[/yellow]")
    for i in range(min(warmup_frames, num_frames)):
        smplx_data = smplx_data_frames[i]
        _ = retarget.retarget(smplx_data)
    
    # Benchmark frames
    print(f"[cyan]Benchmarking {num_frames} frames...[/cyan]")
    start_frame = warmup_frames
    
    for i in range(start_frame, start_frame + num_frames):
        smplx_data = smplx_data_frames[i % total_frames]  # Loop if needed
        
        # Time the retargeting operation
        frame_start = time.time()
        qpos, _ = retarget.retarget(smplx_data)
        frame_end = time.time()
        
        retarget_time = (frame_end - frame_start) * 1000  # Convert to milliseconds
        retarget_times.append(retarget_time)
        
        # Progress update every 100 frames
        if (i - start_frame + 1) % 100 == 0:
            avg_time = statistics.mean(retarget_times)
            print(f"  Processed {i - start_frame + 1}/{num_frames} frames - Avg: {avg_time:.3f} ms/frame")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Calculate statistics
    avg_time = statistics.mean(retarget_times)
    min_time = min(retarget_times)
    max_time = max(retarget_times)
    median_time = statistics.median(retarget_times)
    
    try:
        std_time = statistics.stdev(retarget_times)
    except statistics.StatisticsError:
        std_time = 0.0
    
    # Calculate percentiles
    sorted_times = sorted(retarget_times)
    p50 = sorted_times[int(len(sorted_times) * 0.50)]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]
    
    # Calculate theoretical FPS
    theoretical_fps = 1000.0 / avg_time if avg_time > 0 else 0
    
    # Display results
    print("\n" + "="*60)
    print("[bold green]Benchmark Results[/bold green]")
    print("="*60)
    
    table = Table(title="Retargeting Performance Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_column("Unit", style="yellow")
    
    table.add_row("Total frames benchmarked", str(num_frames), "frames")
    table.add_row("Total time", f"{total_time:.3f}", "seconds")
    table.add_row("", "", "")
    table.add_row("Average time per frame", f"{avg_time:.3f}", "ms")
    table.add_row("Median time per frame", f"{median_time:.3f}", "ms")
    table.add_row("Minimum time per frame", f"{min_time:.3f}", "ms")
    table.add_row("Maximum time per frame", f"{max_time:.3f}", "ms")
    table.add_row("Standard deviation", f"{std_time:.3f}", "ms")
    table.add_row("", "", "")
    table.add_row("50th percentile (P50)", f"{p50:.3f}", "ms")
    table.add_row("95th percentile (P95)", f"{p95:.3f}", "ms")
    table.add_row("99th percentile (P99)", f"{p99:.3f}", "ms")
    table.add_row("", "", "")
    table.add_row("Theoretical FPS", f"{theoretical_fps:.2f}", "FPS")
    table.add_row("Target FPS", f"{aligned_fps:.2f}", "FPS")
    
    console.print(table)
    
    # Performance assessment
    time_per_frame_seconds = avg_time / 1000.0
    required_time_per_frame = 1.0 / aligned_fps
    
    print("\n[bold]Performance Assessment:[/bold]")
    if time_per_frame_seconds <= required_time_per_frame:
        print(f"[green]✓ Real-time capable: {avg_time:.3f} ms/frame < {required_time_per_frame*1000:.3f} ms/frame (required for {aligned_fps} FPS)[/green]")
    else:
        print(f"[red]✗ Not real-time capable: {avg_time:.3f} ms/frame > {required_time_per_frame*1000:.3f} ms/frame (required for {aligned_fps} FPS)[/red]")
        slowdown_factor = time_per_frame_seconds / required_time_per_frame
        print(f"[yellow]  Running at {1/slowdown_factor:.2f}x slower than real-time[/yellow]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark retargeting performance per frame"
    )
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
            "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
            "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
            "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong", "tienkung"
        ],
        default="unitree_g1",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to benchmark (default: all frames)",
    )
    parser.add_argument(
        "--warmup_frames",
        type=int,
        default=5,
        help="Number of warmup frames to skip (default: 5)",
    )
    
    args = parser.parse_args()
    
    benchmark_retarget(
        smplx_file=args.smplx_file,
        robot=args.robot,
        num_frames=args.num_frames,
        warmup_frames=args.warmup_frames,
    )
