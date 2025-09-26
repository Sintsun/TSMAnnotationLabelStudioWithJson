#!/usr/bin/env python3
"""
Create multiple TSM datasets with different configurations
"""

import os
import subprocess
import argparse
import logging
import shutil

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(cmd, description):
    logging.info(f"Running: {description}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ {description} failed: {e.stderr}")
        return False

def create_video_dataset(config_name, video_folder, json_folder, output_folder, label_studio_json, config_args):
    # Create output directory
    config_output_dir = os.path.join(output_folder, f"videos_{config_name}")
    os.makedirs(config_output_dir, exist_ok=True)
    
    # Build command for json_to_24frame_video.py
    cmd = [
        "python", "json_to_24frame_video.py",
        "--video_folder_path", video_folder,
        "--dataset_path", json_folder,
        "--output_video_path", config_output_dir,
        "--temporal_size", "24",
        "--min_confidence", "0.4"
    ]
    cmd.extend(config_args)
    
    # Run video generation
    success = run_command(cmd, f"Generate videos for {config_name}")
    if not success:
        return False
    
    # Create TSM dataset
    tsm_output_dir = os.path.join(output_folder, f"tsm_dataset_{config_name}")
    os.makedirs(tsm_output_dir, exist_ok=True)
    
    # Copy label studio JSON temporarily
    temp_json_path = f"temp_label_studio_{config_name}.json"
    shutil.copy2(label_studio_json, temp_json_path)
    
    try:
        # Build command for generate_tsm_dataset.py
        tsm_cmd = [
            "python", "generate_tsm_dataset.py",
            "--video_folder", config_output_dir,
            "--output_folder", tsm_output_dir,
            "--json_folder", ".",
            "--expected_frames", "24"
        ]
        
        success = run_command(tsm_cmd, f"Generate TSM dataset for {config_name}")
        if success:
            logging.info(f"✓ Successfully created TSM dataset: {tsm_output_dir}")
            return True
        return False
    finally:
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)

def main():
    parser = argparse.ArgumentParser(description="Create multiple TSM datasets")
    parser.add_argument("--video_folder", required=True, help="Path to original video folder")
    parser.add_argument("--json_folder", required=True, help="Path to original JSON annotations folder")
    parser.add_argument("--label_studio_json", required=True, help="Path to Label Studio JSON file")
    parser.add_argument("--output_folder", required=True, help="Base output folder")
    parser.add_argument("--configs", nargs="+", default=["default", "square"], help="Configurations to create")
    
    args = parser.parse_args()
    setup_logging()
    
    # Define configurations
    configurations = {
        "default": {
            "description": "Default settings (no square, no upper crop)",
            "args": ["--bbox_resize_factor", "1.2", "--expand_crop_factor", "1.1"]
        },
        "square": {
            "description": "Square videos",
            "args": ["--square", "--bbox_resize_factor", "1.2", "--expand_crop_factor", "1.1"]
        },
        "upper_crop": {
            "description": "Upper body crop",
            "args": ["--upper_crop", "--crop_ratio", "1.6", "--bbox_resize_factor", "1.2", "--expand_crop_factor", "1.1"]
        }
    }
    
    # Validate paths
    for path in [args.video_folder, args.json_folder, args.label_studio_json]:
        if not os.path.exists(path):
            logging.error(f"Path not found: {path}")
            return 1
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    logging.info("=== Creating Multiple TSM Datasets ===")
    successful_configs = []
    
    for config_name in args.configs:
        if config_name not in configurations:
            logging.warning(f"Unknown configuration: {config_name}")
            continue
        
        config = configurations[config_name]
        logging.info(f"\n--- Processing {config_name}: {config['description']} ---")
        
        success = create_video_dataset(
            config_name, args.video_folder, args.json_folder, 
            args.output_folder, args.label_studio_json, config['args']
        )
        
        if success:
            successful_configs.append(config_name)
    
    logging.info(f"\n=== Summary ===")
    logging.info(f"Successful: {successful_configs}")
    logging.info(f"Output folder: {args.output_folder}")
    
    return 0

if __name__ == "__main__":
    exit(main())
