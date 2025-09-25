import shutil
import sys
import os
import random
from tqdm import tqdm  
import argparse
from pathlib import Path
import math
val_portion = 0.1
"""
The script selects every n-th frame from input folders and outputs folders with 8 segments each.
It then generates train.txt and val.txt by randomly splitting the folders.
Usage:
     python3 segments_select_8frames.py <input_dir> <output_dir> [-s SEGMENTS] [-m MAX_FRAMES]
Example:
     python3 segments_select_8frames.py ./input temp -s 8 -m 16
"""


def generate_unique_output_dir(output_dir: Path) -> Path:
    """
    Generates a unique output directory by appending a numerical suffix if the directory exists.

    :param output_dir: Desired path for the output directory.
    :return: A unique Path object for the output directory.
    """
    if not output_dir.exists():
        return output_dir

    suffix = 1
    while True:
        new_output_dir = output_dir.parent / f"{output_dir.stem}_{suffix}"
        if not new_output_dir.exists():
            return new_output_dir
        suffix += 1


def copy_categories_txt(input_dir: Path, output_dir: Path):
    """
    Copies the 'categories.txt' file from input_dir to output_dir.

    :param input_dir: Path to the input directory containing 'categories.txt'.
    :param output_dir: Path to the output directory where 'categories.txt' will be copied.
    """
    src = input_dir / 'categories.txt'
    dst = output_dir / 'categories.txt'

    if not src.exists():
        raise FileNotFoundError(f"'categories.txt' not found in '{input_dir}'.")

    shutil.copy(src, dst)
    tqdm.write(f"Copied 'categories.txt' to '{dst}'.")


def extract_segmented_frames(input_images_dir: Path, output_images_dir: Path, segments: int, max_frames: int = None):
    """
    Extracts n frames from each subfolder in input_images_dir and copies them to output_images_dir.

    :param input_images_dir: Path to 'images/' with subfolders containing image frames.
    :param output_images_dir: Path to 'images/' in the output directory.
    :param segments: Number of frames to extract per subfolder.
    :param max_frames: (Optional) Max number of frames to read from each subfolder.
    """
    if not input_images_dir.exists():
        raise ValueError(f"Images directory '{input_images_dir}' does not exist.")

    subfolders = [subfolder for subfolder in input_images_dir.iterdir() if subfolder.is_dir()]
    if not subfolders:
        raise ValueError(
            f"No subfolders found in '{input_images_dir}'. Ensure 'images/' has subdirectories with images.")

    for subfolder in tqdm(subfolders, desc="Processing folders", unit="folder"):
        input_subfolder = subfolder
        output_subfolder = output_images_dir / subfolder.name
        output_subfolder.mkdir(parents=True, exist_ok=True)

        image_files = sorted([file for file in input_subfolder.iterdir() if
                              file.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}])
        total_frames = len(image_files)
        if total_frames == 0:
            tqdm.write(f"Warning: No images in '{input_subfolder}'. Skipping.")
            continue

        if segments <= 0:
            tqdm.write(f"Warning: Segments must be positive. Skipping '{subfolder.name}'.")
            continue

        # Limit to first n frames if max_frames is set
        if max_frames is not None:
            if max_frames < segments:
                tqdm.write(
                    f"Warning: max_frames ({max_frames}) < segments ({segments}) in '{subfolder.name}'. Skipping.")
                continue
            frames_to_consider = image_files[:max_frames] if max_frames < total_frames else image_files
            if max_frames < total_frames:
                tqdm.write(f"Folder '{subfolder.name}': Considering first {max_frames} frame(s).")
            else:
                tqdm.write(f"Folder '{subfolder.name}': Using all {total_frames} frame(s).")
        else:
            frames_to_consider = image_files

        actual_total_frames = len(frames_to_consider)
        segments_to_extract = min(segments, actual_total_frames)

        n = math.floor(actual_total_frames / segments_to_extract)
        n = n if n > 0 else 1
        tqdm.write(f"Folder '{subfolder.name}': Extracting every {n} frame(s).")

        selected_frames = [frames_to_consider[i] for i in range(0, actual_total_frames, n)][:segments_to_extract]

        for idx, frame in enumerate(
                tqdm(selected_frames, desc=f"Copying from '{subfolder.name}'", unit="frame", leave=False), start=1):
            src = frame
            ext = frame.suffix
            new_filename = f"img_{idx:03}{ext}"
            dst = output_subfolder / new_filename

            if dst.exists():
                suffix = 1
                unique_filename = f"img_{idx:03}_{suffix}{ext}"
                unique_dst = output_subfolder / unique_filename
                while unique_dst.exists():
                    suffix += 1
                    unique_filename = f"img_{idx:03}_{suffix}{ext}"
                    unique_dst = output_subfolder / unique_filename
                dst = unique_dst
                tqdm.write(f"Note: '{new_filename}' exists. Renamed to '{unique_filename}'.")

            shutil.copy(src, dst)


def gen_txt(dataset_path: Path, images_path: Path, val_portion: float = 0.1):
    """
    Generates train.txt and val.txt with folder names, file counts, and labels.

    Args:
        dataset_path (Path): Directory to save train.txt and val.txt.
        images_path (Path): Directory containing image subfolders.
        val_portion (float): Fraction of folders for validation (default: 0.1).
    """
    tqdm.write(f"Dataset Path: {dataset_path}")
    tqdm.write(f"Images Path: {images_path}")

    train_file = dataset_path / 'train.txt'
    val_file = dataset_path / 'val.txt'

    for file in [train_file, val_file]:
        if file.exists():
            file.unlink()
            tqdm.write(f"Removed existing file: {file}")

    train_count = 0
    val_count = 0

    with train_file.open('a') as train, val_file.open('a') as val:
        for subdir in tqdm(os.listdir(images_path), desc='Generating train.txt and val.txt', unit='folder'):
            subdir_path = images_path / subdir

            if subdir_path.is_dir():
                file_count = len([f for f in subdir_path.iterdir() if f.is_file()])

                if subdir.startswith("cls"):
                    label = subdir.split("_")[0][3:]
                    class_name = subdir
                    line = f"{class_name} {file_count} {label}\n"

                    if random.random() > val_portion:
                        train.write(line)
                        train_count += 1
                    else:
                        val.write(line)
                        val_count += 1

    tqdm.write(f"Training entries written: {train_count}")
    tqdm.write(f"Validation entries written: {val_count}")


def process_frames(input_dir: Path, output_dir_name: str, segments: int, max_frames: int = None):
    """
    Processes input by extracting frames and generating train/val splits.

    :param input_dir: Path to input directory with 'images/' and 'categories.txt'.
    :param output_dir_name: Desired output directory name.
    :param segments: Number of frames to extract per subfolder.
    :param max_frames: (Optional) Max frames to read from each subfolder.
    """
    if max_frames is not None and max_frames < segments:
        raise ValueError(f"max_frames ({max_frames}) must be >= segments ({segments}).")

    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory '{input_dir}' is invalid.")

    output_dir = (input_dir.parent / output_dir_name).resolve()
    unique_output_dir = generate_unique_output_dir(output_dir)
    if unique_output_dir != output_dir:
        tqdm.write(f"Output directory '{output_dir}' exists. Creating '{unique_output_dir}'.")
    else:
        tqdm.write(f"Creating output directory '{unique_output_dir}'.")

    unique_output_dir.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Created output directory '{unique_output_dir}'.")

    copy_categories_txt(input_dir, unique_output_dir)

    input_images_dir = input_dir / 'images'
    output_images_dir = unique_output_dir / 'images'
    output_images_dir.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Created output images directory '{output_images_dir}'.")

    extract_segmented_frames(input_images_dir, output_images_dir, segments, max_frames)

    # Generate train.txt and val.txt
    gen_txt(unique_output_dir, output_images_dir, val_portion=val_portion)


def main():
    parser = argparse.ArgumentParser(description="""
    Extract n frames from each subfolder and generate train.txt and val.txt splits.

    - Output directory is a sibling to input.
    - If output exists, a unique suffix is appended.
    - Renames frames to img_001.ext, etc.
    - Generates train.txt and val.txt with random splits.
    """)
    parser.add_argument("input_dir", type=Path,
                        help="Path to input directory with 'images/' and 'categories.txt'.")
    parser.add_argument("output_dir", type=str, help="""Desired output directory name.
    - Relative name (e.g., 'temp') creates as sibling.
    - Absolute path creates at specified location.
    """)
    parser.add_argument("-s", "--segments", type=int, default=8,
                        help="Number of frames to extract per subfolder (default: 8).")
    parser.add_argument("-m", "--max-frames", type=int, default=None,
                        help="(Optional) Max frames to read from each folder before extraction.")

    args = parser.parse_args()

    try:
        process_frames(args.input_dir, args.output_dir, args.segments, args.max_frames)
        tqdm.write("Processing completed successfully.")
    except Exception as e:
        tqdm.write(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
