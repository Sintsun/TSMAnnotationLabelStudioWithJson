import os
import cv2
import re
import time
import argparse
import sys
from tqdm import tqdm  # Ensure tqdm is installed: pip install tqdm


def natural_sort(l):
    """Sort the list in human order."""

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    return sorted(l, key=lambda x: [convert(c) for c in re.split('([0-9]+)', x)])


def extract_class_label(folder_name):
    """
    Extracts the class label from the folder name using regex.

    Args:
    - folder_name (str): The name of the folder from which to extract the class label.

    Returns:
    - str: The extracted class label or "Unknown" if not found.
    """
    # This regex searches for 'cls' followed by one or more digits
    match = re.search(r'cls(\d+)', folder_name)
    if match:
        return match.group(1)
    return "Unknown"


def play_video_from_frames(folder_path, fps=8, display=True):
    """
    Plays a video composed of frames from a given folder and displays the class label on the video.

    Args:
    - folder_path (str): Path to the folder containing frames.
    - fps (int): Frames per second for video playback (not used in manual mode).
    - display (bool): Whether to display the video using OpenCV.

    Returns:
    - str: Status flag - 'continue', 'skip', or 'quit'.
    """
    # Get all image file names in the folder, sorted naturally
    frame_files = natural_sort([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
    ])

    if not frame_files:
        print(f"No frames found in {folder_path}")
        return 'continue'

    # Extract the class label from the folder name
    folder_name = os.path.basename(folder_path)
    class_label = extract_class_label(folder_name)
    print(f"Processing Folder: {folder_name} | Class Label: {class_label}")

    # Read frames and store them in a list
    frames = []
    failed_frames = 0
    for frame_file in frame_files:
        frame_path = os.path.join(folder_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Failed to read frame: {frame_path}")
            failed_frames += 1
            continue
        frames.append(frame)

    if failed_frames > 0:
        print(f"Total failed frames in {folder_path}: {failed_frames}")

    # Display video if there are frames and display is enabled
    if frames and display:
        print(f"Playing video from folder: {folder_path}")
        paused = False
        for frame in frames:
            # Overlay class label on the frame
            cv2.putText(frame, f"Class: {class_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Video Playback - Press Space to Advance, 'q' to Quit, 'r' to Skip, 'p' to Pause", frame)

            while True:
                # Wait indefinitely for a key press
                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):
                    print("Playback interrupted by user. Exiting...")
                    cv2.destroyAllWindows()
                    return 'quit'
                elif key == ord('r'):
                    print("Skipped to next folder by user.")
                    cv2.destroyAllWindows()
                    return 'skip'
                elif key == ord('p'):
                    paused = not paused  # Toggle paused state
                    if paused:
                        print("Playback paused. Press 'p' again to resume.")
                    else:
                        print("Playback resumed.")
                elif key == 32:  # Spacebar key code is 32
                    if not paused:
                        break  # Exit the inner loop to show the next frame
                else:
                    print("Invalid key pressed. Use Space to advance, 'q' to quit, 'r' to skip, 'p' to pause.")

        cv2.destroyAllWindows()
        time.sleep(0.5)  # Short delay before the next video

    return 'continue'


def is_display_available():
    """Checks if a display is available for OpenCV to use."""
    if os.name == 'posix':
        return 'DISPLAY' in os.environ
    return True  # Assume display is available on non-posix systems


def main():
    """
    Main function to loop through all subfolders in the specified directory and play them as videos.
    """
    parser = argparse.ArgumentParser(description="Play videos from image frames organized in subfolders.")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path to the dataset's images folder (e.g., 'ahk_03122024/images')")
    parser.add_argument('--fps', type=int, default=8,
                        help='Frames per second for video playback (not used in manual mode)')
    parser.add_argument('--no_display', action='store_true',
                        help='Disable video display (useful for headless environments)')
    parser.add_argument('--filter_class', type=int, help="Filter by class ID. Only play videos matching this class ID.")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    fps = args.fps
    display = not args.no_display
    filter_class = args.filter_class

    # Validate dataset path
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(dataset_path):
        print(f"Dataset path is not a directory: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # Get all subfolders in the images directory
    subfolders = [
        os.path.join(dataset_path, d) for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]

    if not subfolders:
        print(f"No subfolders found in the dataset path: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # Sort subfolders naturally
    subfolders = natural_sort(subfolders)

    # Wrap the subfolders list with tqdm for progress visualization
    with tqdm(total=len(subfolders), desc="Processing folders", unit="folder") as pbar:
        # Loop through each subfolder and play its frames as a video
        try:
            for subfolder in subfolders:
                # Extract class label and filter by specified class
                folder_name = os.path.basename(subfolder)
                class_label = extract_class_label(folder_name)
                if filter_class is not None and class_label != str(filter_class):
                    pbar.update(1)
                    continue  # Skip folders that do not match the class filter

                status = play_video_from_frames(subfolder, fps=fps, display=display)
                if status == 'quit':
                    break  # Exit the loop and script
                # Update the progress bar after each folder
                pbar.update(1)
        except KeyboardInterrupt:
            print("\nVideo playback interrupted by user.")
            cv2.destroyAllWindows()
            sys.exit(0)

    print("Completed processing all folders.")


if __name__ == "__main__":
    main()