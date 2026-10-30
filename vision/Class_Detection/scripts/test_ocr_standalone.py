import cv2
import os
import sys

# Add the parent directory to sys.path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ocr_anchor import OCRAnchorDetector

def test_ocr_standalone(video_path: str, bbox: tuple = None, interval_sec: int = 5):
    """
    Test the OCR text detection module independently.
    
    :param video_path: Path to the input video.
    :param bbox: Coordinates of the screen region (x1, y1, x2, y2). If None, uses the full frame.
    :param interval_sec: How often to sample a frame for OCR testing.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    print("Initializing OCR Model (first load may be slow)...")
    # Using a low threshold to see all outputs
    detector = OCRAnchorDetector(change_threshold=0.01)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25 # Fallback
        
    frame_count = 0
    
    print(f"Starting test on video: {video_path}")
    
    # Create artifacts directory for debug outputs if it doesn't exist
    artifacts_dir = os.path.join(os.path.dirname(video_path), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_sec = frame_count / fps
        if frame_count % int(fps * interval_sec) == 0:
            print(f"\n--- Analyzing frame at {current_sec:.1f}s ---")
            
            if bbox:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                print(f"Cropped to bounding box: {bbox}")
            else:
                crop = frame
                print("Using full frame.")
                
            # Save the crop for visual inspection
            debug_img_path = os.path.join(artifacts_dir, f"ocr_debug_{current_sec:.1f}s.jpg")
            cv2.imwrite(debug_img_path, crop)
            
            text = detector.detect_change(crop)
            
            if text:
                print(f"✅ Text Detected: {text}")
            else:
                print(f"❌ No valid text or no significant change detected.")
                
        frame_count += 1

    cap.release()
    print("\nTest finished.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone test for PaddleOCR on a video.")
    default_video = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test.mp4")
    
    parser.add_argument("--source", type=str, default=default_video, help="Path to the input video file.")
    parser.add_argument("--bbox", type=str, default=None, help="Bounding box of the screen in format 'x1,y1,x2,y2' (e.g., '100,50,800,600').")
    parser.add_argument("--interval", type=int, default=5, help="Interval in seconds between processed frames.")
    
    args = parser.parse_args()
    
    ppt_bbox = None
    if args.bbox:
        try:
            # Parse 'x1,y1,x2,y2' into a tuple of ints
            ppt_bbox = tuple(map(int, args.bbox.strip().split(',')))
            if len(ppt_bbox) != 4:
                raise ValueError
        except Exception:
            print("Error: --bbox must be in the format 'x1,y1,x2,y2', e.g., '100,50,800,600'")
            sys.exit(1)
            
    test_ocr_standalone(args.source, bbox=ppt_bbox, interval_sec=args.interval)
