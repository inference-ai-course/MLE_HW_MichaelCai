"""
YouTube NLP Conference Talk Transcriber

Fetches YouTube audio using yt-dlp, extracts frames for OCR-based transcription,
and saves timestamped transcripts in JSONL format.
"""

import yt_dlp
import cv2
import pytesseract
import json
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import tempfile
import subprocess
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptSegment:
    """Data class for transcript segments with timestamps"""
    start_time: float
    end_time: float
    text: str
    confidence: Optional[float] = None

class YouTubeTranscriber:
    """YouTube video transcriber with OCR capabilities"""
    
    def __init__(self, output_dir: str = "transcripts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # yt-dlp configuration
        self.ydl_opts = {
            'format': 'best[height<=720]',  # Reasonable quality for OCR
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'writesubtitles': True,
            'writeautomaticsub': True,
        }
    
    def download_video(self, url: str) -> Optional[str]:
        """Download YouTube video and return local path"""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Extract video info first
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'unknown')
                duration = info.get('duration', 0)
                
                logger.info(f"Video: {title}, Duration: {duration}s")
                
                # Download video
                ydl.download([url])
                
                # Find downloaded video file
                video_files = [f for f in os.listdir(self.output_dir) 
                             if f.startswith(title.replace('/', '_')[:50])]
                
                if video_files:
                    return os.path.join(self.output_dir, video_files[0])
                
        except Exception as e:
            logger.error(f"Error downloading video {url}: {e}")
            return None
    
    def extract_frames(self, video_path: str, interval: int = 5) -> List[str]:
        """Extract frames from video at specified intervals"""
        frame_paths = []
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            logger.info(f"Video FPS: {fps}, Duration: {duration}s")
            
            # Extract frames every `interval` seconds
            frame_interval = int(fps * interval)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    frame_filename = f"frame_{timestamp:.1f}s.png"
                    frame_path = os.path.join(self.output_dir, frame_filename)
                    
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append((frame_path, timestamp))
                    
                frame_count += 1
            
            cap.release()
            return frame_paths
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []
    
    def ocr_frame(self, frame_path: str) -> str:
        """Extract text from frame using Tesseract OCR"""
        try:
            # Load image
            image = Image.open(frame_path)
            
            # Convert to grayscale for better OCR
            image = image.convert('L')
            
            # Enhance image for OCR
            image_array = np.array(image)
            
            # Apply threshold to make text clearer
            _, threshold_image = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)
            
            # OCR configuration for better text detection
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()\'" -'
            
            # Extract text
            text = pytesseract.image_to_string(
                Image.fromarray(threshold_image), 
                config=custom_config
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_path}: {e}")
            return ""
    
    def transcribe_video(self, url: str, frame_interval: int = 5) -> List[TranscriptSegment]:
        """Transcribe video using OCR on extracted frames"""
        segments = []
        
        # Download video
        video_path = self.download_video(url)
        if not video_path:
            return segments
        
        try:
            # Extract frames
            frames = self.extract_frames(video_path, frame_interval)
            
            # Process each frame
            for i, (frame_path, timestamp) in enumerate(frames):
                text = self.ocr_frame(frame_path)
                
                if text:  # Only add non-empty text
                    start_time = timestamp
                    end_time = timestamp + frame_interval if i < len(frames) - 1 else timestamp + 1
                    
                    segment = TranscriptSegment(
                        start_time=start_time,
                        end_time=end_time,
                        text=text
                    )
                    segments.append(segment)
                
                # Clean up frame file
                os.remove(frame_path)
            
            # Clean up video file
            os.remove(video_path)
            
        except Exception as e:
            logger.error(f"Error transcribing video: {e}")
        
        return segments
    
    def save_jsonl(self, segments: List[TranscriptSegment], filename: str):
        """Save transcript segments to JSONL file"""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for segment in segments:
                json_obj = {
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'text': segment.text,
                    'confidence': segment.confidence
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(segments)} segments to {filepath}")
    
    def process_video_list(self, video_urls: List[str]):
        """Process multiple YouTube videos"""
        for i, url in enumerate(video_urls, 1):
            logger.info(f"Processing video {i}/{len(video_urls)}: {url}")
            
            try:
                # Transcribe video
                segments = self.transcribe_video(url)
                
                if segments:
                    # Generate filename from URL or index
                    filename = f"nlp_talk_{i:02d}.jsonl"
                    self.save_jsonl(segments, filename)
                else:
                    logger.warning(f"No transcription generated for video {i}")
                    
            except Exception as e:
                logger.error(f"Error processing video {i}: {e}")

def main():
    """Main function with sample NLP conference talk URLs"""
    
    # Sample NLP conference talk URLs (replace with actual URLs)
    nlp_conference_urls = [
        # Add 10 short NLP conference talk URLs here
        # "https://www.youtube.com/watch?v=VIDEO_ID1",
        # "https://www.youtube.com/watch?v=VIDEO_ID2",
        # ... (add more URLs)
        "https://www.youtube.com/shorts/kKm_0eLmbzQ",
        "https://www.youtube.com/shorts/dWQU3rxjres",
        "https://www.youtube.com/shorts/_zcyylNuX9s"
    ]
    
    # For demonstration, using placeholder URLs
    if not nlp_conference_urls:
        logger.warning("No video URLs provided. Please add YouTube URLs to nlp_conference_urls list.")
        nlp_conference_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Example URL
        ]
    
    # Create transcriber
    transcriber = YouTubeTranscriber(output_dir="nlp_transcripts")
    
    # Process videos
    transcriber.process_video_list(nlp_conference_urls)
    
    logger.info("Transcription completed!")

if __name__ == "__main__":
    main()