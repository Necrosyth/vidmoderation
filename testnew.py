import asyncio
import aiohttp
import os
import random
from typing import List
import time
from collections import defaultdict
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_URL = "http://localhost:8000/analyze"
VIDEO_DIR = "uploaded_videos"
BATCH_SIZE = 5  # Reduced for testing
MAX_BATCHES = 3  # Reduced for testing
WAIT_BETWEEN_BATCHES = 2

async def send_video(session: aiohttp.ClientSession, video_path: str, request_id: int):
    logger.info(f"Starting request {request_id} with video: {os.path.basename(video_path)}")
    start_time = time.time()
    
    try:
        # Check if file exists and is readable
        if not os.path.exists(video_path):
            logger.error(f"File not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        file_size = os.path.getsize(video_path)
        logger.info(f"File size for request {request_id}: {file_size} bytes")

        # Open and send file
        with open(video_path, 'rb') as video_file:
            form = aiohttp.FormData()
            form.add_field('file', 
                         video_file,
                         filename=os.path.basename(video_path),
                         content_type='video/mp4')
            
            logger.info(f"Sending request {request_id} to server...")
            async with session.post(API_URL, data=form, timeout=300) as response:
                logger.info(f"Got response for request {request_id}: {response.status}")
                response_text = await response.text()
                logger.info(f"Response text for request {request_id}: {response_text[:200]}")
                
                duration = time.time() - start_time
                return {
                    'request_id': request_id,
                    'status': response.status,
                    'duration': duration,
                    'success': response.status == 200,
                    'video': os.path.basename(video_path)
                }
    except asyncio.TimeoutError:
        logger.error(f"Timeout for request {request_id}")
        return {
            'request_id': request_id,
            'status': 'timeout',
            'duration': time.time() - start_time,
            'success': False,
            'error': 'Request timed out',
            'video': os.path.basename(video_path)
        }
    except Exception as e:
        logger.error(f"Error in request {request_id}: {str(e)}")
        return {
            'request_id': request_id,
            'status': 'error',
            'duration': time.time() - start_time,
            'success': False,
            'error': str(e),
            'video': os.path.basename(video_path)
        }

async def run_batch(videos: List[str], batch_num: int):
    logger.info(f"\nStarting batch {batch_num + 1}/{MAX_BATCHES}")
    test_videos = [random.choice(videos) for _ in range(BATCH_SIZE)]
    
    logger.info(f"Selected videos for batch {batch_num + 1}:")
    for i, video in enumerate(test_videos):
        logger.info(f"  {i+1}. {os.path.basename(video)}")
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_video(session, video, i + (batch_num * BATCH_SIZE))
            for i, video in enumerate(test_videos)
        ]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            results = [r for r in results if not isinstance(r, Exception)]
        except Exception as e:
            logger.error(f"Error in batch {batch_num + 1}: {str(e)}")
            results = []
    
    duration = time.time() - start_time
    success_count = sum(1 for r in results if r['success'])
    
    logger.info(f"Batch {batch_num + 1} complete:")
    logger.info(f"Success rate: {success_count}/{BATCH_SIZE}")
    logger.info(f"Batch duration: {duration:.2f}s")
    
    return results

async def main():
    logger.info("Starting load test...")
    
    # List and validate videos
    videos = [os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) 
             if f.endswith(('.mp4', '.avi', '.mov'))]
    
    valid_videos = []
    for video in videos:
        try:
            if os.path.exists(video) and os.path.getsize(video) > 0:
                valid_videos.append(video)
            else:
                logger.warning(f"Skipping invalid video: {video}")
        except Exception as e:
            logger.error(f"Error checking video {video}: {str(e)}")
    
    if not valid_videos:
        logger.error("No valid video files found in the specified directory.")
        return
        
    logger.info(f"Found {len(valid_videos)} valid videos")
    
    # Run batches
    all_results = []
    for batch in range(MAX_BATCHES):
        try:
            batch_results = await run_batch(valid_videos, batch)
            all_results.extend(batch_results)
            
            if batch < MAX_BATCHES - 1:
                logger.info(f"Waiting {WAIT_BETWEEN_BATCHES} seconds before next batch...")
                await asyncio.sleep(WAIT_BETWEEN_BATCHES)
        except Exception as e:
            logger.error(f"Error in batch {batch + 1}: {str(e)}")
    
    # Final analysis
    if all_results:
        total_success = sum(1 for r in all_results if r['success'])
        response_times = [r['duration'] for r in all_results]
        
        logger.info("\nFinal Results:")
        logger.info(f"Total Requests: {len(all_results)}")
        logger.info(f"Success Rate: {(total_success/len(all_results))*100:.2f}%")
        logger.info(f"Response Times:")
        logger.info(f"Min: {min(response_times):.3f}s")
        logger.info(f"Max: {max(response_times):.3f}s")
        logger.info(f"Average: {np.mean(response_times):.3f}s")
        
        status_counts = defaultdict(int)
        for r in all_results:
            status_counts[r['status']] += 1
        
        logger.info("\nStatus Code Distribution:")
        for status, count in status_counts.items():
            logger.info(f"Status {status}: {count} requests")
    else:
        logger.error("No results were collected during the test.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")