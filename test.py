import asyncio
import aiohttp
import os
from typing import List

# Configuration
API_URL = "http://localhost:8000/analyze"  # Adjust this to your server's address
VIDEO_DIR = "uploaded_videos"  # Directory where your test videos are stored
CONCURRENT_REQUESTS = 10  # Number of concurrent requests. Adjust based on your system's capacity or test needs.

async def send_video(session: aiohttp.ClientSession, video_path: str):
    try:
        with open(video_path, 'rb') as video_file:
            async with session.post(API_URL, data={'file': video_file}) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"Video {os.path.basename(video_path)} processed: {result['status']}")
                else:
                    print(f"Failed to process {os.path.basename(video_path)}. Status: {response.status}")
    except Exception as e:
        print(f"An error occurred with {os.path.basename(video_path)}: {str(e)}")

async def stress_test(videos: List[str]):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for video in videos:
            tasks.append(asyncio.create_task(send_video(session, video)))
            if len(tasks) >= CONCURRENT_REQUESTS:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:  # Handle any remaining tasks if they're less than CONCURRENT_REQUESTS
            await asyncio.gather(*tasks)

def main():
    # Gather all video files
    videos = [os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not videos:
        print("No video files found in the specified directory.")
        return

    # Limit to a reasonable number if you have too many videos
    videos = videos[:1000]  # Example: Limit to 1000 videos for testing

    # Run the stress test
    asyncio.run(stress_test(videos))

if __name__ == "__main__":
    main()