import asyncio
import aiohttp
import os
import random
import time
from datetime import datetime
from typing import List, Dict
import logging
import statistics
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoLoadTester:
    def __init__(self, 
                 api_url: str = "http://localhost:8000/analyze",
                 video_dir: str = "test_videos",
                 max_concurrent_requests: int = 4,  # Match server capacity
                 total_requests: int = 100):
        self.api_url = api_url
        self.video_dir = video_dir
        self.max_concurrent_requests = max_concurrent_requests
        self.total_requests = total_requests
        self.results: List[Dict] = []
        self.start_time = None
        
        # Configure timeout matching AWS ALB settings
        self.timeout = aiohttp.ClientTimeout(
            total=300,  # 5 minutes total timeout
            connect=10,  # 10 seconds connect timeout
            sock_read=30  # 30 seconds read timeout
        )
        
    async def send_video_request(self, session: aiohttp.ClientSession, 
                               video_path: str, request_id: int) -> Dict:
        """Sends a single video request to the API with realistic delays"""
        start_time = time.time()
        try:
            video_size = os.path.getsize(video_path)
            
            # Read file data with random start delay (0.1-0.5s)
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            with open(video_path, 'rb') as f:
                file_data = f.read()
            
            # Create form data
            form_data = aiohttp.FormData()
            form_data.add_field('file',
                              file_data,
                              filename=os.path.basename(video_path),
                              content_type='video/mp4')

            # Simulate realistic network conditions
            async with session.post(self.api_url, data=form_data, timeout=self.timeout) as response:
                duration = time.time() - start_time
                response_data = await response.json() if response.status == 200 else None
                
                return {
                    'request_id': request_id,
                    'status_code': response.status,
                    'duration': duration,
                    'success': response.status == 200,
                    'response_data': response_data,
                    'video_size_mb': video_size / (1024 * 1024)
                }
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Request {request_id} failed: {str(e)}")
            return {
                'request_id': request_id,
                'status_code': 'error',
                'duration': duration,
                'success': False,
                'error': str(e),
                'video_size_mb': os.path.getsize(video_path) / (1024 * 1024)
            }

    async def run_test(self) -> Dict:
        """Runs the complete load test with realistic pacing"""
        videos = [os.path.join(self.video_dir, f) for f in os.listdir(self.video_dir)
                 if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not videos:
            raise ValueError("No test videos found in directory")
            
        logger.info(f"Starting AWS-like load test with {self.total_requests} requests")
        
        self.start_time = time.time()
        self.results = []
        
        connector = aiohttp.TCPConnector(limit=self.max_concurrent_requests)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for i in range(self.total_requests):
                video = random.choice(videos)
                tasks.append(self.send_video_request(session, video, i))
                
                # Add realistic request spacing
                if i % self.max_concurrent_requests == 0:
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                
                # Throttle task creation
                if len(tasks) >= self.max_concurrent_requests * 2:
                    results = await asyncio.gather(*tasks)
                    self.results.extend(results)
                    tasks = []
                    
            # Gather remaining tasks
            if tasks:
                results = await asyncio.gather(*tasks)
                self.results.extend(results)
            
        return self.analyze_results()

    def analyze_results(self) -> Dict:
        """Generates AWS-focused performance metrics"""
        successful = [r for r in self.results if r['success']]
        return {
            'total_requests': len(self.results),
            'success_rate': len(successful)/len(self.results)*100,
            'throughput_rps': len(self.results)/(time.time()-self.start_time),
            'response_times': {
                'p50': statistics.quantiles([r['duration'] for r in successful], n=4)[1],
                'p95': statistics.quantiles([r['duration'] for r in successful], n=20)[-1],
                'max': max(r['duration'] for r in successful)
            },
            'error_distribution': {
                k: sum(1 for r in self.results if r.get('status_code') == k)
                for k in {r.get('status_code') for r in self.results}
            }
        }

async def main():
    tester = VideoLoadTester(
        api_url="http://localhost:8000/analyze",
        max_concurrent_requests=4,  # Match server concurrency
        total_requests=100
    )
    
    try:
        results = await tester.run_test()
        print(json.dumps(results, indent=2))
    except Exception as e:
        logger.error(f"Load test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 