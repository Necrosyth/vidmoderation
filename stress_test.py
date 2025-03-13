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
                 max_concurrent_requests: int = 10,
                 total_requests: int = 100):
        self.api_url = api_url
        self.video_dir = video_dir
        self.max_concurrent_requests = max_concurrent_requests
        self.total_requests = total_requests
        self.results: List[Dict] = []
        self.start_time = None
        
        # Configure timeout and other client settings
        self.timeout = aiohttp.ClientTimeout(
            total=300,  # 5 minutes total timeout
            connect=30,  # 30 seconds connect timeout
            sock_read=60  # 60 seconds read timeout
        )
        
    async def send_video_request(self, session: aiohttp.ClientSession, 
                               video_path: str, request_id: int) -> Dict:
        """Sends a single video request to the API"""
        start_time = time.time()
        
        try:
            video_size = os.path.getsize(video_path)
            
            # Read file data
            with open(video_path, 'rb') as f:
                file_data = f.read()
            
            # Create form data
            form_data = aiohttp.FormData()
            form_data.add_field('file',
                              file_data,
                              filename=os.path.basename(video_path),
                              content_type='video/mp4')

            # Simulate realistic network delay
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # Make request with retry logic
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    async with session.post(self.api_url, data=form_data, timeout=self.timeout) as response:
                        duration = time.time() - start_time
                        response_data = await response.json() if response.status == 200 else None
                        
                        return {
                            'request_id': request_id,
                            'timestamp': datetime.now().isoformat(),
                            'video_name': os.path.basename(video_path),
                            'video_size_mb': video_size / (1024 * 1024),
                            'status_code': response.status,
                            'duration': duration,
                            'success': response.status == 200,
                            'response_data': response_data,
                            'attempt': attempt + 1
                        }
                        
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == max_retries - 1:  # Last attempt
                        raise
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Request {request_id} failed: {str(e)}")
            return {
                'request_id': request_id,
                'timestamp': datetime.now().isoformat(),
                'video_name': os.path.basename(video_path),
                'video_size_mb': video_size / (1024 * 1024),
                'status_code': 'error',
                'duration': duration,
                'success': False,
                'error': str(e),
                'attempt': attempt + 1 if 'attempt' in locals() else 1
            }

    async def run_batch(self, videos: List[str], batch_size: int) -> List[Dict]:
        """Runs a batch of concurrent requests"""
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent_requests,
            force_close=True,
            enable_cleanup_closed=True
        )
        
        async with aiohttp.ClientSession(connector=connector, timeout=self.timeout) as session:
            tasks = []
            for i in range(batch_size):
                video = random.choice(videos)
                request_id = len(self.results) + i
                tasks.append(self.send_video_request(session, video, request_id))
                
            return await asyncio.gather(*tasks, return_exceptions=True)

    def analyze_results(self) -> Dict:
        """Analyzes test results and returns statistics"""
        if not self.results:
            return {}
            
        durations = [r['duration'] for r in self.results if isinstance(r, dict)]
        successful_requests = [r for r in self.results if isinstance(r, dict) and r.get('success')]
        video_sizes = [r['video_size_mb'] for r in self.results if isinstance(r, dict)]
        
        status_codes = {}
        for result in self.results:
            if isinstance(result, dict):
                status = result['status_code']
                status_codes[status] = status_codes.get(status, 0) + 1
        
        return {
            'total_requests': len(self.results),
            'successful_requests': len(successful_requests),
            'success_rate': (len(successful_requests) / len(self.results)) * 100 if self.results else 0,
            'total_duration': time.time() - self.start_time,
            'response_times': {
                'min': min(durations) if durations else 0,
                'max': max(durations) if durations else 0,
                'mean': statistics.mean(durations) if durations else 0,
                'median': statistics.median(durations) if durations else 0,
                'p95': statistics.quantiles(durations, n=20)[-1] if len(durations) >= 20 else max(durations) if durations else 0
            },
            'video_sizes': {
                'min': min(video_sizes) if video_sizes else 0,
                'max': max(video_sizes) if video_sizes else 0,
                'mean': statistics.mean(video_sizes) if video_sizes else 0
            },
            'status_code_distribution': status_codes,
            'requests_per_second': len(self.results) / (time.time() - self.start_time) if self.start_time else 0
        }

    async def run_test(self) -> Dict:
        """Runs the complete load test"""
        videos = [os.path.join(self.video_dir, f) for f in os.listdir(self.video_dir)
                 if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not videos:
            raise ValueError("No video files found in the specified directory")
            
        logger.info(f"Starting load test with {self.total_requests} total requests, "
                   f"max {self.max_concurrent_requests} concurrent")
        
        self.start_time = time.time()
        self.results = []
        
        remaining_requests = self.total_requests
        while remaining_requests > 0:
            batch_size = min(remaining_requests, self.max_concurrent_requests)
            batch_results = await self.run_batch(videos, batch_size)
            
            # Filter out any exceptions and log them
            valid_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch request failed: {str(result)}")
                else:
                    valid_results.append(result)
            
            self.results.extend(valid_results)
            remaining_requests -= batch_size
            
            completion = ((self.total_requests - remaining_requests) / self.total_requests) * 100
            logger.info(f"Progress: {completion:.1f}% complete")
            
            # Delay between batches
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
        analysis = self.analyze_results()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'load_test_results_{timestamp}.json', 'w') as f:
            json.dump({
                'test_config': {
                    'api_url': self.api_url,
                    'total_requests': self.total_requests,
                    'max_concurrent_requests': self.max_concurrent_requests
                },
                'results': self.results,
                'analysis': analysis
            }, f, indent=2)
            
        return analysis

def print_analysis(analysis: Dict):
    """Prints the analysis results in a readable format"""
    print("\n=== Load Test Results ===")
    print(f"Total Requests: {analysis['total_requests']}")
    print(f"Success Rate: {analysis['success_rate']:.2f}%")
    print(f"Total Duration: {analysis['total_duration']:.2f} seconds")
    print(f"Requests/second: {analysis['requests_per_second']:.2f}")
    
    print("\nResponse Times (seconds):")
    print(f"  Min: {analysis['response_times']['min']:.3f}")
    print(f"  Max: {analysis['response_times']['max']:.3f}")
    print(f"  Mean: {analysis['response_times']['mean']:.3f}")
    print(f"  Median: {analysis['response_times']['median']:.3f}")
    print(f"  95th percentile: {analysis['response_times']['p95']:.3f}")
    
    print("\nVideo Sizes (MB):")
    print(f"  Min: {analysis['video_sizes']['min']:.2f}")
    print(f"  Max: {analysis['video_sizes']['max']:.2f}")
    print(f"  Mean: {analysis['video_sizes']['mean']:.2f}")
    
    print("\nStatus Code Distribution:")
    for status, count in analysis['status_code_distribution'].items():
        print(f"  {status}: {count} requests")

async def main():
    # Create tester instance with configuration
    tester = VideoLoadTester(
        api_url="http://localhost:8000/analyze",
        video_dir="test_videos",
        max_concurrent_requests=5,  # Reduced from 10 to prevent overload
        total_requests=100
    )
    
    try:
        analysis = await tester.run_test()
        print_analysis(analysis)
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())