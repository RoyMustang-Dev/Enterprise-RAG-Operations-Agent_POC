import asyncio
import time
from app.ingestion.crawler_service import CrawlerService

async def run_benchmark():
    TARGET_URL = "https://learnnect.com"
    print(f"Initializing Aeko Crawler for {TARGET_URL} at Depth 3...")
    crawler = CrawlerService()
    
    start_time = time.perf_counter()
    result = await crawler.crawl_url(
        url=TARGET_URL,
        save_folder="./crawler_benchmark_results",
        simulate=False,
        recursive=True,
        max_depth=4
    )
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    print("\n" + "="*50)
    print("CRAWL COMPLETE")
    print("="*50)
    print(f"Total Time Taken : {total_time:.2f} seconds")
    print(f"Pages Crawled    : {result.get('pages_crawled', 0)}")
    print(f"Allowed Links    : {len(result.get('links', {}).get('allowed', []))}")
    print(f"Status           : {result.get('status')}")
    print("="*50)

if __name__ == "__main__":
    # Playwright requires the default ProactorEventLoop on Windows
    # Removing the SelectorEventLoopPolicy override that caused the crash.
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(run_benchmark())
