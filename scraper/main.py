import os
import logging
import asyncio
from datetime import datetime
from scraper import SrealityScraper

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Create a unique log filename with timestamp
log_filename = f"scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(log_dir, log_filename)

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting scraper...")
    scraper = SrealityScraper()
    try:
        await scraper.scrape(pages=list(range(1, 250)))
        logger.info("Scraping completed successfully")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())