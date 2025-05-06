import logging
import argparse
from processors import JSONProcessor, ImageProcessor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_json_processor():
    logger.info("Starting data processing...")
    processor = JSONProcessor(data_dir="../scraper/data", output_dir="processed")
    df = processor.process_all_properties()
    logger.info(f"Processed {len(df)} properties")
    logger.info("Data processing completed")


def run_image_processor():
    logger.info("Starting image processing...")
    processor = ImageProcessor(data_dir="../scraper/data", output_dir="processed")
    processor.process_all_images()
    logger.info("Image processing completed")


def main():
    parser = argparse.ArgumentParser(description="Data and Image Processor")
    parser.add_argument('--method', type=str, choices=['json', 'image'], help='Run JSON or Image processor')
    args = parser.parse_args()
    if args.method == 'image':
        run_image_processor()
    elif args.method == 'json':
        run_json_processor()
    else:
        raise ValueError("Invalid method")


if __name__ == "__main__":
    main() 