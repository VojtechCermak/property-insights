import os
import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from dotenv import load_dotenv
from PIL import Image

# Get logger from parent module
logger = logging.getLogger(__name__)

class SrealityScraper:
    def __init__(self):
        load_dotenv()
        self.base_url = "https://www.sreality.cz"
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.search_url = f"{self.base_url}/hledani/prodej/byty/praha"


    async def setup_playwright(self):
        """Setup Playwright browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.firefox.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled']
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            storage_state=None
        )
        self.page = await self.context.new_page()


    async def handle_consent_dialog(self):
        """Handle the Seznam consent dialog and store consent cookies"""
        try:
            # Wait for the dialog to appear
            await self.page.wait_for_selector(".szn-cmp-dialog-container", timeout=10000)
            
            # Try different selectors for the button
            button_selectors = [
                "button[data-testid='cw-button-agree-with-ads']",
                ".cw-btn--green",
                "text=Souhlasím"
            ]
            
            button_found = False
            for selector in button_selectors:
                try:
                    button = await self.page.wait_for_selector(selector, timeout=5000)
                    if button:
                        await button.click()
                        button_found = True
                        break
                except Exception as e:
                    continue
            
            if not button_found:
                logger.error("Could not find or click the consent button")
                return False
            
            # Wait for the dialog to disappear
            await self.page.wait_for_selector(".szn-cmp-dialog-container", state="hidden", timeout=10000)
            
            # Store the consent cookies
            cookies = await self.page.context.cookies()
            await self.context.add_cookies(cookies)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling consent dialog: {e}")
            return False


    def _parse_next_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Parse the Next.js data from the page"""
        next_data_script = soup.find('script', {'id': '__NEXT_DATA__'})
        if not next_data_script:
            raise ValueError("Next.js data script not found")
        
        try:
            return json.loads(next_data_script.string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing Next.js data: {e}")


    def _extract_property_data(self, next_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract property data from the Next.js data structure"""
        property_data = next_data.get('props', {}).get('pageProps', {})
        dehydrated_data = property_data.get('dehydratedState', {}).get('queries', [])
        
        if not dehydrated_data:
            raise ValueError("No dehydrated data found")
        
        return dehydrated_data[-1].get('state', {}).get('data', {})


    def _process_image(self, img_data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Process a single image from the property data"""
        try:
            img_url = img_data.get('url')
            if not img_url:
                return None
            
            # Ensure URL has proper protocol and add required parameters
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            
            # Add required parameters for image access
            img_url = f"{img_url}?fl=res,800,800,1|shr,,20|webp,60"
            
            # Download image
            response = requests.get(img_url)
            if response.status_code != 200:
                logger.error(f"Failed to download image {idx}: HTTP {response.status_code}")
                return None
            
            # Convert to PIL Image
            image = Image.open(BytesIO(response.content))
            
            return {
                'image': image,
                'width': img_data.get('width'),
                'height': img_data.get('height'),
                'order': img_data.get('order'),
                'url': img_url
            }
            
        except Exception as e:
            logger.error(f"Error processing image {idx}: {e}")
            return None


    def _process_images(self, property_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process all images from the property data"""
        images = []
        for img_data in property_info.get('images', []):
            processed_image = self._process_image(img_data, len(images))
            if processed_image:
                images.append(processed_image)
        return images


    def _get_property_folder_name(self, url: str) -> str:
        """Convert URL to a valid folder name with property details"""
        # Example URL: https://www.sreality.cz/detail/prodej/byt/4+1/praha-4-krc/123456789
        parts = url.split('/')
        property_id = parts[-1]
        try:
            detail_index = parts.index('detail')
            property_details = '_'.join(parts[detail_index + 1:-1])
        except ValueError:
            property_details = 'unknown'
        
        # Combine and sanitize
        folder_name = f"property_{property_id}_{property_details}"
        folder_name = folder_name.encode('ascii', 'ignore').decode('ascii')
        folder_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in folder_name)
        
        return folder_name


    def _property_exists(self, url: str) -> bool:
        """Check if property data already exists"""
        folder_name = self._get_property_folder_name(url)
        property_path = os.path.join('data', folder_name)
        return os.path.exists(property_path)


    def _store_property_data(self, url: str, data: Dict[str, Any], images: List[Dict[str, Any]]) -> None:
        """Store property data in a structured way"""
        folder_name = self._get_property_folder_name(url)
        property_path = os.path.join('data', folder_name)
        os.makedirs(property_path, exist_ok=True)
        
        # Store JSON data
        json_path = os.path.join(property_path, 'data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Store images
        for idx, img_data in enumerate(images):
            img_path = os.path.join(property_path, f'image_{idx}.jpg')
            img_data['image'].save(img_path, 'JPEG')


    async def get_property_details(self, url: str) -> Dict[str, Any]:
        """Fetch detailed information from a property page"""
        try:
            await self.page.goto(url)
            await self.page.wait_for_load_state("networkidle")
            
            # Get page content and parse with BeautifulSoup
            content = await self.page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Parse and process the data
            next_data = self._parse_next_data(soup)
            property_info = self._extract_property_data(next_data)
            images = self._process_images(property_info)
            
            return {
                'url': url,
                'timestamp': datetime.now().isoformat(),
                'data': property_info,
                'images': images
            }
            
        except ValueError as e:
            logger.error(str(e))
            return {'url': url, 'timestamp': datetime.now().isoformat(), 'error': str(e)}
        except Exception as e:
            logger.error(f"Error fetching property details from {url}: {e}")
            return {'url': url, 'timestamp': datetime.now().isoformat(), 'error': str(e)}


    async def get_property_listings(self, page: int = 1) -> List[str]:
        """Fetch property listing URLs from a specific page"""
        url = f"{self.search_url}?strana={page}&vlastnictvi=osobni&cena-od=1000000"
        
        try:
            await self.page.goto(url)
            await self.page.wait_for_load_state("networkidle")
            
            content = await self.page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find all property listing links
            property_links = soup.find_all('a', href=lambda x: x and '/detail/prodej/byt/' in x)
            return [self.base_url + link['href'] for link in property_links]
            
        except Exception as e:
            logger.error(f"Failed to fetch page {page}: {e}")
            return []


    async def scrape(self, pages: List[int]):
        """Main scraping function"""
        try:
            await self.setup_playwright()
            
            # Initialize session
            await self.page.goto(self.search_url)
            await self.page.wait_for_load_state("networkidle")
            await self.handle_consent_dialog()
            os.makedirs('data', exist_ok=True)
            
            # Process pages
            for page in pages:
                urls = await self.get_property_listings(page)
                logger.info(f"Found {len(urls)} properties on page {page}")
                
                # Process properties
                for url in urls:
                    logger.info(f"Processing property details from {url}")
                    if self._property_exists(url):
                        logger.info(f"Already exists - skipping")
                        continue
                    
                    try:
                        result = await self.get_property_details(url)
                        if 'error' in result:
                            logger.error(f"Error processing {url}: {result['error']}")
                            continue
                        
                        self._store_property_data(url, result['data'], result['images'])
                        logger.info(f"Success")
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Failed to process {url}: {e}")
                        continue
                
                await asyncio.sleep(2)
            
        finally:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop() 
