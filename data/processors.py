import re
import json
import logging
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import torch
import timm
from PIL import Image, ImageFile
import numpy as np
import torchvision.transforms as T
import os
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
# Get logger from parent module
logger = logging.getLogger(__name__)


class JSONProcessor:
    def __init__(self, data_dir: str = '../scraper/data', output_dir: str = 'processed'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_property_data(self, property_dir: Path) -> Dict[str, Any]:
        """Load property data from JSON file"""
        json_path = property_dir / 'data.json'
        if not json_path.exists():
            logger.warning(f"No data.json found in {property_dir}")
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {json_path}: {e}")
            return None


    def process_all_properties(self) -> pd.DataFrame:
        """Process all property data and create a DataFrame"""
        metadata_list = []
        
        # Get all property directories
        property_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('property_')]
        logger.info(f"Found {len(property_dirs)} property directories")
        
        # Process each property
        for dir in property_dirs:
            data = self._load_property_data(dir)
            if not data:
                continue
                
            metadata = parse_property_meta(data, id = dir.name)
            if metadata:
                metadata_list.append(metadata)
        
        # Create DataFrame
        df = pd.DataFrame(metadata_list)
        df = cleaning_data(df)
        
        # Save to CSV
        csv_path = self.output_dir / 'metadata.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Saved metadata to {csv_path}")
        return df 
    

def parse_property_meta(data, id):
    locality = data.get("locality", {}) or {}
    premise = data.get("premise", {}) or {}
    nearest = data.get("nearest", []) or []
    params = data.get("params", {}) or {}
    
    features = {
        "id": id,

        # Category info
        "category_main": data.get("categoryMainCb", {}).get("name", None),
        "category_sub": data.get("categorySubCb", {}).get("name", None),
        "category_type": data.get("categoryTypeCb", {}).get("name", None),

        # Textual data
        "description": re.sub(r'\s+', ' ', data.get("description", '')).strip(),

        # Location data
        "latitude": locality.get("latitude", None),
        "longitude": locality.get("longitude", None),
        "city": locality.get("city", None),
        "city_part": locality.get("cityPart", None),
        "district": locality.get("district", None),
        "region": locality.get("region", None),

        # Price info
        "price": data.get("priceCzk", None),
        "price_per_sqm": data.get("priceCzkPerSqM", None),
        "currency": data.get("priceCurrencyCb", {}).get("name", None),
        "price_unit": data.get("priceUnitCb", {}).get("name", None),

        # Premise info
        "premise_name": premise.get("name", None),
        "premise_review_count": premise.get("reviewCount", None),
        "premise_review_score": premise.get("reviewScore", None),

        # Property params
        'costOfLiving': params.get('costOfLiving', ''),
        'acceptanceYear': params.get('acceptanceYear', None),
        'balcony': params.get('balcony', False),
        'balconyArea': params.get('balconyArea', None),
        'basin': params.get('basin', False),
        'basinArea': params.get('basinArea', None),
        'buildingArea': params.get('buildingArea', None),
        'buildingCondition': params.get('buildingCondition', {}).get('name', ''),
        'buildingType': params.get('buildingType', {}).get('name', ''),
        'cellar': params.get('cellar', False),
        'cellarArea': params.get('cellarArea', None),
        'easyAccess': params.get('easyAccess', {}).get('name', ''),
        'edited': params.get('edited', ''),
        'electricitySet': [e.get('name') for e in params.get('electricitySet', [])],
        'elevator': params.get('elevator', {}).get('name', ''),
        'energyEfficiencyRating': params.get('energyEfficiencyRating', {}).get('name', ''),
        'energyPerformanceCertificate': params.get('energyPerformanceCertificate', {}).get('name', ''),
        'floorArea': params.get('floorArea', None),
        'floorNumber': params.get('floorNumber', None),
        'floors': params.get('floors', None),
        'furnished': params.get('furnished', {}).get('name', ''),
        'garage': params.get('garage', False),
        'gardenArea': params.get('gardenArea', None),
        'garret': params.get('garret', False),
        'gullySet': [g.get('name') for g in params.get('gullySet', [])],
        'heatingSet': [h.get('name') for h in params.get('heatingSet', [])],
        'internetConnectionProvider': params.get('internetConnectionProvider', ''),
        'internetConnectionSpeed': params.get('internetConnectionSpeed', None),
        'loggia': params.get('loggia', False),
        'loggiaArea': params.get('loggiaArea', None),
        'lowEnergy': params.get('lowEnergy', False),
        'objectAge': params.get('objectAge', None),
        'objectLocation': params.get('objectLocation', {}).get('name', ''),
        'ownership': params.get('ownership', {}).get('name', ''),
        'parking': params.get('parking', None),
        'parkingLots': params.get('parkingLots', False),
        'roadTypeSet': [r.get('name') for r in params.get('roadTypeSet', [])],
        'since': params.get('since', ''),
        'stateCb': params.get('stateCb', {}).get('name', ''),
        'stats': params.get('stats', None),
        'surroundingsType': params.get('surroundingsType', {}).get('name', ''),
        'telecommunicationSet': [t.get('name') for t in params.get('telecommunicationSet', [])],
        'terrace': params.get('terrace', False),
        'terraceArea': params.get('terraceArea', None),
        'transportSet': [t.get('name') for t in params.get('transportSet', [])],
        'undergroundFloors': params.get('undergroundFloors', None),
        'usableArea': params.get('usableArea', None),
        'waterSet': [w.get('name') for w in params.get('waterSet', [])],
        'priceFlagNegotiationCb': params.get('priceFlagNegotiationCb', False),
        'priceNote': params.get('priceNote', '')
    }

    # Loop through each category of nearest places (hospital, school, etc.)
    for place in nearest:
        name = place.get("name", None)
        distance = place.get('distance', None)
        features[f'nearest_{name}'] = distance

    return features


def cleaning_data(df):
    # Filtering rows
    df = df[df['price_unit'] == 'za nemovitost']
    df = df[df['currency'] == 'Kč']
    df = df[df['category_main'] == 'Byty']
    df = df[df['category_type'] == 'Prodej']
    df = df[df['city'] == 'Praha']
    df = df[df['region'] == 'Hlavní město Praha']
    df = df[df['ownership'] == 'Osobní']

    # Filtering columns
    drop_cols = [
        # Text
        'priceNote',
        'description',
    
        # Maybe interesting, drop for now
        'edited', 
    
        # Used as filters
        'category_type',
        'category_main',
        'city',
        'region',
        'currency',
        'price_unit',
        'ownership',
    
        # # Not informative - Low variance or missing
        'internetConnectionSpeed', # ~90% missing
        'internetConnectionProvider', # ~90% missing
        'buildingArea', # ~90% missing
        'costOfLiving', # ~90% missing
        'telecommunicationSet', #65% missing
        'waterSet', # ~50% missing
        'transportSet', # 50% missing
        'roadTypeSet', # 65% missing
        'heatingSet', # 65% missing
        'gullySet', # 50% missing, remaining are in 1 category
        'garret', # 50% missing, remaining are in 1 category
        'surroundingsType', # 65% missing
        'stateCb', # 97% have the same value
        'electricitySet', # 60% missing
        'easyAccess', # 60% missing
    
        # Duplicates - have integer with area
        'balcony',
        'cellar',
        'terrace',
        'loggia',
        'basin',
    ]
    df =  df.drop(drop_cols, axis=1)

    # Extra area

    # Months since listing
    dates = pd.to_datetime(df['since'])
    reference = pd.to_datetime('2025-05-01')
    df['sinceMonths'] = (reference.year - dates.dt.year) * 12 + (reference.month - dates.dt.month)

    # Object age - categorical (lot of nans)
    bins = [-np.inf, 0, 1, 5, 10, 20, 30, 50, np.inf]
    df['objectAge'] = pd.cut(df['objectAge'], bins=bins).astype(str)
    
    # Acceptance Year - categorical (lot of nans)
    bins = [-np.inf, 0, 1, 5, 10, 20, 30, 50, np.inf]
    df['acceptanceYear'] = pd.cut(df['acceptanceYear'], bins=bins).astype(str)
    df['priceFlagNegotiationCb'] = df['priceFlagNegotiationCb'].fillna(False)

    # Is basement
    df['isBasement'] = df['floorNumber'] <= 0

    # Fix elevator
    df['elevator'] = df['elevator'].map({'- nezadáno': False, 'Ano': True, 'Ne': False})
    return df


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transforms):
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        file_path = self.image_paths[idx]
        img = Image.open(file_path).convert('RGB')
        img = self.transforms(img)
        return img


class ImageProcessor:
    def __init__(
        self,
        data_dir: str = '../scraper/data',
        output_dir: str = 'processed',
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        batch_size: int = 4,
        num_workers: int = 4,
        device: str = "cuda"
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(model_name, pretrained=True).eval().to(self.device)
        self.transforms = T.Compose([
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def process_all_images(self):
        # Gather all image paths and IDs
        property_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('property_')]
        image_paths = []
        property_ids = []
        for prop in property_dirs:
            for f in prop.iterdir():
                if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    image_paths.append(f)
                    property_ids.append(f"{prop.name}")

        dataset = ImageDataset(image_paths, self.transforms)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

        features = []
        for imgs in tqdm(loader, desc="Extracting Features"):
            with torch.no_grad():
                features.append(self.model(imgs.to(self.device)).cpu())
        features = torch.cat(features).numpy()

        result = {'features': features, 'property_ids': property_ids}
        path = self.output_dir / "image_features.pkl"
        with open(path, 'wb') as f:
            pickle.dump(result, f)
        logging.info(f"Image processing completed. Saved to {path}")