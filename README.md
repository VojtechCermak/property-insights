# Prague Real Estate Price

This project aims to create a system for prediction of residential property prices in Prague.

## Project Structure

- `scraper/` - Contains modules for scraping data from sreality.cz
- `data/` - Contains modules for data processing
- `model/` - Contains modules for price prediction

## Setup

#### 1. Create a virtual environment and install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Run the scraper to collect data:
```bash
cd scraper
python main.py
```

#### 3. Run data processing
```bash
cd data
python main.py --method "json"
python main.py --method "images"
```


#### 4. Train the models:
This runs tree based models, image model and ensembles. Each stores artefacts.
```bash
cd model
sh run.sh
```

## Features

- Web scraping of real estate data from sreality.cz
- Data preprocessing and feature engineering
- Machine learning model for price prediction
- Evaluation metrics and visualization tools

## Data Sources

- sreality.cz - Primary source of real estate listings
