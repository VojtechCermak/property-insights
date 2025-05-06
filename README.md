# Czech Real Estate Price Evaluator

This project aims to create a system for evaluating real estate prices in the Czech Republic, focusing on residential properties and apartments.

## Project Structure

- `scraper/` - Contains modules for scraping data from sreality.cz
- `model/` - Contains modules for data processing and price prediction
- `data/` - Directory for storing scraped data and model artifacts
- `config/` - Configuration files and environment variables

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your configuration.

## Usage

1. Run the scraper to collect data:
```bash
python scraper/main.py
```

2. Train the model:
```bash
python model/train.py
```

3. Use the model for predictions:
```bash
python model/predict.py
```

## Features

- Web scraping of real estate data from sreality.cz
- Data preprocessing and feature engineering
- Machine learning model for price prediction
- Evaluation metrics and visualization tools

## Data Sources

- sreality.cz - Primary source of real estate listings
- Additional sources may be added in the future

## License

MIT License 