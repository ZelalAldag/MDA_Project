Group 3 - MDA Project

# Modern Data Analytics Project

## Setup virtual environment

Run the following command in to create a virtual environment if you don't have one already:

```
python -m venv .venv
```

To activate the virtual environment, run the following command:

Windows:
```
.venv\Scripts\activate
```

Bash:
```
source .venv/bin/activate
```

## Install dependencies

Run the following command to install the required dependencies:

```
pip install -r requirements.txt
```

## Datasets

The datasets used in this project are stored in the Hugging Face dataset repository. Use the ,`huggingface_hub` library to download the datasets to local storage.

Weather data files: 
- `weather_data_2024-2026.csv`: The hourly weather data for each site for the years 2024 to 2026.
- `weather_metadata_2024-2026.csv`: The metadata about the weather data stations for each site.