import torch
import geopandas as gpd
import rasterio
import xgboost
import lightgbm
import prophet
import folium
import ee

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("geopandas:", gpd.__version__)
print("xgboost:", xgboost.__version__)
print("all good ✓")
