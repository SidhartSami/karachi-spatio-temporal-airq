import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import osmnx as ox
import pydeck as pdk
import matplotlib.colors as mcolors

def create_grid(bbox, step=0.01):
    min_lon, min_lat, max_lon, max_lat = bbox
    lons = np.arange(min_lon, max_lon, step)
    lats = np.arange(min_lat, max_lat, step)
    
    grid = []
    for lon in lons:
        for lat in lats:
            grid.append({
                'geometry': box(lon, lat, lon + step, lat + step),
                'longitude': lon + step/2,
                'latitude': lat + step/2
            })
    return gpd.GeoDataFrame(grid, crs="EPSG:4326")

def get_color(val, min_val, max_val):
    # Kepler.gl "Global Warming" color palette
    colors = ["#5A1846", "#900C3F", "#C70039", "#E3611C", "#F1920E", "#FFC300"]
    # Normalize value between 0 and 1
    norm = (val - min_val) / (max_val - min_val) if max_val > min_val else 0
    norm = max(0, min(1, norm))
    
    # Interpolate
    index = norm * (len(colors) - 1)
    idx1 = int(np.floor(index))
    idx2 = int(np.ceil(index))
    weight = index - idx1
    
    c1 = np.array(mcolors.to_rgb(colors[idx1]))
    c2 = np.array(mcolors.to_rgb(colors[idx2]))
    c = c1 * (1 - weight) + c2 * weight
    return [int(c[0]*255), int(c[1]*255), int(c[2]*255), 200]

def main():
    print("Fetching exact Karachi municipal boundary...")
    try:
        karachi_boundary = ox.geocode_to_gdf("Karachi, Pakistan")
    except Exception as e:
        print(f"Error fetching boundary: {e}")
        return
        
    print("Generating Geo-Grid...")
    bounds = karachi_boundary.total_bounds
    karachi_bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]
    
    gdf = create_grid(karachi_bbox, step=0.015) # ~1.5km grid
    
    print("Clipping grid to exact Karachi shape (removing ocean/outside areas)...")
    gdf = gpd.clip(gdf, karachi_boundary)
    
    def simulate_pm25(row):
        lon, lat = row['geometry'].centroid.x, row['geometry'].centroid.y
        dist_site = np.sqrt((lon - 66.98)**2 + (lat - 24.94)**2)
        dist_korangi = np.sqrt((lon - 67.03)**2 + (lat - 24.82)**2)
        pm25 = 40.0 + (0.1 / (dist_site + 0.01)) * 5 + (0.1 / (dist_korangi + 0.01)) * 8
        pm25 += np.random.normal(0, 5)
        return max(10, pm25)
        
    print("Simulating PM2.5 spatial distribution...")
    gdf['PM2.5'] = gdf.apply(simulate_pm25, axis=1)
    
    print("Configuring PyDeck Map with real basemap and Dark Theme...")
    # Pre-calculate colors and elevations
    min_pm = gdf['PM2.5'].min()
    max_pm = gdf['PM2.5'].max()
    
    gdf['fill_color'] = gdf['PM2.5'].apply(lambda x: get_color(x, min_pm, max_pm))
    gdf['elevation'] = gdf['PM2.5'] * 150  # Extrusion height scale
    
    # Format PM2.5 to 2 decimal places for the tooltip
    gdf['pm25_formatted'] = gdf['PM2.5'].round(2).astype(str) + ' µg/m³'

    layer = pdk.Layer(
        "GeoJsonLayer",
        gdf,
        opacity=0.8,
        stroked=False,
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation="elevation",
        get_fill_color="fill_color",
        pickable=True,
        auto_highlight=True
    )

    view_state = pdk.ViewState(
        latitude=24.9,
        longitude=67.05,
        zoom=10,
        pitch=50,
        bearing=24
    )
    
    tooltip = {
        "html": "<b>Predicted PM2.5:</b> {pm25_formatted}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }

    # "dark" map_style uses CartoDB Dark Matter automatically (no API key needed!)
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="dark", tooltip=tooltip)
    
    output_file = "karachi_digital_twin.html"
    r.to_html(output_file)
    print(f"✅ Success! Map rendered and saved to {output_file}")
    print("Open this file in Google Chrome. You will now see the actual world map underneath (Dark Theme) without needing any API key!")

if __name__ == "__main__":
    main()
