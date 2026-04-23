import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from keplergl import KeplerGl

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

def main():
    print("Generating Karachi Digital Twin Geo-Grid...")
    # Karachi roughly: [66.85, 24.75, 67.25, 25.05]
    karachi_bbox = [66.85, 24.75, 67.25, 25.05]
    
    gdf = create_grid(karachi_bbox, step=0.01) # ~1km grid
    
    # Simulate some PM2.5 data (gradient with some noise)
    # Higher pollution near industrial areas (SITE, Korangi)
    # SITE: 66.98, 24.94 | Korangi: 67.03, 24.82
    
    def simulate_pm25(row):
        lon, lat = row['longitude'], row['latitude']
        
        # Distance to SITE
        dist_site = np.sqrt((lon - 66.98)**2 + (lat - 24.94)**2)
        # Distance to Korangi
        dist_korangi = np.sqrt((lon - 67.03)**2 + (lat - 24.82)**2)
        
        # Base PM2.5 + emissions from hotspots
        pm25 = 40.0 + (0.1 / (dist_site + 0.01)) * 5 + (0.1 / (dist_korangi + 0.01)) * 8
        # Add some random noise
        pm25 += np.random.normal(0, 5)
        return max(10, pm25) # Cap min at 10
        
    print("Simulating PM2.5 spatial distribution...")
    gdf['PM2.5'] = gdf.apply(simulate_pm25, axis=1)
    
    print("Configuring Kepler.gl Map...")
    # Setup map config for dark mode + 3D
    config = {
      "version": "v1",
      "config": {
        "visState": {
          "filters": [],
          "layers": [
            {
              "id": "digital_twin_grid",
              "type": "geojson",
              "config": {
                "dataId": "karachi_pm25",
                "label": "Predicted PM2.5",
                "color": [255, 0, 0],
                "columns": {"geojson": "geometry"},
                "isVisible": True,
                "visConfig": {
                  "opacity": 0.8,
                  "thickness": 0.5,
                  "strokeColor": None,
                  "colorRange": {
                    "name": "Global Warming",
                    "type": "sequential",
                    "category": "Uber",
                    "colors": ["#5A1846", "#900C3F", "#C70039", "#E3611C", "#F1920E", "#FFC300"]
                  },
                  "filled": True,
                  "enable3d": True,
                  "elevationScale": 100,
                  "sizeField": {"name": "PM2.5", "type": "real"}
                }
              },
              "visualChannels": {
                "colorField": {"name": "PM2.5", "type": "real"},
                "colorScale": "quantile",
                "heightField": {"name": "PM2.5", "type": "real"},
                "heightScale": "linear"
              }
            }
          ],
          "interactionConfig": {
            "tooltip": {
              "fieldsToShow": {
                "karachi_pm25": [{"name": "PM2.5", "format": None}]
              },
              "compareMode": False,
              "compareType": "absolute",
              "enabled": True
            }
          }
        },
        "mapState": {
          "bearing": 24,
          "dragRotate": True,
          "latitude": 24.9,
          "longitude": 67.05,
          "pitch": 50,
          "zoom": 10,
          "isSplit": False
        },
        "mapStyle": {
          "styleType": "dark",
          "topLayerGroups": {},
          "visibleLayerGroups": {"label": True, "road": True, "border": False, "building": True, "water": True, "land": True}
        }
      }
    }

    m = KeplerGl(height=800, data={"karachi_pm25": gdf}, config=config)
    output_file = "karachi_digital_twin.html"
    m.save_to_html(file_name=output_file)
    print(f"✅ Success! Map rendered and saved to {output_file}")
    print("Open this file in Google Chrome to see the 3D map.")

if __name__ == "__main__":
    main()
