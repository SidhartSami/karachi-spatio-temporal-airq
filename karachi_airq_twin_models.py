"""
Simple Model Switcher for PyDeck Digital Twin
Shows 5 model predictions with clean buttons
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import osmnx as ox
import pydeck as pdk
import matplotlib.colors as mcolors
import os

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
    colors = ["#00E400", "#FFFF00", "#FF7E00", "#FF0000", "#8F3F97", "#7E0023"]
    norm = (val - min_val) / (max_val - min_val) if max_val > min_val else 0
    norm = max(0, min(1, norm))
    index = norm * (len(colors) - 1)
    idx1 = int(np.floor(index))
    idx2 = int(np.ceil(index))
    weight = index - idx1
    c1 = np.array(mcolors.to_rgb(colors[idx1]))
    c2 = np.array(mcolors.to_rgb(colors[idx2]))
    c = c1 * (1 - weight) + c2 * weight
    return [int(c[0]*255), int(c[1]*255), int(c[2]*255), 210]

def simulate_pm25_for_model(gdf, model_name):
    """Simulate PM2.5 with model-specific characteristics"""
    
    # Base hotspots [lon, lat, intensity, spread]
    hotspots = [
        [66.98, 24.90, 70, 0.04],  # SITE
        [67.11, 24.82, 85, 0.05],  # Korangi
        [67.05, 24.88, 50, 0.06],  # Central
        [67.20, 24.88, 40, 0.08],  # Malir
        [66.98, 24.82, 55, 0.03],  # Port
    ]
    
    # Model-specific seeds for consistent results
    np.random.seed(hash(model_name) % 10000)
    
    def calc_val(row):
        lon, lat = row['geometry'].centroid.x, row['geometry'].centroid.y
        pm25 = 25.0
        
        for hl, ha, intensity, spread in hotspots:
            dist_sq = (lon - hl)**2 + (lat - ha)**2
            
            # Base contribution
            contribution = intensity * np.exp(-dist_sq / (2 * spread**2))
            
            # Model-specific adjustments
            if model_name == "LSTM (Deep Learning)":
                # LSTM: Smoother predictions, less noise
                contribution *= 0.95
                noise = np.random.normal(0, 1.5)
            elif model_name == "XGBoost Ensemble":
                # XGBoost: Good balance
                contribution *= 1.0
                noise = np.random.normal(0, 2.5)
            elif model_name == "Random Forest":
                # RF: Slightly higher variance
                contribution *= 1.02
                noise = np.random.normal(0, 3.0)
            elif model_name == "Support Vector (SVR)":
                # SVR: More smoothing, underpredicts peaks
                contribution *= 0.88
                noise = np.random.normal(0, 4.5)
            else:  # Ensemble Average
                contribution *= 0.98
                noise = np.random.normal(0, 2.0)
            
            pm25 += contribution + noise * 0.3
        
        return max(15, pm25)
    
    return gdf.apply(calc_val, axis=1)

def generate_model_dashboard(model_name, all_models, active_model, gdf, global_min, global_max):
    """Generate HTML with clean model switcher buttons"""
    
    # Generate model buttons
    model_buttons = ""
    model_colors = {
        "Ensemble Average": "#c8f04a",
        "LSTM (Deep Learning)": "#4af0c8",
        "XGBoost Ensemble": "#f04a7a",
        "Random Forest": "#f0c84a",
        "Support Vector (SVR)": "#7a4af0"
    }
    
    for m in all_models:
        active_class = "active" if m == active_model else ""
        color = model_colors.get(m, "#888")
        m_file = f"model_{m.replace(' ', '_').replace('(', '').replace(')', '').lower()}.html"
        model_buttons += f'''
        <a href="{m_file}" class="model-btn {active_class}" style="--model-color: {color}">
            <span class="model-dot" style="background: {color}"></span>
            {m}
        </a>
        '''
    
    # Calculate stats
    pm25_values = gdf['PM2.5'].values
    mean_pm25 = np.mean(pm25_values)
    max_pm25 = np.max(pm25_values)
    exceed_24h = np.sum(pm25_values > 15)
    exceed_pct = 100 * exceed_24h / len(pm25_values)
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Karachi AirQ Twin — {model_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            margin: 0;
            padding: 0;
            background: #0d0d14;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            overflow: hidden;
            color: white;
        }}
        
        #dashboard {{
            position: absolute;
            top: 20px;
            left: 20px;
            width: 320px;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.6);
            z-index: 1000;
        }}
        
        h1 {{
            font-size: 1.3rem;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 4px;
        }}
        
        .subtitle {{
            color: #94a3b8;
            font-size: 0.85rem;
            margin-bottom: 24px;
        }}
        
        .current-model {{
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .current-label {{
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        
        .current-name {{
            font-size: 1.1rem;
            font-weight: 700;
            color: #38bdf8;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 24px;
        }}
        
        .stat-box {{
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #f8fafc;
        }}
        
        .stat-value.danger {{ color: #f87171; }}
        .stat-value.warning {{ color: #fbbf24; }}
        
        .stat-label {{
            font-size: 0.7rem;
            color: #94a3b8;
            margin-top: 4px;
        }}
        
        .models-title {{
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }}
        
        .model-btn {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px;
            margin-bottom: 8px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            color: #cbd5e1;
            text-decoration: none;
            font-size: 0.9rem;
            transition: all 0.2s;
        }}
        
        .model-btn:hover {{
            background: rgba(255, 255, 255, 0.1);
            color: white;
            transform: translateX(4px);
        }}
        
        .model-btn.active {{
            background: rgba(56, 189, 248, 0.2);
            border-color: rgba(56, 189, 248, 0.5);
            color: white;
        }}
        
        .model-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        
        .legend {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .legend-bar {{
            height: 10px;
            background: linear-gradient(to right, #00E400, #FFFF00, #FF7E00, #FF0000, #8F3F97, #7E0023);
            border-radius: 5px;
            position: relative;
        }}
        
        .legend-whomarker {{
            position: absolute;
            top: -3px;
            width: 2px;
            height: 16px;
            background: white;
        }}
        
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 0.65rem;
            color: #64748b;
            margin-top: 6px;
        }}
        
        .who-info {{
            margin-top: 16px;
            padding: 12px;
            background: rgba(248, 113, 113, 0.1);
            border: 1px solid rgba(248, 113, 113, 0.2);
            border-radius: 8px;
            font-size: 0.8rem;
            color: #f87171;
        }}
        
        #map-frame {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: none;
        }}
    </style>
</head>
<body>
    <div id="dashboard">
        <h1>🌍 Karachi AirQ Twin</h1>
        <div class="subtitle">Model Comparison Dashboard</div>
        
        <div class="current-model">
            <div class="current-label">Active Model</div>
            <div class="current-name">{model_name}</div>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value {'danger' if mean_pm25 > 35 else 'warning'}">{mean_pm25:.1f}</div>
                <div class="stat-label">Mean PM2.5<br>µg/m³</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{max_pm25:.1f}</div>
                <div class="stat-label">Max PM2.5<br>µg/m³</div>
            </div>
            <div class="stat-box">
                <div class="stat-value {'danger' if exceed_pct > 50 else 'warning'}">{exceed_pct:.1f}%</div>
                <div class="stat-label">Exceeds<br>WHO 24h</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{int(len(pm25_values))}</div>
                <div class="stat-label">Grid<br>Cells</div>
            </div>
        </div>
        
        <div class="models-title">🤖 Switch Model</div>
        {model_buttons}
        
        <div class="legend">
            <div class="legend-bar">
                <div class="legend-whomarker" style="left: 13%;"></div>
                <div class="legend-whomarker" style="left: 40%;"></div>
            </div>
            <div class="legend-labels">
                <span>Clean</span>
                <span>WHO</span>
                <span>Unhealthy</span>
                <span>Hazardous</span>
            </div>
        </div>
        
        <div class="who-info">
            ⚠️ WHO 24h limit: 15 µg/m³ | Annual: 5 µg/m³<br>
            Karachi exceeds by {(mean_pm25/5):.1f}×
        </div>
    </div>
    
    <iframe id="map-frame" src="map_{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.html"></iframe>
</body>
</html>
'''
    return html

def main():
    print("🏭 Creating Simple Model Switcher Dashboard...")
    print("=" * 60)
    
    # Fetch Karachi boundary
    print("Fetching Karachi boundary...")
    try:
        karachi_boundary = ox.geocode_to_gdf("Karachi, Pakistan")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Create grid
    karachi_bbox = [66.85, 24.76, 67.25, 25.10]
    print("Generating grid...")
    gdf_base = create_grid(karachi_bbox, step=0.01)
    gdf_base = gpd.clip(gdf_base, karachi_boundary)
    
    # Define models
    models = [
        "Ensemble Average",
        "LSTM (Deep Learning)",
        "XGBoost Ensemble",
        "Random Forest",
        "Support Vector (SVR)"
    ]
    
    # Calculate PM2.5 for all models to get global min/max
    print("\nSimulating PM2.5 for all models...")
    model_data = {}
    all_values = []
    
    for model in models:
        pm25 = simulate_pm25_for_model(gdf_base, model)
        model_data[model] = pm25
        all_values.extend(pm25.values)
        print(f"  {model:<25}: mean={pm25.mean():.1f}, max={pm25.max():.1f}")
    
    global_min = min(all_values)
    global_max = max(all_values)
    print(f"\nGlobal range: {global_min:.1f} - {global_max:.1f} µg/m³")
    
    # Generate map and dashboard for each model
    print("\nGenerating dashboards...")
    for model in models:
        print(f"  Creating {model}...")
        
        gdf = gdf_base.copy()
        gdf['PM2.5'] = model_data[model]
        gdf['fill_color'] = gdf['PM2.5'].apply(lambda x: get_color(x, global_min, global_max))
        gdf['elevation'] = gdf['PM2.5'] * 120
        gdf['pm25_formatted'] = gdf['PM2.5'].round(2).astype(str) + ' µg/m³'
        
        # Generate PyDeck map
        grid_layer = pdk.Layer(
            "GeoJsonLayer",
            gdf,
            opacity=0.85,
            stroked=False,
            filled=True,
            extruded=True,
            wireframe=True,
            get_elevation="elevation",
            get_fill_color="fill_color",
            pickable=True,
            auto_highlight=True,
        )
        
        labels_data = [
            {"name": "SITE INDUSTRIAL", "position": [66.98, 24.90, 9000]},
            {"name": "KORANGI", "position": [67.11, 24.82, 10500]},
            {"name": "CENTRAL", "position": [67.05, 24.88, 7000]},
        ]
        
        text_layer = pdk.Layer(
            "TextLayer",
            labels_data,
            get_position="position",
            get_text="name",
            get_size=24,
            get_color=[255, 255, 255, 255],
            get_alignment_baseline="'bottom'",
            background=True,
            get_background_color=[0, 0, 0, 180],
            font_weight="bold",
            billboard=True,
            pickable=False
        )
        
        view_state = pdk.ViewState(
            latitude=24.88,
            longitude=67.05,
            zoom=10.8,
            pitch=50,
            bearing=20
        )
        
        tooltip = {
            "html": "<b>PM2.5:</b> {pm25_formatted}",
            "style": {"backgroundColor": "rgba(15, 23, 42, 0.9)", "color": "white"}
        }
        
        r = pdk.Deck(
            layers=[grid_layer, text_layer],
            initial_view_state=view_state,
            map_style="dark",
            tooltip=tooltip
        )
        
        # Save map
        map_filename = f"dashboard/map_{model.replace(' ', '_').replace('(', '').replace(')', '').lower()}.html"
        r.to_html(map_filename)
        
        # Save dashboard
        dashboard_html = generate_model_dashboard(model, models, model, gdf, global_min, global_max)
        dashboard_filename = f"dashboard/model_{model.replace(' ', '_').replace('(', '').replace(')', '').lower()}.html"
        with open(dashboard_filename, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
    
    print(f"\n✅ Success! {len(models)} model dashboards created")
    print(f"\n🌍 Open any of these to explore:")
    for model in models:
        fname = f"dashboard/model_{model.replace(' ', '_').replace('(', '').replace(')', '').lower()}.html"
        print(f"  {fname}")
    
    print(f"\n💡 Features:")
    print(f"  • Clean 3D PyDeck visualization")
    print(f"  • Simple model buttons (no sliders/percentages)")
    print(f"  • Click any model to see its predictions")
    print(f"  • Stats update for each model")
    print(f"  • Consistent color scale across all models")

if __name__ == "__main__":
    main()
