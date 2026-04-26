"""
Working Karachi Air Quality Digital Twin Generator
Creates ACTUAL scenario variants that users can switch between
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import osmnx as ox
import pydeck as pdk
import matplotlib.colors as mcolors
import os

# WHO Guidelines
WHO_24H = 15
WHO_ANNUAL = 5

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

def calculate_who_stats(pm25_values):
    """Calculate WHO exceedance statistics"""
    total_cells = len(pm25_values)
    exceeding_24h = np.sum(pm25_values > WHO_24H)
    exceeding_annual = np.sum(pm25_values > WHO_ANNUAL)
    
    return {
        'total': total_cells,
        'exceed_24h': int(exceeding_24h),
        'pct_exceed_24h': round(100 * exceeding_24h / total_cells, 1),
        'exceed_annual': int(exceeding_annual),
        'pct_exceed_annual': round(100 * exceeding_annual / total_cells, 1),
        'mean_pm25': round(np.mean(pm25_values), 1),
        'max_pm25': round(np.max(pm25_values), 1)
    }

def simulate_pm25_scenario(gdf, model_name, scenario_type='baseline', month=6):
    """
    Simulate PM2.5 with different policy scenarios
    
    Args:
        gdf: GeoDataFrame with grid cells
        model_name: Model identifier
        scenario_type: 'baseline', 'industry_cut', 'traffic_cut', 'green_expansion', 'combined'
        month: Month (1-12) for seasonal adjustment
    """
    # Base hotspots [lon, lat, intensity, spread, type]
    hotspots = [
        [66.98, 24.90, 70, 0.04, 'industrial'],  # SITE
        [67.11, 24.82, 85, 0.05, 'industrial'],  # Korangi
        [67.05, 24.88, 50, 0.06, 'traffic'],     # Central
        [67.20, 24.88, 40, 0.08, 'mixed'],       # Malir
        [66.98, 24.82, 55, 0.03, 'port'],       # Port
    ]
    
    np.random.seed(len(model_name) + month + hash(scenario_type) % 1000)
    
    # Seasonal factors
    seasonal_factors = {
        1: 1.25, 2: 1.20, 3: 1.05, 4: 0.95, 5: 0.90, 6: 0.85,
        7: 0.75, 8: 0.80, 9: 0.90, 10: 1.10, 11: 1.20, 12: 1.30
    }
    seasonal_mult = seasonal_factors.get(month, 1.0)
    
    # Scenario multipliers
    scenario_config = {
        'baseline': {'industry': 1.0, 'traffic': 1.0, 'green': 0.0},
        'industry_30': {'industry': 0.70, 'traffic': 1.0, 'green': 0.0},
        'industry_50': {'industry': 0.50, 'traffic': 1.0, 'green': 0.0},
        'traffic_30': {'industry': 1.0, 'traffic': 0.70, 'green': 0.0},
        'traffic_50': {'industry': 1.0, 'traffic': 0.50, 'green': 0.0},
        'green_30': {'industry': 1.0, 'traffic': 1.0, 'green': 0.30},
        'green_50': {'industry': 1.0, 'traffic': 1.0, 'green': 0.50},
        'combined': {'industry': 0.75, 'traffic': 0.75, 'green': 0.30},
        'aggressive': {'industry': 0.50, 'traffic': 0.50, 'green': 0.50},
    }
    
    config = scenario_config.get(scenario_type, scenario_config['baseline'])
    
    def calc_val(row):
        lon, lat = row['geometry'].centroid.x, row['geometry'].centroid.y
        pm25 = 25.0 * seasonal_mult
        
        for hl, ha, intensity, spread, htype in hotspots:
            dist_sq = (lon - hl)**2 + (lat - ha)**2
            
            # Apply scenario reductions
            adjusted_intensity = intensity
            if htype == 'industrial':
                adjusted_intensity *= config['industry']
            elif htype == 'traffic':
                adjusted_intensity *= config['traffic']
            
            # Green belt applies uniform reduction
            green_reduction = config['green'] * 0.5  # Max 50% reduction
            
            model_bias = np.random.normal(0, 5) if model_name != "LSTM" else 0
            contribution = (adjusted_intensity + model_bias) * np.exp(-dist_sq / (2 * spread**2))
            pm25 += contribution * (1 - green_reduction)
        
        # Model-specific noise
        noise = np.random.normal(0, 2)
        if model_name == "SVR": noise = np.random.normal(0, 6)
        if model_name == "Random Forest": noise = np.random.normal(0, 4)
        
        return max(5, pm25 + noise)
    
    return gdf.apply(calc_val, axis=1)

def generate_dashboard_html(scenario_data, active_scenario, all_scenarios, global_min, global_max):
    """Generate HTML dashboard with WORKING scenario switcher"""
    
    # Generate scenario buttons
    scenario_buttons = ""
    scenario_labels = {
        'baseline': '🏠 Baseline',
        'industry_30': '🏭 Industry -30%',
        'industry_50': '🏭 Industry -50%',
        'traffic_30': '🚗 Traffic -30%',
        'traffic_50': '🚗 Traffic -50%',
        'green_30': '🌳 Green +30%',
        'green_50': '🌳 Green +50%',
        'combined': '⚡ Combined',
        'aggressive': '🎯 Aggressive'
    }
    
    for scen_id, scen_label in scenario_labels.items():
        if scen_id in all_scenarios:
            active_class = "active" if scen_id == active_scenario else ""
            scen_file = f"karachi_twin_{scen_id}.html"
            scenario_buttons += f'''
            <a href="{scen_file}" class="scenario-btn {active_class}">{scen_label}</a>
            '''
    
    # Get stats for this scenario
    stats = scenario_data['stats']
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Karachi AirQ Twin — {scenario_labels.get(active_scenario, active_scenario)}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #0d0d14;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            overflow: hidden;
        }}
        
        #dashboard {{
            position: absolute;
            top: 20px;
            left: 20px;
            width: 340px;
            max-height: 95vh;
            overflow-y: auto;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            padding: 24px;
            color: white;
            box-shadow: 0 20px 50px rgba(0,0,0,0.6);
            z-index: 1000;
        }}
        
        #dashboard::-webkit-scrollbar {{ width: 6px; }}
        #dashboard::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.2); border-radius: 3px; }}
        
        h2 {{
            margin-top: 0;
            font-size: 1.3rem;
            font-weight: 700;
            color: #38bdf8;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .subtitle {{
            color: #94a3b8;
            font-size: 0.85rem;
            margin-bottom: 20px;
        }}
        
        .who-section {{
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        
        .who-title {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94a3b8;
            margin-bottom: 12px;
        }}
        
        .who-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}
        
        .who-stat {{
            text-align: center;
            padding: 12px 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
        }}
        
        .who-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #f8fafc;
        }}
        
        .who-value.warning {{ color: #fbbf24; }}
        .who-value.danger {{ color: #f87171; }}
        .who-value.good {{ color: #4ade80; }}
        
        .who-label {{
            font-size: 0.7rem;
            color: #94a3b8;
            margin-top: 4px;
        }}
        
        .scenario-section {{
            background: rgba(56, 189, 248, 0.08);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
            border: 1px solid rgba(56, 189, 248, 0.15);
        }}
        
        .scenario-title {{
            font-size: 0.85rem;
            color: #38bdf8;
            margin-bottom: 12px;
            font-weight: 600;
        }}
        
        .scenario-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }}
        
        .scenario-btn {{
            display: block;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: #cbd5e1;
            text-decoration: none;
            font-size: 0.75rem;
            text-align: center;
            transition: all 0.2s;
        }}
        
        .scenario-btn:hover {{
            background: rgba(255, 255, 255, 0.1);
            color: white;
        }}
        
        .scenario-btn.active {{
            background: rgba(56, 189, 248, 0.25);
            border-color: rgba(56, 189, 248, 0.5);
            color: white;
            font-weight: 600;
        }}
        
        #legend {{
            margin: 16px 0;
            height: 12px;
            background: linear-gradient(to right, #00E400, #FFFF00, #FF7E00, #FF0000, #8F3F97, #7E0023);
            border-radius: 6px;
            position: relative;
        }}
        
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            color: #64748b;
            margin-top: 6px;
        }}
        
        .legend-who {{
            position: absolute;
            top: -4px;
            width: 3px;
            height: 20px;
            background: white;
            border-radius: 2px;
        }}
        
        .info-box {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 12px;
            font-size: 0.8rem;
            color: #94a3b8;
            line-height: 1.5;
        }}
        
        .map-container {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}
    </style>
</head>
<body>
    <div id="dashboard">
        <h2>🌍 Karachi AirQ Twin</h2>
        <div class="subtitle">Interactive Policy Scenario Simulator</div>
        
        <div class="who-section">
            <div class="who-title">⚠️ WHO Guideline Compliance — {scenario_labels.get(active_scenario, active_scenario)}</div>
            <div class="who-grid">
                <div class="who-stat">
                    <div class="who-value {'danger' if stats['exceed_24h'] > 50 else 'warning' if stats['exceed_24h'] > 20 else 'good'}">{stats['exceed_24h']}</div>
                    <div class="who-label">cells > 24h limit<br>(15 µg/m³)</div>
                </div>
                <div class="who-stat">
                    <div class="who-value">{stats['pct_exceed_24h']}%</div>
                    <div class="who-label">of urban area</div>
                </div>
                <div class="who-stat">
                    <div class="who-value {'danger' if stats['mean_pm25'] > 35 else 'warning' if stats['mean_pm25'] > 15 else 'good'}">{stats['mean_pm25']}</div>
                    <div class="who-label">mean PM2.5<br>(µg/m³)</div>
                </div>
                <div class="who-stat">
                    <div class="who-value">{stats['max_pm25']}</div>
                    <div class="who-label">max PM2.5<br>(µg/m³)</div>
                </div>
            </div>
        </div>
        
        <div class="scenario-section">
            <div class="scenario-title">🎛️ Click to Switch Scenario</div>
            <div class="scenario-grid">
                {scenario_buttons}
            </div>
        </div>
        
        <div id="legend">
            <div class="legend-who" style="left: 13%;" title="WHO Annual (5 µg/m³)"></div>
            <div class="legend-who" style="left: 40%;" title="WHO 24h (15 µg/m³)"></div>
        </div>
        <div class="legend-labels">
            <span>Clean ({int(global_min)})</span>
            <span>Hazardous ({int(global_max)})</span>
        </div>
        
        <div class="info-box">
            <b>Current:</b> {scenario_labels.get(active_scenario, active_scenario)}<br>
            Each button loads a pre-computed scenario with different policy assumptions. 
            The 3D map updates to show PM2.5 levels under that scenario.
        </div>
    </div>
    
    <div class="map-container">
        <iframe src="map_{active_scenario}.html" width="100%" height="100%" frameborder="0"></iframe>
    </div>
</body>
</html>
"""
    return html

def generate_map_layer(gdf, pm25_values, filename, global_min, global_max):
    """Generate a standalone PyDeck map HTML for embedding"""
    
    gdf['PM2.5'] = pm25_values
    gdf['fill_color'] = gdf['PM2.5'].apply(lambda x: get_color(x, global_min, global_max))
    gdf['elevation'] = gdf['PM2.5'] * 120
    gdf['pm25_formatted'] = gdf['PM2.5'].round(2).astype(str) + ' µg/m³'
    
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
        {"name": "KARACHI URBAN CORE", "position": [67.02, 24.86, 30000]},
        {"name": "SITE INDUSTRIAL", "position": [66.98, 24.90, 32000]},
        {"name": "KORANGI INDUSTRIAL", "position": [67.11, 24.82, 35000]},
        {"name": "MALIR", "position": [67.20, 24.88, 28000]},
        {"name": "CLIFTON / DHA", "position": [67.03, 24.80, 25000]}
    ]
    
    text_layer = pdk.Layer(
        "TextLayer",
        labels_data,
        get_position="position",
        get_text="name",
        get_size=28,
        get_color=[255, 255, 255, 255],
        get_alignment_baseline="'bottom'",
        background=True,
        get_background_color=[0, 0, 0, 180],
        font_weight="bold",
        font_family="Roboto, sans-serif",
        billboard=True,
        pickable=False
    )
    
    view_state = pdk.ViewState(
        latitude=24.88,
        longitude=67.05,
        zoom=10.5,
        pitch=55,
        bearing=15
    )
    
    tooltip = {
        "html": "<b>Predicted PM2.5:</b> {pm25_formatted}",
        "style": {"backgroundColor": "rgba(15, 23, 42, 0.9)", "color": "white", "borderRadius": "4px"}
    }
    
    r = pdk.Deck(layers=[grid_layer, text_layer], initial_view_state=view_state, map_style="dark", tooltip=tooltip)
    r.to_html(filename)

def main():
    print("🏭 Karachi Air Quality Digital Twin — Working Edition")
    print("=" * 60)
    
    print("Fetching Karachi municipal boundary...")
    try:
        karachi_boundary = ox.geocode_to_gdf("Karachi, Pakistan")
    except Exception as e:
        print(f"Error fetching boundary: {e}")
        return
    
    karachi_bbox = [66.85, 24.76, 67.25, 25.10]
    
    print("Generating 1km x 1km Geo-Grid...")
    gdf_base = create_grid(karachi_bbox, step=0.01)
    gdf_base = gpd.clip(gdf_base, karachi_boundary)
    
    # Define all scenarios to generate
    scenarios = {
        'baseline': 'Baseline (No Policy)',
        'industry_30': 'Industry -30%',
        'industry_50': 'Industry -50%',
        'traffic_30': 'Traffic -30%',
        'traffic_50': 'Traffic -50%',
        'green_30': 'Green Belt +30%',
        'green_50': 'Green Belt +50%',
        'combined': 'Combined Policies',
        'aggressive': 'Aggressive (All -50%)',
    }
    
    # Calculate PM2.5 for all scenarios to get global min/max for consistent colors
    print("\nSimulating PM2.5 for all scenarios...")
    all_pm25_values = []
    scenario_data = {}
    
    for scen_id in scenarios.keys():
        pm25 = simulate_pm25_scenario(gdf_base, "Ensemble", scen_id, month=6)
        all_pm25_values.extend(pm25.values)
        stats = calculate_who_stats(pm25.values)
        scenario_data[scen_id] = {
            'pm25': pm25,
            'stats': stats
        }
        print(f"  {scen_id:<15}: mean={stats['mean_pm25']:.1f}, exceed={stats['pct_exceed_24h']:.1f}%")
    
    global_min = min(all_pm25_values)
    global_max = max(all_pm25_values)
    print(f"\nGlobal PM2.5 range: {global_min:.1f} - {global_max:.1f} µg/m³")
    
    # Generate dashboard + map for each scenario
    print("\nGenerating scenario dashboards...")
    for scen_id in scenarios.keys():
        print(f"  Creating {scen_id}...")
        
        # Generate map layer
        map_filename = f"dashboard/map_{scen_id}.html"
        generate_map_layer(
            gdf_base.copy(), 
            scenario_data[scen_id]['pm25'], 
            map_filename,
            global_min, 
            global_max
        )
        
        # Generate dashboard HTML
        dashboard_html = generate_dashboard_html(
            scenario_data[scen_id],
            scen_id,
            list(scenarios.keys()),
            global_min,
            global_max
        )
        
        dashboard_filename = f"dashboard/karachi_twin_{scen_id}.html"
        with open(dashboard_filename, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
    
    print(f"\n✅ Success! {len(scenarios)} working scenario dashboards generated")
    print(f"\n🌍 Open any of these files to explore:")
    for scen_id, label in scenarios.items():
        print(f"  dashboard/karachi_twin_{scen_id}.html  ({label})")
    
    print(f"\n💡 How it works:")
    print(f"  • Each scenario is a pre-computed PM2.5 simulation")
    print(f"  • Clicking a scenario button loads a different map")
    print(f"  • The WHO counter shows real stats for that scenario")
    print(f"  • 3D visualization actually changes between scenarios!")

if __name__ == "__main__":
    main()
