import ee
import time

def main():
    try:
        # Use the specific project ID
        ee.Initialize(project='gen-lang-client-0478151371')
        print("GEE Initialized Successfully")
    except Exception as e:
        print(f"GEE Initialization Failed: {e}")
        return

    # 1. Define Stations (aligned with notebook)
    stations_dict = {
        'Gulshan-e-Iqbal': [67.0822, 24.9056],
        'Saddar': [67.0100, 24.8560],
        'SITE_Industrial': [66.9800, 24.9400],
        'Korangi_Industrial': [67.0300, 24.8200],
        'North_Nazimabad': [67.1200, 24.9800],
        'Gulistan_Jauhar': [67.1300, 24.8900],
        'Landhi': [66.9900, 24.8100],
        'Federal_B_Area': [67.0500, 24.9200]
    }
    
    station_features = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point(coord), {'station': name}) 
        for name, coord in stations_dict.items()
    ])

    start_date = '2019-01-01'
    end_date = '2026-03-31'
    karachi_rect = ee.Geometry.Rectangle([66.60, 24.60, 67.60, 25.20])

    def export_collection(col, name, band_names):
        def extract_image(image):
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            reduced = image.select(band_names).reduceRegions(
                collection=station_features,
                reducer=ee.Reducer.mean(),
                scale=1000
            )
            return reduced.map(lambda f: f.set('date', date))
        
        results = col.map(extract_image).flatten()
        
        task = ee.batch.Export.table.toDrive(
            collection=results,
            description=name,
            folder='karachi_airq_exports',
            fileFormat='CSV'
        )
        task.start()
        print(f"✓ Queued Export: {name}")

    # --- 1. Sentinel-5P ---
    s5p_sources = {
        'aer_ai': ('COPERNICUS/S5P/NRTI/L3_AER_AI', ['absorbing_aerosol_index']),
        'no2':    ('COPERNICUS/S5P/NRTI/L3_NO2',    ['NO2_column_number_density']),
        'so2':    ('COPERNICUS/S5P/NRTI/L3_SO2',    ['SO2_column_number_density']),
        'co':     ('COPERNICUS/S5P/NRTI/L3_CO',     ['CO_column_number_density'])
    }
    
    for label, (asset_id, bands) in s5p_sources.items():
        col = ee.ImageCollection(asset_id).filterDate(start_date, end_date).filterBounds(karachi_rect)
        export_collection(col, f'karachi_s5p_{label}', bands)

    # --- 2. MODIS AOD (MAIAC 1km) ---
    modis = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES') \
        .filterDate(start_date, end_date) \
        .filterBounds(karachi_rect) \
        .select(['Optical_Depth_047', 'Optical_Depth_055']) \
        .map(lambda img: img.multiply(0.001).set('system:time_start', img.get('system:time_start')))
    export_collection(modis, 'karachi_modis_aod', ['Optical_Depth_047', 'Optical_Depth_055'])

    # --- 3. ERA5 Land ---
    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
        .filterDate(start_date, end_date) \
        .filterBounds(karachi_rect)
    
    def engineer_era5(img):
        u = img.select('u_component_of_wind_10m')
        v = img.select('v_component_of_wind_10m')
        ws = u.hypot(v).rename('wind_speed')
        t = img.select('temperature_2m').subtract(273.15)
        td = img.select('dewpoint_temperature_2m').subtract(273.15)
        rh = td.subtract(t).multiply(100/23.0).add(100).rename('rh')
        return img.addBands([ws, rh]).set('system:time_start', img.get('system:time_start'))

    era5_eng = era5.map(engineer_era5).select(['wind_speed', 'rh', 'temperature_2m', 'total_precipitation_sum'])
    export_collection(era5_eng, 'karachi_era5_met', ['wind_speed', 'rh', 'temperature_2m', 'total_precipitation_sum'])

    print("\nAll 6 primary export tasks have been queued in GEE.")
    print("Please wait 10-15 minutes, then check your Google Drive 'karachi_airq_exports' folder.")

if __name__ == "__main__":
    main()
