import ee
try:
    ee.Initialize(project='gen-lang-client-0478151371')
    
    # Karachi Point
    karachi_point = ee.Geometry.Point([67.0, 24.9])
    start_date = '2019-01-01'
    end_date = '2024-01-01'

    # S5P AER
    s5p = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI') \
        .filterDate(start_date, end_date) \
        .filterBounds(karachi_point)
    
    # MODIS AOD (Corrected ID: MCD19A2_GRANULES)
    modis = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES') \
        .filterDate(start_date, end_date) \
        .filterBounds(karachi_point)

    # ERA5 (ERA5 Land Daily Aggregated)
    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
        .filterDate(start_date, end_date) \
        .filterBounds(karachi_point)

    print(f"S5P AER images: {s5p.size().getInfo()}")
    print(f"MODIS AOD images: {modis.size().getInfo()}")
    print(f"ERA5 images: {era5.size().getInfo()}")

except Exception as e:
    print(f"Error: {e}")
