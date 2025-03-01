from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
import metpy.calc as mpcalc
from metpy.units import units
from siphon.catalog import TDSCatalog
import locale
import numpy as np
import geojson

# Set the locale to "C" to force a period as the decimal separator
locale.setlocale(locale.LC_ALL, 'C')

app = Flask(__name__)
CORS(app)

def categorize_risk(cape, srh):
    """Categorize the storm risk based on CAPE and SRH."""
    if cape >= 3000 and srh >= 300:
        return 'HIGH'
    elif cape >= 2500 and srh >= 250:
        return 'MDT'
    elif cape >= 2000 and srh >= 200:
        return 'ENH'
    elif cape >= 1500 and srh >= 150:
        return 'SLGT'
    elif cape >= 1000 and srh >= 100:
        return 'MRGL'
    else:
        return 'NONE'

def get_gfs_data(valid_time):
    try:
        # GFS model access URL and dataset
        cat_url = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml'

        # Access the catalog and datasets
        cat = TDSCatalog(cat_url)
        datasets = cat.datasets
        print(f"Available datasets: {[ds.name for ds in datasets.values()]}")

        # Loop over the available datasets and find the one we want
        dataset_name = None
        for ds in datasets.values():
            if 'GFS' in ds.name:
                dataset_name = ds.name
                break
        
        if not dataset_name:
            raise ValueError("GFS dataset not found in catalog")

        print(f"Using dataset: {dataset_name}")
        dataset = datasets[dataset_name]
        ncss = dataset.subset()

        # Define the bounding box for CONUS (United States)
        lat_min = 24.396308  # Southern boundary of CONUS
        lat_max = 49.384358  # Northern boundary of CONUS
        lon_min = -125.0     # Western boundary of CONUS
        lon_max = -66.93457  # Eastern boundary of CONUS

        # Create query for CAPE and SRH over the entire CONUS region
        query = ncss.query()
        query.lonlat_box(west=lon_min, east=lon_max, north=lat_max, south=lat_min).time(valid_time).accept('netcdf4')
        query.variables('Convective_available_potential_energy_surface', 'Storm_relative_helicity_height_above_ground_layer')

        # Fetch the data
        data = ncss.get_data(query)

        # Extract CAPE and SRH
        cape_var = data.variables['Convective_available_potential_energy_surface']
        srh_var = data.variables['Storm_relative_helicity_height_above_ground_layer']

        # Ensure to extract values correctly from the MaskedArray
        cape = cape_var[:].squeeze()
        srh = srh_var[:].squeeze()

        # Check if the data is a MaskedArray and handle accordingly
        if isinstance(cape, np.ma.MaskedArray):
            cape = np.ma.filled(cape, np.nan)  # Fill masked values with NaN
        if isinstance(srh, np.ma.MaskedArray):
            srh = np.ma.filled(srh, np.nan)  # Fill masked values with NaN

        # Convert to float and handle array values properly
        cape_values = [float(val) if np.isreal(val) else np.nan for val in cape.flatten()]
        srh_values = [float(val) if np.isreal(val) else np.nan for val in srh.flatten()]

        # GeoJSON to represent the risk areas as polygons
        features = []
        for lat_idx, lon_idx in np.ndindex(cape.shape):  # iterate over all points in the array
            lat = lat_min + lat_idx * (lat_max - lat_min) / cape.shape[0]
            lon = lon_min + lon_idx * (lon_max - lon_min) / cape.shape[1]

            # Categorize the risk level
            risk_level = categorize_risk(cape_values[lat_idx * cape.shape[1] + lon_idx], srh_values[lat_idx * srh.shape[1] + lon_idx])

            # Only include risk levels that are not 'NONE'
            if risk_level != 'NONE':
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [lon, lat]
                    },
                    'properties': {
                        'risk_level': risk_level,
                        'cape': cape_values[lat_idx * cape.shape[1] + lon_idx],
                        'srh': srh_values[lat_idx * srh.shape[1] + lon_idx]
                    }
                }
                features.append(feature)

        # Create GeoJSON FeatureCollection
        geojson_data = geojson.FeatureCollection(features)

        return geojson_data

    except Exception as e:
        print(f"Error fetching data for GFS: {e}")
        return {'error': str(e)}, 500


@app.route('/api/weatherData')
def weather_data():
    days_str = request.args.get('days', '1')

    try:
        days = int(days_str)
    except ValueError:
        return jsonify({'error': 'Invalid days parameter. Must be an integer'}), 400

    now = datetime.now(timezone.utc)

    # Valid time based on the requested days
    valid_time = now + timedelta(hours=12 * (days - 1))  # GFS has forecasts every 12 hours
    valid_time = valid_time.replace(minute=0, second=0, microsecond=0)

    # Fetch the data for CONUS
    data = get_gfs_data(valid_time)

    # Return data or error
    if isinstance(data, tuple):
        return jsonify(data[0]), data[1]  # Return error message and status code
    elif data:
        return jsonify(data)
    else:
        return jsonify({'error': 'Failed to retrieve weather data'}), 500


@app.route('/api/weatherData/1day')
def weather_data_1day():
    return weather_data()

@app.route('/api/weatherData/3day')
def weather_data_3day():
    return weather_data()

@app.route('/api/weatherData/7day')
def weather_data_7day():
    return weather_data()


if __name__ == '__main__':
    app.run(debug=True, port=5000)
