import os
import json
import time
import requests
import datetime
import matplotlib
import cartopy
import warnings
import subprocess
import numpy as np

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

import util.plot_util
# import util.constants

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Imagery:

    """
    A class for handling imagery data.

    Args:
        data_source (str): The source of the imagery data.
        satellite (str): The satellite used to capture the imagery.
        acq_dt (str): The acquisition datetime of the imagery.
        outdir (str): The output directory for saving the processed imagery.
        geojson_fpath (str): The file path to the GeoJSON file containing the polygon coordinates.
        buoys (str): The file path to the JSON file containing buoy URLs.
        mode (str): The mode of operation - one of 'lincoln', 'canada', or 'baffin'
        quicklook_fdir (str, optional): The path to the directory where quicklook images.
                                        Defaults to None, i.e., will not generate quicklooks.
        verbose (bool, optional): Whether to display verbose output. Defaults to False.
    """
    def __init__(self,
                 data_source,
                 satellite,
                 acq_dt,
                 outdir,
                 geojson_fpath,
                 buoys,
                 norway_ship,
                 odin_ship,
                 mode,
                 quicklook_fdir,
                 verbose=False):

        self.data_source     = data_source
        self.satellite       = satellite
        self.acq_dt          = acq_dt
        self.outdir          = outdir
        self.geojson_fpath   = geojson_fpath
        self.buoys           = buoys
        self.norway_ship     = norway_ship
        self.odin_ship       = odin_ship
        self.mode            = mode
        self.quicklook_fdir  = quicklook_fdir
        self.verbose         = verbose

        self.get_instrument()
        if self.buoys is not None:
            self.buoy_dts, self.buoy_lons, self.buoy_lats = self.get_buoy_data()

        if self.norway_ship is not None:
            self.norway_lons, self.norway_lats = self.get_norway_icebreaker_data()

        if self.odin_ship is not None:
            self.odin_lons, self.odin_lats = self.get_odin_icebreaker_data()


    def scale_table(self, arr):
        """
        Scaling table (non-clouds) used by WorldView
        0   0
        30 110
        60 160
        120 210
        190 240
        255 255
        """
        arr[arr == 0] = 0
    #     arr[arr == 30] = 110
    #     arr[arr == 60] = 160
    #     arr[arr == 120] = 210
    #     arr[arr == 190] = 240
        arr[arr == 255] = 255
        return arr


    def doy2date_str(self, acq_dt):
        year, doy, hours, minutes = acq_dt[1:5], acq_dt[5:8], acq_dt[9:11], acq_dt[11:13]
        date = datetime.datetime.strptime('{} {} {} {}'.format(year, doy, hours, minutes), '%Y %j %H %M')
        return date.strftime('%B %d, %Y: %H%M')

    def ql_doy2date_str(self, acq_dt):
        """for quicklooks"""
        acq_dt = acq_dt[1:]
        date = datetime.datetime.strptime(acq_dt, '%Y%j.%H%M')
        return date.strftime('%Y-%m-%d-%H%M%SZ')


    def format_acq_dt(self, acq_dt):
        """format acquisition datetime for filename """
        year, doy, hours, minutes = acq_dt[1:5], acq_dt[5:8], acq_dt[9:11], acq_dt[11:13]
        date = datetime.datetime.strptime(year+doy, '%Y%j').date()
        return date.strftime('%Y-%m-%d') + '-{}{}Z'.format(hours, minutes)


    def normalize_data(self, data):
        return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


    def scale_255(self, arr, vmax):
        return ((arr - np.nanmin(arr)) * (1/((vmax - np.nanmin(arr))) * 255)).astype('uint8')


    def preprocess_band(self, band, sza=None, scale=False):

        if sza is not None: # can be None for radiance
            if isinstance(sza, np.ma.MaskedArray):
                np.ma.set_fill_value(sza, np.nan) # in-place
                sza = sza.filled() # get rid of mask

            band = band/np.cos(np.deg2rad(sza))

        # handle outlier pixels by essentially discarding them
        vmax = np.nanpercentile(band, 99)
        band = np.clip(band, 0., vmax)
        if scale: # for RGB integer uint8
            band_mask = np.ma.masked_where((np.isnan(band)), band)
            band = self.scale_255(band, vmax)
            band = np.ma.masked_where(np.ma.getmask(band_mask), band)
            return band
        else: # for all others
            band = (band - np.nanmin(band)) / (vmax - np.nanmin(band))
            band = np.ma.masked_where((np.isnan(band)), band)
            return band


    def create_3d_mask(self, red, green, blue):
        """ create 3d mask for plotting NaNs and data"""
        red_mask, green_mask, blue_mask = red.mask, green.mask, blue.mask
        if red_mask.size != red.data.size:
            red_mask = np.full(red.shape, False)
        if green_mask.size != green.data.size:
            green_mask = np.full(green.shape, False)
        if blue_mask.size != blue.data.size:
            blue_mask = np.full(blue.shape, False)

        return np.stack([red_mask, green_mask, blue_mask], axis=-1)


    def get_instrument(self):
        if (self.satellite == 'Aqua') or (self.satellite == 'Terra'):
            self.instrument = 'MODIS'
        elif (self.satellite == 'Suomi-NPP') or (self.satellite == 'NOAA-20/JPSS-1') or (self.satellite == 'NOAA-21/JPSS-2'):
            self.instrument = 'VIIRS'
        else:
            self.instrument = 'Unknown'


    def get_norway_icebreaker_data(self, update_threshold_hrs=1, force_download=False):
        """
        Retrieves the Norway icebreaker data from a specified URL or a local JSON file.

        Args:
            update_threshold_hrs (int, optional): The time threshold in hours for updating the data. If the difference between current time and last downloaded time exceeds this value, data will be update via the URL download. Defaults to 1 i.e., updates hourly.
            force_download (bool, optional): Whether to force download the data even if a local file exists. Defaults to False.

        Returns:
            tuple: A tuple containing the longitude and latitude coordinates of the icebreaker.

        Live Map: http://www.padodd.com/arcticmap/
        """

        url = 'http://vessel.npolar.io:8443/?token=asdfghjkl&mmsi=257275000'

        utc_now_dt = datetime.datetime.now(datetime.timezone.utc)
        utc_now_dt = utc_now_dt.replace(tzinfo=None) # so that timedelta does not raise an error

        if force_download:
            try:
                data = requests.get(url, timeout=(5, 10)).json()
                slon, slat = data['geometry']['coordinates']
                with open(self.norway_ship, 'w') as f: # save for next time
                    json.dump(data, f)

            except Exception as api_err:
                if self.verbose:
                    print("Message [get_norway_icebreaker_data] Following error occurred when downloading data: {}\n Will attempt to read from local file instead...".format(api_err))
                try:
                    with open(self.norway_ship, 'r') as f:
                        data = json.load(f)
                    slon, slat = data['geometry']['coordinates']

                except Exception as json_err:
                    print("Error [get_norway_icebreaker_data] Following error occurred when reading from local file: {}\n Defaulting to (nan, nan)..".format(json_err))
                    slon, slat = np.nan, np.nan

            return slon, slat


        if os.path.isfile(self.norway_ship):
            with open(self.norway_ship, 'r') as f:
                data = json.load(f)

            # if less than time threshold, read old data; otherwise download from server and save to file
            try: # seems to be fluctuating between different time formats
                last_checked_dt = datetime.datetime.strptime(data['properties']['datetime_utc'], '%Y-%m-%dT%H:%M:%S.%fZ')
            except Exception as time_err:
                if self.verbose:
                    print('Message [get_norway_icebreaker_data]: Error with time: {}, trying with different format...'.format(time_err))
                try:
                    last_checked_dt = datetime.datetime.strptime(data['properties']['datetime_utc'], '%Y-%m-%dT%H:%M:%SZ')
                except Exception as another_time_err:
                    print('Error [get_norway_icebreaker_data]: Error with time again: {}, returning nan, nan'.format(another_time_err))
                    return np.nan, np.nan

            if ((utc_now_dt - last_checked_dt) < datetime.timedelta(hours=update_threshold_hrs)) and (not force_download):
                slon, slat = data['geometry']['coordinates']

            else:
                try:
                    data = requests.get(url, timeout=(5, 10)).json()
                    slon, slat = data['geometry']['coordinates']
                    with open(self.norway_ship, 'w') as f: # save for next time
                        json.dump(data, f)
                except Exception as api_err:
                    if self.verbose:
                        print("Message [get_norway_icebreaker_data] Following error occurred when downloading data: {}\n Will attempt to read from local file instead...".format(api_err))
                    try:
                        with open(self.norway_ship, 'r') as f:
                            data = json.load(f)
                        slon, slat = data['geometry']['coordinates']

                    except Exception as json_err:
                        print("Error [get_norway_icebreaker_data] Following error occurred when reading from local file: {}\n Defaulting to (nan, nan)..".format(json_err))
                        slon, slat = np.nan, np.nan

        else:
            try:
                data = requests.get(url, timeout=(5, 10)).json()
                slon, slat = data['geometry']['coordinates']
                with open(self.norway_ship, 'w') as f: # save for next time
                    json.dump(data, f)
            except Exception as api_err:
                if self.verbose:
                    print("Message [get_norway_icebreaker_data] Following error occurred when downloading data: {}\n Will attempt to read from local file instead...".format(api_err))
                try:
                    with open(self.norway_ship, 'r') as f:
                        data = json.load(f)
                    slon, slat = data['geometry']['coordinates']

                except Exception as json_err:
                    print("Error [get_norway_icebreaker_data] Following error occurred when reading from local file: {}\n Defaulting to (nan, nan)..".format(json_err))
                    slon, slat = np.nan, np.nan

        # so that it does not go out of the map bounds and stretch the image
        if self.mode == 'lincoln':
            if (slon <= -125) or (slon >= 49) or (slat <= 76) or (slat >= 89.75):
                if self.verbose:
                    print("Message [get_norway_icebreaker_data]: Coordinates out of bounds, will not be plotted.")
                slon = np.nan
                slat = np.nan

        elif (self.mode == 'canada') or (self.mode == 'platypus') or (self.mode == 'ca_archipelago'):
            if (slon <= -140) or (slon >= -50) or (slat <= 76) or (slat >= 82):
                if self.verbose:
                    print("Message [get_norway_icebreaker_data]: Coordinates out of bounds, will not be plotted.")
                slon = np.nan
                slat = np.nan

        elif (self.mode == 'baffin') or (self.mode == 'baffin_bay'):
            if (slon <= -80) or (slon >= -50) or (slat <= 67) or (slat >= 81):
                if self.verbose:
                    print("Message [get_norway_icebreaker_data]: Coordinates out of bounds, will not be plotted.")
                slon = np.nan
                slat = np.nan

        else:
            if self.verbose:
                print("Message [get_norway_icebreaker_data]: Region not valid, will not be plotted.")
            slon, slat = np.nan, np.nan

        return slon, slat


    def get_odin_icebreaker_data(self, update_threshold_hrs=1, force_download=False):
        """
        Retrieves the latest icebreaker data for Odin.

        Args:
            update_threshold_hrs (int, optional): The time threshold in hours. If the time since the last data retrieval is less than this threshold, the function will return the previously retrieved data. Otherwise, it will download new data from the server. Defaults to 1.
            force_download (bool, optional): If set to True, the function will always download new data from the server, regardless of the time threshold. Defaults to False.

        Returns:
            float: The longitude of the icebreaker's location.
            float: The latitude of the icebreaker's location.
        """

        utc_now_dt = datetime.datetime.now(datetime.timezone.utc)
        utc_now_dt = utc_now_dt.replace(tzinfo=None) # so that timedelta does not raise an error
        start_dt   = utc_now_dt - datetime.timedelta(days=2)

        start_dt_str = start_dt.strftime("%Y-%m-%d")
        end_dt_str = utc_now_dt.strftime("%Y-%m-%d")

        url = 'https://sbd.arcticmarinesolutions.se/api/v1/buoy/statuses/300434066003000/{}/{}'.format(start_dt_str, end_dt_str)

        command = ['curl', '--header', 'Content-Type: application/json;charset=UTF-8', '--header', 'Authorization: Basic c2JkOjhmNGNmODMzYWRiZWI5NmNkZTVhOGVhZTg0MTM4YzNhMTU0NTIyZWI=', '{}'.format(url)]


        if force_download:
            try:
                # result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                result = subprocess.run(command, capture_output=True, check=True, text=True)
                coords = json.loads(result.stdout)
                data = coords[-1] # get only the latest coordinates
                slon, slat = data['lon'], data['lat']

                with open(self.odin_ship, 'w') as f: # save for next time
                    json.dump(data, f)

            except Exception as api_err:
                if self.verbose:
                    print("Message [get_odin_icebreaker_data] Following error occurred when downloading data: {}\n Will attempt to read from local file instead...".format(api_err))
                try:
                    with open(self.odin_ship, 'r') as f:
                        data = json.load(f)
                    slon, slat = data['lon'], data['lat']

                except Exception as json_err:
                    print("Error [get_odin_icebreaker_data] Following error occurred when reading from local file: {}\n Defaulting to (nan, nan)..".format(json_err))
                    slon, slat = np.nan, np.nan

            return slon, slat


        if os.path.isfile(self.odin_ship):
            with open(self.odin_ship, 'r') as f:
                data = json.load(f)

            # if less than time threshold, read old data; otherwise download from server and save to file
            try: # seems to be fluctuating between different time formats
                last_checked_dt = datetime.datetime.strptime(data['timestamp'], '%Y-%m-%dT%H:%M:%SZ')

            except Exception as time_err:
                if self.verbose:
                    print('Message [get_odin_icebreaker_data]: Error with time: {}, trying with different format...'.format(time_err))
                try:
                    last_checked_dt = datetime.datetime.strptime(data['timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ')
                except Exception as another_time_err:
                    print('Error [get_odin_icebreaker_data]: Error with time again: {}, returning nan, nan'.format(another_time_err))
                    return np.nan, np.nan

            if ((utc_now_dt - last_checked_dt) < datetime.timedelta(hours=update_threshold_hrs)) and (not force_download):
                slon, slat = data['lon'], data['lat']

            else:
                try:
                    result = subprocess.run(command, capture_output=True, check=True, text=True)
                    coords = json.loads(result.stdout)
                    data = coords[-1] # get only the latest coordinates
                    slon, slat = data['lon'], data['lat']
                    with open(self.odin_ship, 'w') as f: # save for next time
                        json.dump(data, f)

                except Exception as api_err:
                    if self.verbose:
                        print("Message [get_odin_icebreaker_data] Following error occurred when downloading data: {}\n Will attempt to read from local file instead...".format(api_err))
                    try:
                        with open(self.odin_ship, 'r') as f:
                            data = json.load(f)
                        slon, slat = data['lon'], data['lat']

                    except Exception as json_err:
                        print("Error [get_odin_icebreaker_data] Following error occurred when reading from local file: {}\n Defaulting to (nan, nan)..".format(json_err))
                        slon, slat = np.nan, np.nan

        else:
            try:
                result = subprocess.run(command, capture_output=True, check=True, text=True)
                coords = json.loads(result.stdout)
                data = coords[-1] # get only the latest coordinates
                slon, slat = data['lon'], data['lat']
                with open(self.odin_ship, 'w') as f: # save for next time
                    json.dump(data, f)

            except Exception as api_err:
                if self.verbose:
                    print("Message [get_odin_icebreaker_data] Following error occurred when downloading data: {}\n Will attempt to read from local file instead...".format(api_err))
                try:
                    with open(self.odin_ship, 'r') as f:
                        data = json.load(f)
                    slon, slat = data['lon'], data['lat']

                except Exception as json_err:
                    print("Error [get_odin_icebreaker_data] Following error occurred when reading from local file: {}\n Defaulting to (nan, nan)..".format(json_err))
                    slon, slat = np.nan, np.nan

        # so that it does not go out of the map bounds and stretch the image
        if self.mode == 'lincoln':
            if (slon <= -125) or (slon >= 49) or (slat <= 76) or (slat >= 89.75):
                if self.verbose:
                    print("Message [get_odin_icebreaker_data]: Coordinates out of bounds, will not be plotted.")
                slon = np.nan
                slat = np.nan

        elif (self.mode == 'canada') or (self.mode == 'platypus') or (self.mode == 'ca_archipelago'):
            if (slon <= -140) or (slon >= -50) or (slat <= 76) or (slat >= 82):
                if self.verbose:
                    print("Message [get_odin_icebreaker_data]: Coordinates out of bounds, will not be plotted.")
                slon = np.nan
                slat = np.nan

        elif (self.mode == 'baffin') or (self.mode == 'baffin_bay'):
            if (slon <= -80) or (slon >= -50) or (slat <= 67) or (slat >= 81):
                if self.verbose:
                    print("Message [get_odin_icebreaker_data]: Coordinates out of bounds, will not be plotted.")
                slon = np.nan
                slat = np.nan

        else:
            if self.verbose:
                print("Message [get_odin_icebreaker_data]: Region not valid, will not be plotted.")
            slon, slat = np.nan, np.nan

        return slon, slat


    def get_buoy_data(self, API_KEY="kU7vw3YBcIvnxqTH8DlDeR08QTTfNYiZ", days=7, download_threshold_hrs=6, force_download=False, force_local=False):
        """
        Retrieves buoy data from a JSON file or downloads it from a server.

        Args:
            json_fpath (str): The file path to the JSON file containing the buoy data.
            API_KEY (str, optional): The API key for accessing the server. Obtain one from your cryosphere innovation account.
            days (int, optional): The number of days of data to retrieve. Defaults to 7.
            download_threshold_hrs (int, optional): The time threshold in hours for checking if the data should be downloaded. Defaults to 6.
            force_download (bool, optional): Whether to force download the data even if it is available locally. Defaults to False.

        Returns:
            - dts: A dictionary mapping buoy names to lists of datetime objects representing the timestamps of the data.
            - lons: A dictionary mapping buoy names to lists of longitude values.
            - lats: A dictionary mapping buoy names to lists of latitude values.
        """
        import pandas as pd

        fdir = os.path.dirname(os.path.abspath(self.buoys))
        fields_str = '?field=time_stamp&field=latitude&field=longitude'
        # for field in fields:
        #     fields_str += 'field={}&'.format(field)

        with open(self.buoys, 'r') as fp:
            urls = json.load(fp)

        # read metadata
        buoy_metadata_json = os.path.join(fdir, 'metadata.json')

        meta = {}
        if os.path.isfile(buoy_metadata_json):
            with open(buoy_metadata_json, "r") as fm:
                meta = json.load(fm)

        if meta is None:
            meta = {}

        utc_now_dt = datetime.datetime.now(datetime.timezone.utc)
        utc_now_dt = utc_now_dt.replace(tzinfo=None) # so that timedelta does not raise an error
        dts, lons, lats = {}, {}, {}
        # meta_dict = {}
        for bname, burl in urls.items():
            url = burl + fields_str

            buoy_download_json = os.path.join(fdir, 'download_buoy{}.json'.format(bname))

            if force_local and os.path.exists(buoy_download_json):
                if self.verbose:
                    print('Message [get_buoy_data]: Reading local files only for buoy: ', bname)
                with open(buoy_download_json, 'r') as f:
                    data = json.load(f)
                    df   = pd.DataFrame(data)

            else:

                # essentially force download; read old data only if it's actually available
                check_dt = utc_now_dt - datetime.timedelta(hours=download_threshold_hrs+1)
                key = 'last_downloaded_buoy{}'.format(bname.upper())
                if os.path.isfile(buoy_download_json) and (meta is not None) and (len(meta) > 0) and (key in list(meta.keys())):
                    check_dt = datetime.datetime.strptime(meta[key], "%Y-%m-%d_%H:%M:%S")

                # if less than time threshold, read old data; otherwise download from server and save to file
                if ((utc_now_dt - check_dt) < datetime.timedelta(hours=download_threshold_hrs)) and (not force_download):
                    # print("Message [get_buoy_data]: Using existing data file for: Buoy {}".format(bname))
                    with open(buoy_download_json, 'r') as f:
                        data = json.load(f)
                        df   = pd.DataFrame(data)
                else:
                    # print("Message [get_buoy_data]: Downloading data for: Buoy {}".format(bname))
                    try:
                        data = requests.get(url, headers={'Authorization':'Bearer {}'.format(API_KEY)}, timeout=(5, 10)).json()
                        df   = pd.DataFrame(data)
                        with open(buoy_download_json, 'w') as f: # save for next time
                            json.dump(data, f)

                        meta['last_downloaded_buoy{}'.format(bname.upper())] = utc_now_dt.strftime("%Y-%m-%d_%H:%M:%S")

                    except Exception as api_err:
                        print("Message [get_buoy_data] Following error occurred when downloading data: {}\n Will attempt to read from local file instead...".format(api_err))
                        with open(buoy_download_json, 'r') as f:
                            data = json.load(f)
                            df   = pd.DataFrame(data)

            with open(buoy_metadata_json, 'w') as fm: # save metadata for next time
                json.dump(meta, fm)

            df['time_stamp'] = df['time_stamp'].apply(lambda x: datetime.datetime.fromtimestamp(time.mktime(time.gmtime(x))))
            end_dt = df['time_stamp'].iloc[-1]
            start_dt = df['time_stamp'].iloc[-1] - datetime.timedelta(days=days)
            time_logic = (df['time_stamp'] >= start_dt) & (df['time_stamp'] <= end_dt)
            df = df[time_logic].reset_index(drop=True)

            # [-180, 180] format for longitudes
            df.loc[df['longitude'] > 180, 'longitude'] -= 360.0
            # drop rows that have erroneous or irrelevant information
            # df = df.drop(df[df.longitude < 70].index)
            df = df.drop(df[(df.latitude < 75) | (df.latitude > 89) | (df.longitude > 20) | (df.longitude < -100)].index)
            df = df.dropna(subset=['time_stamp', 'longitude', 'latitude'])
            df = df.reset_index(drop=True)

            dts[bname]  = list(df['time_stamp'])
            lons[bname] = list(df['longitude'])
            lats[bname] = list(df['latitude'])

        return dts, lons, lats


    def mask_geojson(self, lon_2d, lat_2d, dat, proj_plot, proj_data):

        import json

        with open(self.geojson_fpath, 'r') as f:
            gdata = json.load(f)
            # n_coords = len(data['features'][0]['geometry']['coordinates'][0])

        coords = gdata['features'][0]['geometry']['coordinates']

        plons = np.array(coords[0])[:, 0]
        plats = np.array(coords[0])[:, 1]

        # Transform the polygon and grid points to the new orthographic projection
        pgon_ortho = proj_plot.transform_points(proj_data, plons, plats)[:, [0, 1]]
        lons_ortho, lats_ortho = proj_plot.transform_points(proj_data, lon_2d, lat_2d)[:, :, 0], proj_plot.transform_points(proj_data, lon_2d, lat_2d)[:, :, 1]

        # Flatten and stack the latitude and longitude arrays of the first array
        points = np.vstack((lons_ortho.flatten(), lats_ortho.flatten())).T

        # Create a Path object from the polygon
        path = matplotlib.path.Path(pgon_ortho)

        # Create a mask for the points in the first array that are inside the Path
        mask = path.contains_points(points)

        # Reshape the mask to the shape of the data array
        if len(dat.shape) == 2:
            mask = mask.reshape(dat.shape)
        elif len(dat.shape) == 3:
            mask = mask.reshape(dat.shape[:-1])
            mask = np.stack([mask, mask, mask], axis=-1)

        # Mask the data array with the mask
        masked_dat = np.where(mask, dat, np.nan)
        return masked_dat


    def create_metadata(self):
        now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        metadata = {'system': os.uname()[0], 'created': now, 'data_source': self.data_source, 'author': 'Vikas Nataraja'}
        return metadata


    def add_ancillary(self, ax, title=None, scale=1):
        """
        Adds ancillary features to the plot.

        Args:
            ax (matplotlib.axes.Axes): The axes object to add the features to.
            title (str, optional): The title of the plot. Defaults to None.
            scale (float, optional): The scale factor for the font sizes. Defaults to 1.
        """
        # ax.scatter(cfs_alert[0], cfs_alert[1], marker='*', s=30, color='white', transform=proj_data, zorder=2)
        # ax.text(cfs_alert[0]-0.5, cfs_alert[1]-0.5, "Stn. Alert", ha="center", color='white', transform=proj_data,
        #         fontsize=14, fontweight="bold", zorder=2)
        # ax.scatter(stn_nord[0], stn_nord[1], marker='*', s=30, color='white', transform=proj_data, zorder=2)
        # ax.text(stn_nord[0]-0.5, stn_nord[1]-0.5, "Stn. Nord\n(Villum)", ha="center", color='white', transform=proj_data,
        #         fontsize=14,fontweight="bold", zorder=2)
        # ax.scatter(thule_pituffik[0], thule_pituffik[1], marker='*', s=30, color='white', transform=proj_data, zorder=2)
        # ax.text(thule_pituffik[0]-0.5, thule_pituffik[1]-0.5, "Thule\n(Pituffik)", ha="center", color='white', transform=proj_data,
        #         fontsize=14, fontweight="bold", zorder=2)


        # set title manually because of boundary
        if title is not None:
            # ax.text(0.5, 0.95, title, ha="center", color='white', fontsize=20, fontweight="bold", transform=ax.transAxes,
            #         bbox=dict(facecolor='black', boxstyle='round', pad=1))
            title_fontsize = int(22 * scale)

            ax.set_title(title, pad=10, fontsize=title_fontsize, fontweight="bold")

        ax.add_feature(cartopy.feature.OCEAN.with_scale('50m'), zorder=0, facecolor='black', edgecolor='none')
        ax.add_feature(cartopy.feature.LAND.with_scale('50m'), zorder=0, facecolor='black', edgecolor='none')
        ax.add_feature(cartopy.feature.COASTLINE.with_scale('50m'), zorder=2, edgecolor='darkgray', linewidth=1, alpha=1)

        # ax.set_aspect("auto")

        gl = ax.gridlines(linewidth=1.5, color='darkgray',
                    draw_labels=True, zorder=2, alpha=1, linestyle=(0, (1, 1)),
                    x_inline=False, y_inline=True, crs=util.plot_util.proj_data)

        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 20))
        gl.ylocator = mticker.FixedLocator(np.arange(0, 90, 5))
        gl.xlabel_style = {'size': int(12 * scale), 'color': 'black'}
        gl.ylabel_style = {'size': int(12 * scale), 'color': 'white'}
        gl.rotate_labels = False
        gl.top_labels    = False
        gl.xpadding = 5
        gl.ypadding = 5
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

        # visualize buoys
        if self.buoys is not None:
            # dt, lons, lats = self.get_buoy_data()

            # dt_text_box = ''
            buoy_ids = list(self.buoy_dts.keys())
            colors = plt.cm.brg(np.linspace(0, 1, len(buoy_ids)))
            for i, bid in enumerate(buoy_ids):
                ax.scatter(self.buoy_lons[bid][0], self.buoy_lats[bid][0], marker='s', s=30, edgecolor='black', facecolor=colors[i], transform=util.plot_util.proj_data, zorder=2, alpha=0.7)
                ax.plot(self.buoy_lons[bid], self.buoy_lats[bid], linewidth=1.5, color=colors[i], transform=util.plot_util.proj_data, zorder=2, alpha=0.7)
                ax.scatter(self.buoy_lons[bid][-1], self.buoy_lats[bid][-1], marker='*', s=100, edgecolor='black', facecolor=colors[i], transform=util.plot_util.proj_data, zorder=2, alpha=1)

                text = str(bid)
                x_offset, y_offset = 1.5, -0.2 # offset for text
                if text == "J":
                    x_offset, y_offset = 1.5, -0.1 # buoy J is drifting east, others are drifting west for the most part
                ax.text(self.buoy_lons[bid][-1] + x_offset, self.buoy_lats[bid][-1] + y_offset, text, ha="center", va="center", transform=util.plot_util.proj_data, color=colors[i], fontsize=10, fontweight="bold", zorder=2)

        # visualize Norwegian icebreaker
        if self.norway_ship is not None:
            # slon, slat = self.get_norway_icebreaker_data()
            if (np.isnan(self.norway_lons)) or (np.isnan(self.norway_lats)):
                if self.verbose:
                    print('Message [add_ancillary]: Norwegian Icebreaker will not be plotted.')

            else:
                # diamond marker
                ax.scatter(self.norway_lons, self.norway_lats, transform=util.plot_util.proj_data, marker='D', facecolor='magenta', edgecolor='black', s=40, zorder=2, alpha=1)
                x_offset, y_offset = 1.5, -0.1
                ax.text(self.norway_lons + x_offset, self.norway_lats + y_offset, '3YYQ', color='magenta', ha="center", va="center", transform=util.plot_util.proj_data, fontsize=8, fontweight="bold", zorder=2)

        # visualize Odin icebreaker
        if self.odin_ship is not None:
            # slon, slat = self.get_odin_icebreaker_data()

            if (np.isnan(self.odin_lons)) or (np.isnan(self.odin_lats)):
                if self.verbose:
                    print('Message [add_ancillary]: Oden Icebreaker will not be plotted.')

            else:
                # diamond marker
                ax.scatter(self.odin_lons, self.odin_lats, transform=util.plot_util.proj_data, marker='D', facecolor='turquoise', edgecolor='black', s=40, zorder=2, alpha=1)
                x_offset, y_offset = 1.0, -0.1
                ax.text(self.odin_lons + x_offset, self.odin_lats + y_offset, 'Oden', color='turquoise', ha="center", va="center", transform=util.plot_util.proj_data, fontsize=8, fontweight="bold", zorder=2)

    def add_esri_features(ax, land_proj_filepath, ocean_proj_filepath, land_shapefile_path, ocean_shapefile_path, title=None, scale=1, dx=20, dy=5, cartopy_black=False, ccrs_data=None, ocean=True, gridlines=True, coastline=True, land=True, x_fontcolor='black', y_fontcolor='black', zorders={'land': 0, 'ocean': 1, 'coastline': 2, 'gridlines': 2}, colors=None, y_inline=True):
        """
        Add ESRI features and styling elements (title, ocean/land color, coastlines, gridlines) to a cartopy map plot.

        Args:
        ----
            ax: A matplotlib or cartopy axes object where the features will be drawn.
            title (str, optional): The title of the plot. Defaults to None.
            scale (float, optional): A scaling factor for text size. Defaults to 1.
            dx (int, optional): Longitude spacing in degrees. Defaults to 20.
            dy (int, optional): Latitude spacing in degrees. Defaults to 5.
            cartopy_black (bool, optional): Whether to use a black color scheme for background
                and cartographic features. Defaults to False.
            ccrs_data (cartopy.crs, optional): Coordinate reference system to use
                for the plot. Defaults to ccrs.PlateCarree().
            coastline (bool, optional): Whether to draw coastlines. Defaults to True.
            ocean (bool, optional): Whether to fill ocean areas. Defaults to True.
            gridlines (bool, optional): Whether to draw gridlines. Defaults to True.
            land (bool, optional): Whether to fill land areas. Defaults to True.
            x_fontcolor (str, optional): Font color for x-axis gridline labels. Defaults to 'black'.
            y_fontcolor (str, optional): Font color for y-axis gridline labels. Defaults to 'black'.
            zorders (dict, optional): Z-order values for different features (land, ocean, coastline,
                gridlines). Defaults to {'land': 0, 'ocean': 1, 'coastline': 2, 'gridlines': 2}.
            colors (dict, optional): Color mappings for features like ocean, land, coastline,
                title, and background. If None, defaults are used.

        Returns:
        -------
            None, modifies axis in-place
        """

        if ccrs_data is None:
            ccrs_data = ccrs.PlateCarree()

        # set title
        if title is not None:
            title_fontsize = int(18 * scale)
            ax.set_title(title, pad=7.5, fontsize=title_fontsize, fontweight="bold")

        if colors is None:
            if cartopy_black:
                colors = {'ocean':'black', 'land':'black', 'coastline':'black', 'title':'white', 'background':'black'}

            else:
                colors = {'ocean':'aliceblue', 'land':'#fcf4e8', 'coastline':'black', 'title':'black', 'background':'white'}

        if ocean:
            with open(ocean_proj_filepath) as fprj:
                oprj = fprj.read()

            ocean_feature = ShapelyFeature(Reader(ocean_shapefile_path).geometries(),
                                        crs=ccrs.CRS(oprj, globe=None),
                                        facecolor=colors['ocean'],
                                        edgecolor='none',
                                        zorder=zorders['ocean'],
                                        alpha=1)
            ax.add_feature(ocean_feature)


        if land:
            with open(land_proj_filepath) as fprj:
                lprj = fprj.read() # read the proj4 string

            land_feature = ShapelyFeature(Reader(land_shapefile_path).geometries(),
                                        crs=ccrs.CRS(lprj, globe=None), # create crs from proj string
                                        facecolor=colors['land'],
                                        edgecolor='none',
                                        zorder=zorders['land'],
                                        alpha=1)
            ax.add_feature(land_feature)

        if coastline:
            with open(land_proj_filepath) as fprj:
                cprj = fprj.read()

            coastline_feature = ShapelyFeature(Reader(land_shapefile_path).geometries(),
                                        crs=ccrs.CRS(cprj, globe=None),
                                        facecolor='none',
                                        edgecolor=colors['coastline'],
                                        linewidth=1,
                                        zorder=zorders['coastline'],
                                        alpha=1)
            ax.add_feature(coastline_feature)

        if gridlines:
            gl = ax.gridlines(linewidth=2.5, color='darkgray',
                        draw_labels=True, zorder=zorders['gridlines'], alpha=1, linestyle=(0, (1, 1)),
                        x_inline=False, y_inline=y_inline, crs=ccrs_data)

            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, dx))
            gl.ylocator = mticker.FixedLocator(np.arange(0, 90, dy))
            gl.xlabel_style = {'size': int(12 * scale), 'color': x_fontcolor}
            gl.ylabel_style = {'size': int(12 * scale), 'color': y_fontcolor}
            gl.rotate_labels = False
            gl.top_labels    = True
            gl.right_labels  = True
            gl.bottom_labels = True
            gl.left_labels  = True
            gl.xpadding = 15
            gl.ypadding = 5

        # for spine in ax.spines.values():
        #     if cartopy_black:
        #         spine.set_edgecolor('white')
        #     else:
        #         spine.set_edgecolor('black')

        #     spine.set_linewidth(1.5)


    def convert_ir_ctp(self, ctp_ir_arr):
        ctp_ir_arr[ctp_ir_arr==6] = 4
        ctp_ir_arr = np.ma.masked_where(np.isnan(ctp_ir_arr), ctp_ir_arr)
        ctp_ir_arr = ctp_ir_arr.astype('int8')
        return ctp_ir_arr


    def bin_cth_to_class(self, cth_true):
        """ Bins CTH pixels to a class and returns as a mask """
        cth_bins = np.array([0, 100., 2000., 6000.])
        classmap = np.zeros((cth_true.shape[0], cth_true.shape[1]), dtype='uint8')
        classmap[(cth_true < 0) | (np.isnan(cth_true))] = 0
        classmap[(cth_true >= cth_bins[0]) & (cth_true < cth_bins[1])] = 1
        classmap[(cth_true >= cth_bins[1]) & (cth_true < cth_bins[2])] = 2
        classmap[(cth_true >= cth_bins[2]) & (cth_true < cth_bins[3])] = 3
        classmap[(cth_true >= cth_bins[3])] = 4
        # classmap = np.ma.masked_where((np.isnan(cth_true) | (cth_true < 0)), classmap)
        # np.ma.set_fill_value(classmap, 99)
        return classmap


    def correct_ctt(self, ctt_arr, cth_binned):
        """
        mask cloud top temperature wherever there is any one or more of :
            - no cloud top height retrieval
            - nan values
            - negative values
        """
        ctt_arr = np.ma.masked_where(((ctt_arr <= 0) | (np.isnan(ctt_arr)) | (cth_binned <= 1)), ctt_arr)
        return ctt_arr


    def round_ctt(self, x, base=5):
        return base * round(x/base)


    def bin_ctt_to_class(self, ctt_true, ctt_bins, pxvals):
        """ Bins CTT pixels to a class"""

        classmap = np.zeros((ctt_true.shape[0], ctt_true.shape[1]), dtype='uint8')
        ctt_cmap_labels = []
        for k in range(pxvals.size):
            if k < (pxvals.size - 1):
                classmap[(ctt_true >= ctt_bins[k]) & (ctt_true < ctt_bins[k+1])] = pxvals[k]
                ctt_cmap_labels.append("[{}, {})".format(ctt_bins[k], ctt_bins[k+1]))
            else:
                classmap[(ctt_true >= ctt_bins[k])] = pxvals[k]
                ctt_cmap_labels.append(">={}".format(ctt_bins[k]))

        classmap = np.ma.masked_where(np.ma.getmask(ctt_true), classmap) # copy mask
        return classmap, ctt_cmap_labels



    def create_false_color_721_imagery(self, lon_2d, lat_2d, red, green, blue, sza):

        proj_plot = ccrs.Orthographic(central_longitude=util.plot_util.ccrs_views[self.mode]['vlon'], central_latitude=util.plot_util.ccrs_views[self.mode]['vlat'])

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        save_dir = os.path.join(self.outdir, "false_color_721")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_title = self.doy2date_str(self.acq_dt) + "Z"
        fname_target = self.format_acq_dt(self.acq_dt)
        sat_fname    = self.satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            if self.verbose:
                print("Message [false_color_721]: {} skipped since it already exists.".format(full_fname))
            return 0

        red    = self.preprocess_band(red,   sza=sza)
        green  = self.preprocess_band(green, sza=sza)
        blue   = self.preprocess_band(blue,  sza=sza)

        img_fci = np.stack([red, green, blue], axis=-1)

        if self.geojson_fpath is not None:
            img_fci = self.mask_geojson(lon_2d, lat_2d, img_fci, proj_plot, util.plot_util.proj_data)

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(util.plot_util.mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, img_fci,
                        shading='nearest',
                        zorder=2,
                        transform=util.plot_util.proj_data)

        title = "{} ({}) False Color (7-2-1) - ".format(self.instrument, self.satellite) + dt_title
        self.add_ancillary(ax, title=title)

        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)

        metadata = self.create_metadata()
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight", metadata=metadata)
        plt.close()

        if self.quicklook_fdir is not None: # generate quicklook imagery

            try: # reduce risk since this is lower priority
                # granule extent ignored for now since NRT 03 files don't have the right lat lon limits
                # instead preset
                # gextent = [np.nanmin(lon_2d), np.nanmax(lon_2d), np.nanmin(lat_2d), np.nanmax(lat_2d)]
                gextent = util.plot_util.ql_settings['extent']

                # generate figure
                fig = plt.figure(figsize=(12, 12))
                plt.style.use('default')
                gs  = GridSpec(1, 1, figure=fig)
                ax = fig.add_subplot(gs[0], projection=ccrs.Orthographic(central_longitude=np.mean(gextent[:2]),
                                                                         central_latitude=np.mean(gextent[2:])))

                ax.pcolormesh(lon_2d, lat_2d, img_fci,
                            shading='nearest',
                            zorder=2,
                            transform=util.plot_util.ql_settings['proj_data'])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_extent(gextent, util.plot_util.ql_settings['proj_data'])
                ax.set_aspect('auto')

                # save the figure
                ql_dt_str = self.ql_doy2date_str(self.acq_dt)
                extent_xy = list(ax.get_xlim()) + list(ax.get_ylim())
                ql_fname = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f})_({:.4f},{:.4f},{:.4f},{:.4f}).png".format(self.instrument.upper(), sat_fname.upper(), "FalseColor721", ql_dt_str, *extent_xy, *gextent)
                ql_full_fname = os.path.join(self.quicklook_fdir, ql_fname)

                fig.savefig(ql_full_fname, dpi=util.plot_util.ql_settings['dpi'], pad_inches=util.plot_util.ql_settings['pad_inches'], bbox_inches=util.plot_util.ql_settings['bbox_inches'], metadata=metadata)
                plt.close()
            except Exception as ql_err:
                print('Error [false_color_721]: Following error occurred when creating {}\n {}'.format(ql_full_fname, ql_err))

        return 1


    def create_false_color_367_imagery(self, lon_2d, lat_2d, red, green, blue, sza):

        proj_plot = ccrs.Orthographic(central_longitude=util.plot_util.ccrs_views[self.mode]['vlon'], central_latitude=util.plot_util.ccrs_views[self.mode]['vlat'])

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        save_dir = os.path.join(self.outdir, "false_color_367")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_title = self.doy2date_str(self.acq_dt) + "Z"
        fname_target = self.format_acq_dt(self.acq_dt)
        sat_fname    = self.satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            if self.verbose:
                print("Message [false_color_367]: {} skipped since it already exists.".format(full_fname))
            return 0

        red    = self.preprocess_band(red,   sza=sza)
        green  = self.preprocess_band(green, sza=sza)
        blue   = self.preprocess_band(blue,  sza=sza)

        img_fci = np.stack([red, green, blue], axis=-1)

        if self.geojson_fpath is not None:
            img_fci = self.mask_geojson(lon_2d, lat_2d, img_fci, proj_plot, util.plot_util.proj_data)

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(util.plot_util.mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, img_fci,
                        shading='nearest',
                        zorder=2,
                        transform=util.plot_util.proj_data)

        title = "{} ({}) False Color (3-6-7) - ".format(self.instrument, self.satellite) + dt_title
        self.add_ancillary(ax, title=title)

        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)

        metadata = self.create_metadata()
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight", metadata=metadata)
        plt.close()

        if self.quicklook_fdir is not None: # generate quicklook imagery

            try: # reduce risk since this is lower priority
                # granule extent ignored for now since NRT 03 files don't have the right lat lon limits
                # instead preset
                # gextent = [np.nanmin(lon_2d), np.nanmax(lon_2d), np.nanmin(lat_2d), np.nanmax(lat_2d)]
                gextent = util.plot_util.ql_settings['extent']

                # generate figure
                fig = plt.figure(figsize=(12, 12))
                plt.style.use('default')
                gs  = GridSpec(1, 1, figure=fig)
                ax = fig.add_subplot(gs[0], projection=ccrs.Orthographic(central_longitude=np.mean(gextent[:2]),
                                                                         central_latitude=np.mean(gextent[2:])))

                ax.pcolormesh(lon_2d, lat_2d, img_fci,
                            shading='nearest',
                            zorder=2,
                            transform=util.plot_util.ql_settings['proj_data'])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_extent(gextent, util.plot_util.ql_settings['proj_data'])
                ax.set_aspect('auto')

                # save the figure
                ql_dt_str = self.ql_doy2date_str(self.acq_dt)
                extent_xy = list(ax.get_xlim()) + list(ax.get_ylim())
                ql_fname = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f})_({:.4f},{:.4f},{:.4f},{:.4f}).png".format(self.instrument.upper(), sat_fname.upper(), "FalseColor367", ql_dt_str, *extent_xy, *gextent)
                ql_full_fname = os.path.join(self.quicklook_fdir, ql_fname)

                fig.savefig(ql_full_fname, dpi=util.plot_util.ql_settings['dpi'], pad_inches=util.plot_util.ql_settings['pad_inches'], bbox_inches=util.plot_util.ql_settings['bbox_inches'], metadata=metadata)
                plt.close()
            except Exception as ql_err:
                print('Error [false_color_367]: Following error occurred when creating {}\n {}'.format(ql_full_fname, ql_err))

        return 1


    def create_true_color_imagery(self, lon_2d, lat_2d, red, green, blue, sza):

        proj_plot = ccrs.Orthographic(central_longitude=util.plot_util.ccrs_views[self.mode]['vlon'], central_latitude=util.plot_util.ccrs_views[self.mode]['vlat'])

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        save_dir = os.path.join(self.outdir, "true_color")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_title = self.doy2date_str(self.acq_dt) + "Z"
        fname_target = self.format_acq_dt(self.acq_dt)
        sat_fname    = self.satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            if self.verbose:
                print("Message [true_color]: {} skipped since it already exists.".format(full_fname))
            return 0

        red    = self.preprocess_band(red,   sza=sza, scale=True)
        green  = self.preprocess_band(green, sza=sza, scale=True)
        blue   = self.preprocess_band(blue,  sza=sza, scale=True)

        rgb      = np.stack([red, green, blue], axis=-1)
        rgb_mask = self.create_3d_mask(red, green, blue)
        rgb      = np.ma.masked_array(rgb, mask=rgb_mask)
        # rgb = np.interp(rgb, (np.nanpercentile(rgb, 1), np.nanpercentile(rgb, 99)), (0, 1))

        if self.geojson_fpath is not None:
            rgb = self.mask_geojson(lon_2d, lat_2d, rgb, proj_plot, util.plot_util.proj_data)

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(util.plot_util.mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, rgb,
                        shading='nearest',
                        zorder=2,
                        # vmin=0., vmax=1.,
                        transform=util.plot_util.proj_data)

        title = "{} ({}) True Color - ".format(self.instrument, self.satellite) + dt_title
        self.add_ancillary(ax, title=title)
        ax.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)
        metadata = self.create_metadata()
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight", metadata=metadata)
        plt.close()

        if self.quicklook_fdir is not None: # generate quicklook imagery

            try: # reduce risk since this is lower priority
                # granule extent ignored for now since NRT 03 files don't have the right lat lon limits
                # instead preset
                # gextent = [np.nanmin(lon_2d), np.nanmax(lon_2d), np.nanmin(lat_2d), np.nanmax(lat_2d)]
                gextent = util.plot_util.ql_settings['extent']

                # generate figure
                fig = plt.figure(figsize=(12, 12))
                plt.style.use('default')
                gs  = GridSpec(1, 1, figure=fig)
                ax = fig.add_subplot(gs[0], projection=ccrs.Orthographic(central_longitude=np.mean(gextent[:2]),
                                                                         central_latitude=np.mean(gextent[2:])))

                ax.pcolormesh(lon_2d, lat_2d, rgb,
                            shading='nearest',
                            zorder=2,
                            transform=util.plot_util.ql_settings['proj_data'])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_extent(gextent, util.plot_util.ql_settings['proj_data'])
                ax.set_aspect('auto')

                # save the figure
                ql_dt_str = self.ql_doy2date_str(self.acq_dt)
                extent_xy = list(ax.get_xlim()) + list(ax.get_ylim())
                ql_fname = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f})_({:.4f},{:.4f},{:.4f},{:.4f}).png".format(self.instrument.upper(), sat_fname.upper(), "TrueColor", ql_dt_str, *extent_xy, *gextent)
                ql_full_fname = os.path.join(self.quicklook_fdir, ql_fname)

                fig.savefig(ql_full_fname, dpi=util.plot_util.ql_settings['dpi'], pad_inches=util.plot_util.ql_settings['pad_inches'], bbox_inches=util.plot_util.ql_settings['bbox_inches'], metadata=metadata)
                plt.close()
            except Exception as ql_err:
                print('Error [true_color]: Following error occurred when creating {}\n {}'.format(ql_full_fname, ql_err))

        return 1


    def create_false_color_cirrus_imagery(self, lon_2d, lat_2d, red, green, blue, sza):

        proj_plot = ccrs.Orthographic(central_longitude=util.plot_util.ccrs_views[self.mode]['vlon'], central_latitude=util.plot_util.ccrs_views[self.mode]['vlat'])

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        save_dir = os.path.join(self.outdir, "false_color_cirrus")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_title = self.doy2date_str(self.acq_dt) + "Z"
        fname_target = self.format_acq_dt(self.acq_dt)
        sat_fname    = self.satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            if self.verbose:
                print("Message [false_color_cirrus]: {} skipped since it already exists.".format(full_fname))
            return 0

        red    = self.preprocess_band(red,   sza=sza)
        green  = self.preprocess_band(green, sza=sza)
        blue   = self.preprocess_band(blue,  sza=sza)

        img_fci = np.stack([red, green, blue], axis=-1)

        if self.geojson_fpath is not None:
            img_fci = self.mask_geojson(lon_2d, lat_2d, img_fci, proj_plot, util.plot_util.proj_data)

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(util.plot_util.mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, img_fci,
                        shading='nearest',
                        zorder=2,
                        transform=util.plot_util.proj_data)
        # ax.set_boundary(boundary, transform=proj_data)
        if self.instrument == 'MODIS':
            title = "{} ({}) False Color (1.38-1.64-2.13$\;\mu m$) - ".format(self.instrument, self.satellite) + dt_title
        else:
            title = "{} ({}) False Color (1.38-1.61-2.25$\;\mu m$) - ".format(self.instrument, self.satellite) + dt_title

        self.add_ancillary(ax, title=title)
        ax.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)
        metadata = self.create_metadata()
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight", metadata=metadata)
        plt.close()

        if self.quicklook_fdir is not None: # generate quicklook imagery

            try: # reduce risk since this is lower priority
                # granule extent ignored for now since NRT 03 files don't have the right lat lon limits
                # instead preset
                # gextent = [np.nanmin(lon_2d), np.nanmax(lon_2d), np.nanmin(lat_2d), np.nanmax(lat_2d)]
                gextent = util.plot_util.ql_settings['extent']

                # generate figure
                fig = plt.figure(figsize=(12, 12))
                plt.style.use('default')
                gs  = GridSpec(1, 1, figure=fig)
                ax = fig.add_subplot(gs[0], projection=ccrs.Orthographic(central_longitude=np.mean(gextent[:2]),
                                                                         central_latitude=np.mean(gextent[2:])))

                ax.pcolormesh(lon_2d, lat_2d, img_fci,
                            shading='nearest',
                            zorder=2,
                            transform=util.plot_util.ql_settings['proj_data'])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_extent(gextent, util.plot_util.ql_settings['proj_data'])
                ax.set_aspect('auto')

                # save the figure
                ql_dt_str = self.ql_doy2date_str(self.acq_dt)
                extent_xy = list(ax.get_xlim()) + list(ax.get_ylim())
                ql_fname = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f})_({:.4f},{:.4f},{:.4f},{:.4f}).png".format(self.instrument.upper(), sat_fname.upper(), "FalseColorCirrus", ql_dt_str, *extent_xy, *gextent)
                ql_full_fname = os.path.join(self.quicklook_fdir, ql_fname)

                fig.savefig(ql_full_fname, dpi=util.plot_util.ql_settings['dpi'], pad_inches=util.plot_util.ql_settings['pad_inches'], bbox_inches=util.plot_util.ql_settings['bbox_inches'], metadata=metadata)
                plt.close()
            except Exception as ql_err:
                print('Error [false_color_cirrus]: Following error occurred when creating {}\n {}'.format(ql_full_fname, ql_err))

        return 1


    def create_false_color_ir_imagery(self, lon_2d, lat_2d, red, green, blue):

        proj_plot = ccrs.Orthographic(central_longitude=util.plot_util.ccrs_views[self.mode]['vlon'], central_latitude=util.plot_util.ccrs_views[self.mode]['vlat'])

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        save_dir = os.path.join(self.outdir, "false_color_ir")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_title = self.doy2date_str(self.acq_dt) + "Z"
        fname_target = self.format_acq_dt(self.acq_dt)
        sat_fname    = self.satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            if self.verbose:
                print("Message [false_color_ir]: {} skipped since it already exists.".format(full_fname))
            return 0

        # radiance so no cosine sza correction needed
        red    = self.preprocess_band(red,   sza=None)
        green  = self.preprocess_band(green, sza=None)
        blue   = self.preprocess_band(blue,  sza=None)

        img_fci = np.stack([red, green, blue], axis=-1)

        if self.geojson_fpath is not None:
            img_fci = self.mask_geojson(lon_2d, lat_2d, img_fci, proj_plot, util.plot_util.proj_data)

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(util.plot_util.mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, img_fci,
                        shading='nearest',
                        zorder=2,
                        transform=util.plot_util.proj_data)
        # ax.set_boundary(boundary, transform=proj_data)
        if self.instrument == 'MODIS':
            title = "{} ({}) False Color (11.03-1.64-2.13$\;\mu m$) - ".format(self.instrument, self.satellite) + dt_title
        else:
            title = "{} ({}) False Color (10.76-1.61-2.25$\;\mu m$) - ".format(self.instrument, self.satellite) + dt_title

        self.add_ancillary(ax, title=title)
        ax.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)
        metadata = self.create_metadata()
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight", metadata=metadata)
        plt.close()

        if self.quicklook_fdir is not None: # generate quicklook imagery

            try: # reduce risk since this is lower priority
                # granule extent ignored for now since NRT 03 files don't have the right lat lon limits
                # instead preset
                # gextent = [np.nanmin(lon_2d), np.nanmax(lon_2d), np.nanmin(lat_2d), np.nanmax(lat_2d)]
                gextent = util.plot_util.ql_settings['extent']

                # generate figure
                fig = plt.figure(figsize=(12, 12))
                plt.style.use('default')
                gs  = GridSpec(1, 1, figure=fig)
                ax = fig.add_subplot(gs[0], projection=ccrs.Orthographic(central_longitude=np.mean(gextent[:2]),
                                                                         central_latitude=np.mean(gextent[2:])))

                ax.pcolormesh(lon_2d, lat_2d, img_fci,
                            shading='nearest',
                            zorder=2,
                            transform=util.plot_util.ql_settings['proj_data'])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_extent(gextent, util.plot_util.ql_settings['proj_data'])
                ax.set_aspect('auto')

                # save the figure
                ql_dt_str = self.ql_doy2date_str(self.acq_dt)
                extent_xy = list(ax.get_xlim()) + list(ax.get_ylim())
                ql_fname = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f})_({:.4f},{:.4f},{:.4f},{:.4f}).png".format(self.instrument.upper(), sat_fname.upper(), "FalseColorIR", ql_dt_str, *extent_xy, *gextent)
                ql_full_fname = os.path.join(self.quicklook_fdir, ql_fname)

                fig.savefig(ql_full_fname, dpi=util.plot_util.ql_settings['dpi'], pad_inches=util.plot_util.ql_settings['pad_inches'], bbox_inches=util.plot_util.ql_settings['bbox_inches'], metadata=metadata)
                plt.close()
            except Exception as ql_err:
                print('Error [false_color_ir]: Following error occurred when creating {}\n {}'.format(ql_full_fname, ql_err))

        return 1


    def plot_liquid_water_paths(self, lon_2d, lat_2d, ctp, cwp, cwp_1621):

        #     " The values in this SDS are set to mean the following:                              \n",
        #     " 0 -- cloud mask undetermined                                                       \n",
        #     " 1 -- clear sky                                                                     \n",
        #     " 2 -- liquid water cloud                                                            \n",
        #     " 3 -- ice cloud                                                                     \n",
        #     " 4 -- undetermined phase cloud (but retrieval is attempted as  liquid water)        \n",

        proj_plot = ccrs.Orthographic(central_longitude=util.plot_util.ccrs_views[self.mode]['vlon'], central_latitude=util.plot_util.ccrs_views[self.mode]['vlat'])

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        save_dir = os.path.join(self.outdir, "water_path")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_title  = self.doy2date_str(self.acq_dt) + "Z"
        fname_target = self.format_acq_dt(self.acq_dt)
        sat_fname    = self.satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            if self.verbose:
                print("Message [liquid_water_path]: {} skipped since it already exists.".format(full_fname))
            return 0

        # pha            = ctp[acq_dt]
        liquid_logic   = np.where((ctp == 2) | (ctp == 4)) # liquid water
        ice_logic      = np.where((ctp == 3))
        # undt_logic     = np.where((pha == 0))
        clear_logic    = np.where((ctp == 1))

        # standard retrieval liquid water
        # cloud_wp       = cwp[acq_dt]
        im_lwp         = np.empty(cwp.shape)
        im_lwp[:]      = np.nan
        im_lwp[liquid_logic] = cwp[liquid_logic]
        im_lwp[clear_logic]  = 0.
        im_lwp[ice_logic]    = 0.

        # 1621 retrieval liquid water
        # cloud_wp_1621  = cwp_1621[acq_dt]
        im_lwp_1621    = np.empty(cwp_1621.shape)
        im_lwp_1621[:] = np.nan
        im_lwp_1621[liquid_logic] = cwp_1621[liquid_logic]
        im_lwp_1621[clear_logic]  = 0.
        im_lwp_1621[ice_logic]    = 0.

        vmax = 500.
        if vmax is not None:
            im_lwp[im_lwp > vmax]           = vmax
            im_lwp_1621[im_lwp_1621 > vmax] = vmax
            cbar_ticks = np.linspace(0, vmax, 5, dtype='int')
            cmap = util.plot_util.arctic_cloud_cmap
            extend = 'max'
        else:
            cbar_ticks = np.linspace(0, np.nanmax([im_lwp, im_lwp_1621]), 5, dtype='int')
            cmap = util.plot_util.arctic_cloud_alt_cmap
            extend = 'neither'

        if self.geojson_fpath is not None:
            im_lwp      = self.mask_geojson(lon_2d, lat_2d, im_lwp, proj_plot, util.plot_util.proj_data)
            im_lwp_1621 = self.mask_geojson(lon_2d, lat_2d, im_lwp_1621, proj_plot, util.plot_util.proj_data)

        ##############################################################

        fig = plt.figure(figsize=(40, 40))
        plt.style.use(util.plot_util.mpl_style)
        gs  = GridSpec(1, 2, figure=fig)
        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_lwp,
                            shading='nearest',
                            zorder=2,
                            cmap=cmap,
                            transform=util.plot_util.proj_data)
        # ax00.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) LWP - ".format(self.instrument, self.satellite) + dt_title
        self.add_ancillary(ax00, title=title, scale=1.4)
        cbar = fig.colorbar(y00, ax=ax00, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.05, shrink=0.45)
        cbar.ax.set_title('$LWP \;\;[g/m^2]$', fontsize=24)
        cbar.ax.tick_params(length=0, labelsize=24)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        ax00.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)
        ax00.set_aspect(1.25)

        ##############################################################

        ax01 = fig.add_subplot(gs[1], projection=proj_plot)
        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_lwp_1621,
                        shading='nearest',
                        zorder=2,
                        cmap=cmap,
                        transform=util.plot_util.proj_data)
        # ax01.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) LWP (1621) - ".format(self.instrument, self.satellite) + dt_title
        self.add_ancillary(ax01, title=title, scale=1.4)
        cbar = fig.colorbar(y01, ax=ax01, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.05, shrink=0.45)
        cbar.ax.set_title('$LWP \;\;[g/m^2]$', fontsize=24)
        cbar.ax.tick_params(length=0, labelsize=24)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        ax01.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)
        ax01.set_aspect(1.25)

        ##############################################################

        fig.subplots_adjust(wspace=0.1)
        metadata = self.create_metadata()
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight", metadata=metadata)
        plt.close()
        return 1


    def plot_ice_water_paths(self, lon_2d, lat_2d, ctp, cwp, cwp_1621):

        #     " The values in this SDS are set to mean the following:                              \n",
        #     " 0 -- cloud mask undetermined                                                       \n",
        #     " 1 -- clear sky                                                                     \n",
        #     " 2 -- liquid water cloud                                                            \n",
        #     " 3 -- ice cloud                                                                     \n",
        #     " 4 -- undetermined phase cloud (but retrieval is attempted as  liquid water)        \n",

        proj_plot = ccrs.Orthographic(central_longitude=util.plot_util.ccrs_views[self.mode]['vlon'], central_latitude=util.plot_util.ccrs_views[self.mode]['vlat'])

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        save_dir = os.path.join(self.outdir, "ice_path")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_title  = self.doy2date_str(self.acq_dt) + "Z"
        fname_target = self.format_acq_dt(self.acq_dt)
        sat_fname    = self.satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            if self.verbose:
                print("Message [ice_water_path]: {} skipped since it already exists.".format(full_fname))
            return 0

        # pha            = ctp[acq_dt]
        liquid_logic   = np.where((ctp == 2) | (ctp == 4)) # liquid water
        ice_logic      = np.where((ctp == 3))
        # undt_logic     = np.where((pha == 0))
        clear_logic    = np.where((ctp == 1))

        # standard retrieval liquid water
        # cloud_wp       = cwp[acq_dt]

        # standard retrieval ice logic
        im_iwp         = np.empty(cwp.shape)
        im_iwp[:]      = np.nan
        im_iwp[ice_logic] = cwp[ice_logic]
        im_iwp[clear_logic]  = 0.
        im_iwp[liquid_logic] = 0.


        # cloud_wp_1621  = cwp_1621[acq_dt]

        # 1621 retrieval ice logic
        im_iwp_1621         = np.empty(cwp_1621.shape)
        im_iwp_1621[:]      = np.nan
        im_iwp_1621[ice_logic] = cwp_1621[ice_logic]
        im_iwp_1621[clear_logic]  = 0.
        im_iwp_1621[liquid_logic] = 0.

        vmax = 500.
        if vmax is not None:
            im_iwp[im_iwp > vmax]           = vmax
            im_iwp_1621[im_iwp_1621 > vmax] = vmax
            cbar_ticks = np.linspace(0, vmax, 5, dtype='int')
            cmap = util.plot_util.arctic_cloud_cmap
            extend = 'max'
        else:
            cbar_ticks = np.linspace(0, np.nanmax([im_iwp_1621, im_iwp_1621]), 5, dtype='int')
            cmap = util.plot_util.arctic_cloud_alt_cmap
            extend = 'neither'

        if self.geojson_fpath is not None:
            im_iwp      = self.mask_geojson(lon_2d, lat_2d, im_iwp, proj_plot, util.plot_util.proj_data)
            im_iwp_1621 = self.mask_geojson(lon_2d, lat_2d, im_iwp_1621, proj_plot, util.plot_util.proj_data)

        fig = plt.figure(figsize=(40, 40))
        plt.style.use(util.plot_util.mpl_style)
        gs  = GridSpec(1, 2, figure=fig)
        ##############################################################

        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_iwp,
                            shading='nearest',
                            zorder=2,
                            cmap=cmap,
                            transform=util.plot_util.proj_data)
        title = "{} ({}) IWP - ".format(self.instrument, self.satellite) + dt_title
        self.add_ancillary(ax00, title=title, scale=1.4)
        cbar = fig.colorbar(y00, ax=ax00, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.05, shrink=0.45)
        cbar.ax.set_title('$IWP \;\;[g/m^2]$', fontsize=24)
        cbar.ax.tick_params(length=0, labelsize=24)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        ax00.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)
        ax00.set_aspect(1.25)

        ##############################################################

        ax01 = fig.add_subplot(gs[1], projection=proj_plot)
        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_iwp_1621,
                            shading='nearest',
                            zorder=2,
                            cmap=cmap,
                            transform=util.plot_util.proj_data)
        title = "{} ({}) IWP (1621) - ".format(self.instrument, self.satellite) + dt_title
        self.add_ancillary(ax01, title=title, scale=1.4)
        cbar = fig.colorbar(y01, ax=ax01, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.05, shrink=0.45)
        cbar.ax.set_title('$IWP \;\;[g/m^2]$', fontsize=24)
        cbar.ax.tick_params(length=0, labelsize=24)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        ax01.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)
        ax01.set_aspect(1.25)

        ##############################################################

        fig.subplots_adjust(wspace=0.1)
        metadata = self.create_metadata()
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight", metadata=metadata)
        plt.close()
        return 1



    def plot_optical_depths(self, lon_2d, lat_2d, cot, cot_1621):

        proj_plot = ccrs.Orthographic(central_longitude=util.plot_util.ccrs_views[self.mode]['vlon'], central_latitude=util.plot_util.ccrs_views[self.mode]['vlat'])

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        save_dir = os.path.join(self.outdir, "optical_thickness")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        vmax = 100.

        dt_title     = self.doy2date_str(self.acq_dt) + "Z"
        fname_target = self.format_acq_dt(self.acq_dt)
        sat_fname    = self.satellite.split('/')[0]
        full_fname   = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            if self.verbose:
                print("Message [optical_depths]: {} skipped since it already exists.".format(full_fname))
            return 0

        im_cot      = cot
        im_cot_1621 = cot_1621

        if vmax is not None:
            im_cot[im_cot > vmax]           = vmax
            im_cot_1621[im_cot_1621 > vmax] = vmax
            cbar_ticks = np.linspace(0, vmax, 5, dtype='int')
            cmap = util.plot_util.arctic_cloud_cmap
            extend = 'max'
        else:
            cbar_ticks = np.linspace(0, np.nanmax([im_cot, im_cot_1621]), 5, dtype='int')
            cmap = util.plot_util.arctic_cloud_alt_cmap
            extend = 'neither'

        if self.geojson_fpath is not None:
            im_cot      = self.mask_geojson(lon_2d, lat_2d, im_cot, proj_plot, util.plot_util.proj_data)
            im_cot_1621 = self.mask_geojson(lon_2d, lat_2d, im_cot_1621, proj_plot, util.plot_util.proj_data)

        fig = plt.figure(figsize=(40, 40))
        plt.style.use(util.plot_util.mpl_style)
        gs  = GridSpec(1, 2, figure=fig)
        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_cot,
                            shading='nearest',
                            zorder=2,
                            cmap=cmap,
                            transform=util.plot_util.proj_data)
        # ax00.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) COT - ".format(self.instrument, self.satellite) + dt_title
        self.add_ancillary(ax00, title=title, scale=1.4)
        cbar = fig.colorbar(y00, ax=ax00, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.05, shrink=0.45)
        cbar.ax.set_title('$COT$', fontsize=24)
        cbar.ax.tick_params(length=0, labelsize=24)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        ax00.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)
        ax00.set_aspect(1.25)

        ##############################################################

        ax01 = fig.add_subplot(gs[1], projection=proj_plot)
        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_cot_1621,
                            shading='nearest',
                            zorder=2,
                            cmap=cmap,
                            transform=util.plot_util.proj_data)

        title = "{} ({}) COT (1621) - ".format(self.instrument, self.satellite) + dt_title
        self.add_ancillary(ax01, title=title, scale=1.4)
        cbar = fig.colorbar(y01, ax=ax01, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.05, shrink=0.45)
        cbar.ax.set_title('$COT$', fontsize=24)
        cbar.ax.tick_params(length=0, labelsize=24)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        ax01.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)
        ax01.set_aspect(1.25)

        ##############################################################

        fig.subplots_adjust(wspace=0.1)
        metadata = self.create_metadata()
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight", metadata=metadata)
        plt.close()
        return 1


    def plot_cloud_phase(self, lon_2d, lat_2d, ctp_swir, ctp_ir):

        """
        IR Phase
        0 -- cloud free
        1 -- water cloud
        2 -- ice cloud
        3 -- mixed phase cloud
        6 -- undetermined phase (converted to 4 here)

        SWIR/COP Phase
        0 -- cloud mask undetermined
        1 -- clear sky
        2 -- liquid water cloud
        3 -- ice cloud
        4 -- undetermined phase cloud (but retrieval is attempted as  liquid water)
        """

        proj_plot = ccrs.Orthographic(central_longitude=util.plot_util.ccrs_views[self.mode]['vlon'], central_latitude=util.plot_util.ccrs_views[self.mode]['vlat'])

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        save_dir = os.path.join(self.outdir, "cloud_phase")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        patches_legend_swir = []
        patches_legend_ir   = []
        dt_title     = self.doy2date_str(self.acq_dt) + "Z"
        fname_target = self.format_acq_dt(self.acq_dt)
        sat_fname    = self.satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            if self.verbose:
                print("Message [cloud_phase]: {} skipped since it already exists.".format(full_fname))
            return 0

        im_ctp_swir = ctp_swir
        im_ctp_ir   = ctp_ir

        if self.geojson_fpath is not None:
            im_ctp_swir = self.mask_geojson(lon_2d, lat_2d, im_ctp_swir, proj_plot, util.plot_util.proj_data)
            im_ctp_ir   = self.mask_geojson(lon_2d, lat_2d, im_ctp_ir, proj_plot, util.plot_util.proj_data)

        # im_ctp_swir = np.nan_to_num(im_ctp_swir, 0)
        im_ctp_swir = np.ma.masked_where((np.isnan(im_ctp_swir) | (im_ctp_swir == 0)), im_ctp_swir)
        im_ctp_swir = im_ctp_swir.astype('int8')

        fig = plt.figure(figsize=(40, 40))
        plt.style.use(util.plot_util.mpl_style)
        gs  = GridSpec(1, 2, figure=fig)
        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_ctp_swir,
                            shading='nearest',
                            zorder=2,
                            cmap=util.plot_util.ctp_swir_cmap,
                            transform=util.plot_util.proj_data)
        # ax00.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) Cloud Phase (SWIR/COP) - ".format(self.instrument, self.satellite) + dt_title
        self.add_ancillary(ax00, title=title, scale=1.4)
        ax00.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)

        # create legend labels
        for i in range(len(util.plot_util.ctp_swir_cmap_ticklabels)):
            patches_legend_swir.append(matplotlib.patches.Patch(color=util.plot_util.ctp_swir_cmap_arr[i], label=util.plot_util.ctp_swir_cmap_ticklabels[i]))

        ax00.legend(handles=patches_legend_swir, loc='upper center', bbox_to_anchor=(0.5, -0.05), facecolor='white',
                    ncol=len(patches_legend_swir), fancybox=True, shadow=False, frameon=False, prop={'size': 24})
        ax00.set_aspect(1.25)
        ##############################################################

        ax01 = fig.add_subplot(gs[1], projection=proj_plot)
        im_ctp_ir = self.convert_ir_ctp(im_ctp_ir)
        labels = np.unique(im_ctp_ir).astype('int8') # index; for legend
        ctp_ir_cmap = matplotlib.colors.ListedColormap(util.plot_util.ctp_ir_cmap_arr[labels])

        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_ctp_ir,
                              shading='nearest',
                              zorder=2,
                              cmap=ctp_ir_cmap,
                              transform=util.plot_util.proj_data)

        title = "{} ({}) Cloud Phase (IR) - ".format(self.instrument, self.satellite) + dt_title
        self.add_ancillary(ax01, title=title, scale=1.4)
        ax01.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)
        # create legend labels
        for i in labels:
            patches_legend_ir.append(matplotlib.patches.Patch(color=util.plot_util.ctp_ir_cmap_arr[i], label=util.plot_util.ctp_ir_cmap_ticklabels[i]))

        ax01.legend(handles=patches_legend_ir, loc='upper center', bbox_to_anchor=(0.5, -0.05), facecolor='white',
                    ncol=len(patches_legend_ir), fancybox=True, shadow=False, frameon=False, prop={'size': 24})
        ax01.set_aspect(1.25)
        ##############################################################

        fig.subplots_adjust(wspace=0.1)
        metadata = self.create_metadata()
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight", metadata=metadata)
        plt.close()
        return 1


    def plot_cloud_top(self, lon_2d, lat_2d, cth, ctt):

        proj_plot = ccrs.Orthographic(central_longitude=util.plot_util.ccrs_views[self.mode]['vlon'], central_latitude=util.plot_util.ccrs_views[self.mode]['vlat'])

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        save_dir = os.path.join(self.outdir, "cloud_top_height_temperature")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        patches_legend_cth = []
        patches_legend_ctt = []

        dt_title  = self.doy2date_str(self.acq_dt) + "Z"
        fname_target = self.format_acq_dt(self.acq_dt)
        sat_fname    = self.satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            if self.verbose:
                print("Message [cloud_top]: {} skipped since it already exists.".format(full_fname))
            return 0

        im_cth = cth
        im_ctt = ctt

        if self.geojson_fpath is not None:
            im_cth = self.mask_geojson(lon_2d, lat_2d, im_cth, proj_plot, util.plot_util.proj_data)
            im_ctt = self.mask_geojson(lon_2d, lat_2d, im_ctt, proj_plot, util.plot_util.proj_data)

        fig = plt.figure(figsize=(40, 40))
        plt.style.use(util.plot_util.mpl_style)
        gs  = GridSpec(1, 2, figure=fig)
        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        im_cth_binned = self.bin_cth_to_class(im_cth)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_cth_binned,
                            shading='nearest',
                            zorder=2,
                            cmap=util.plot_util.cth_cmap,
                            transform=util.plot_util.proj_data)

        title = "{} ({}) Cloud Top Height - ".format(self.instrument, self.satellite) + dt_title
        self.add_ancillary(ax00, title=title, scale=1.4)

        for i in range(len(util.plot_util.cth_cmap_ticklabels)):
            patches_legend_cth.append(matplotlib.patches.Patch(color=util.plot_util.cth_cmap_arr[i] , label=util.plot_util.cth_cmap_ticklabels[i]))

        ax00.legend(handles=patches_legend_cth, loc='upper center', bbox_to_anchor=(0.5, -0.05), facecolor='white',
                    ncol=len(patches_legend_cth), fancybox=True, shadow=False, frameon=False, prop={'size': 24})
        ax00.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)
        ax00.set_aspect(1.25)
        ##############################################################

        ctt_bins = np.array([-40, -20, -10, -5, 0, 5, 10, 15])
        im_ctt = self.correct_ctt(im_ctt, im_cth_binned)
        im_ctt_c = im_ctt - 273.15  # convert to Celsius
        vmin_ctt = ctt_bins[0]
        vmax_ctt = ctt_bins[-1] + 5
        im_ctt_c = np.clip(im_ctt_c, vmin_ctt, vmax_ctt)
        # vmin_ctt = round_ctt(np.nanmin(im_ctt_c), 5)
        # vmax_ctt = round_ctt(np.nanmax(im_ctt_c), 5)
        # ctt_bins = np.arange(vmin_ctt, vmax_ctt, 10)
        pxvals = np.arange(0, ctt_bins.size, 1)
        im_ctt_binned, ctt_cmap_ticklabels = self.bin_ctt_to_class(im_ctt_c, ctt_bins, pxvals)
        ctt_tick_locs = (np.arange(len(ctt_cmap_ticklabels)) + 0.5)*(len(ctt_cmap_ticklabels) - 1)/len(ctt_cmap_ticklabels)
        ctt_cmap = plt.get_cmap('RdBu_r', pxvals.size)
        ctt_cmap.set_bad('black', 1)

        ax01 = fig.add_subplot(gs[1], projection=proj_plot)
        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_ctt_binned,
                            shading='nearest',
                            zorder=2,
                            cmap=ctt_cmap,
                            transform=util.plot_util.proj_data)

        title = "{} ({}) Cloud Top Temperature - ".format(self.instrument, self.satellite) + dt_title
        self.add_ancillary(ax01, title=title, scale=1.4)
        for i in range(len(ctt_cmap_ticklabels)):
            patches_legend_ctt.append(matplotlib.patches.Patch(color=util.plot_util.ctt_cmap_arr[i] , label=ctt_cmap_ticklabels[i]))

        ax01.legend(handles=patches_legend_ctt, loc='upper center', bbox_to_anchor=(0.5, -0.025), facecolor='white',
            ncol=int(len(patches_legend_ctt)/2), fancybox=True, shadow=False, frameon=False, prop={'size': 24},
            title='Cloud Top Temperature ($^\circ C$ )', title_fontsize=18)
        ax01.set_extent(util.plot_util.ccrs_views[self.mode]['view_extent'], util.plot_util.proj_data)
        ax01.set_aspect(1.25)
        ##############################################################

        fig.subplots_adjust(wspace=0.1)
        metadata = self.create_metadata()
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight", metadata=metadata)
        plt.close()
        return 1
    ############################################################################################################################

################################################################################################################
