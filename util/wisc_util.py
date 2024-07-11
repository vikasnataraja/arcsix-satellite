"""
Versions and collections of Wisconsin products

Product                      Satellite            Sensor               Version
CLDPROP_L2_VIIRS_NOAA20_NRT  noaa20               viirs                1.1.4dev1
CLDPROP_L2_VIIRS_SNPP_NRT    snpp                 viirs                1.1.3dev9
VJ102MOD-nrt                 noaa20               viirs                3.1.0
VJ103MOD-nrt                 noaa20               viirs                3.1.0
VNP02MOD-nrt                 snpp                 viirs                3.1.0
VNP03MOD-nrt                 snpp                 viirs                3.1.0

Last updated: July 10, 2024
"""
import os
import sys
import json
import datetime
import numpy as np

_now_dt    = datetime.datetime.now(datetime.timezone.utc)
_now_dt    = _now_dt.replace(tzinfo=None) # so that timedelta does not raise an error

wisc_info = {
    'CLDPROP_L2_VIIRS_NOAA20': {
                                'api_fname' : 'CLDPROP_L2_VIIRS_NOAA20',
                                'cli_fname' : 'CLDPROP_L2_VIIRS_NOAA20_NRT',
                                'version'   : '1.1.4dev1',
                                'collection': '011',
                                },

    'CLDPROP_L2_VIIRS_SNPP':   {
                                'api_fname' : 'CLDPROP_L2_VIIRS_SNP',
                                'cli_fname' : 'CLDPROP_L2_VIIRS_SNP_NRT',
                                'version'   : '1.1.3dev9',
                                'collection': '011',
                                },

    'VJ102MOD':                {
                                'api_fname' : 'VJ102MOD',
                                'cli_fname' : 'VJ102MOD-nrt',
                                'version'   : '3.1.0',
                                'collection': '021',
                                },

    'VJ103MOD':                {
                                'api_fname' : 'VJ103MOD',
                                'cli_fname' : 'VJ103MOD-nrt',
                                'version'   : '3.1.0',
                                'collection': '021',
                                },

    'VNP02MOD':                {
                                'api_fname' : 'VNP02MOD',
                                'cli_fname' : 'VNP02MOD-nrt',
                                'version'   : '3.1.0',
                                'collection': '002',
                                },

    'VNP03MOD':                {
                                'api_fname' : 'VNP03MOD',
                                'cli_fname' : 'VNP03MOD-nrt',
                                'version'   : '3.1.0',
                                'collection': '002',
                                },


}

def get_pacq_dts(fdir):
    """
    Get product acquisition datetimes of files in `fdir`
    Example: ['VNP02MOD.A2024185.2210', 'MOD03.A2024185.2215']
    """
    pacq_dts = []
    for file in os.listdir(fdir):
        if file.endswith(('hdf', 'nc')): # support only modis and viirs files
            split_fparts = os.path.basename(file).split('.')
            pacq_dts.append(split_fparts[0] + '.' + split_fparts[1] + '.' + split_fparts[2])

    return pacq_dts


def parse_geojson(geojson_fpath):
    with open(geojson_fpath, 'r') as f:
        data = json.load(f)

    coords = data['features'][0]['geometry']['coordinates']

    lons = np.array(coords[0])[:, 0]
    lats = np.array(coords[0])[:, 1]
    return lons, lats


def region_parser(extent):

    if len(extent) != 4:
        print('Error [wisc_util.region_parser]: `extent` must be in [lon1 lon2 lat1 lat2] format')
        sys.exit()

# check to make sure extent is correct
    if (extent[0] >= extent[1]) or (extent[2] >= extent[3]):
        msg = 'Error [wisc_util.region_parser]: The given extents of lon/lat are incorrect: %s.\nPlease check to make sure extent is passed as `lon1 lon2 lat1 lat2` format i.e. West, East, South, North.' % extent
        print(msg)
        sys.exit()

    llons = np.linspace(extent[0], extent[1], 200)
    llats = np.linspace(extent[2], extent[3], 200)
    return llons, llats

def time_handler(start_time, end_time, latest_nhours):

    if (start_time is not None) and (end_time is not None) and (latest_nhours is None):

        # start at 0 UTC unless specified by user
        start_hr = 0
        start_min = 0
        if len(start_time) == 12:
            start_hr, start_min = int(start_time[8:10]), int(start_time[10:12])

        # look for data until the last minute of the day
        end_hr = 23
        end_min = 59
        if (len(end_time) == 12):
            if (int(end_time[8:10]) < end_hr): # update only if different
                end_hr = int(end_time[8:10])

            if (int(end_time[10:12]) < end_min):
                end_min = int(end_time[10:12])


        start_dt  = datetime.datetime(int(start_time[:4]), int(start_time[4:6]), int(start_time[6:8]), start_hr, start_min)
        end_dt    = datetime.datetime(int(end_time[:4]), int(end_time[4:6]), int(end_time[6:8]), end_hr, end_min)

    else:
        end_dt = _now_dt
        if latest_nhours is not None:
            start_dt = end_dt - datetime.timedelta(hours=latest_nhours)
        else:
            print("Warning [wisc_util.region_parser]: Did not receive `latest_nhours`. Automatically set to 3 hours.")
            start_dt = end_dt - datetime.timedelta(hours=3)

    return start_dt, end_dt
