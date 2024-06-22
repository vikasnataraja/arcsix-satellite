import os
import sys
import h5py
import datetime
import numpy as np

from scipy import interpolate
import math

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

EARTH_RADIUS = 6371.009

MODIS_L1B_QKM_BANDS = {
                        1: 650,
                        2: 860,
                      }


MODIS_L1B_HKM_1KM_BANDS = {
                        1: 650,
                        2: 860,
                        3: 470,
                        4: 555,
                        5: 1240,
                        6: 1640,
                        7: 2130,
                        26: 1375
                      }


VIIRS_ALL_BANDS = {
                    'I01': 640,
                    'I02': 865,
                    'I03': 1610,
                    'I04': 3740,
                    'I05': 11450,
                    'M01': 415,
                    'M02': 445,
                    'M03': 490,
                    'M04': 555,
                    'M05': 673,
                    'M06': 746,
                    'M07': 865,
                    'M08': 1240,
                    'M09': 1378,
                    'M10': 1610,
                    'M11': 2250,
                    'M12': 3700,
                    'M13': 4050,
                    'M14': 8550,
                    'M15': 12013
                    }


VIIRS_L1B_MOD_BANDS = {
                    'M01': 415,
                    'M02': 445,
                    'M03': 490,
                    'M04': 555,
                    'M05': 673,
                    'M06': 746,
                    'M07': 865,
                    'M08': 1240,
                    'M10': 1610,
                    'M11': 2250
                    }

VIIRS_L1B_IMG_BANDS = {
                    'I01': 640,
                    'I02': 865,
                    'I03': 1610
                    }




def grid_by_extent(lon, lat, data, extent=None, NxNy=None, method='nearest'):

    """
    Grid irregular data into a regular grid by input 'extent' (westmost, eastmost, southmost, northmost)

    Input:
        lon: numpy array, input longitude to be gridded
        lat: numpy array, input latitude to be gridded
        data: numpy array, input data to be gridded
        extent=: Python list, [westmost, eastmost, southmost, northmost]
        NxNy=: Python list, [Nx, Ny], lon_2d = np.linspace(westmost, eastmost, Nx)
                                      lat_2d = np.linspace(southmost, northmost, Ny)

    Output:
        lon_2d : numpy array, gridded longitude
        lat_2d : numpy array, gridded latitude
        data_2d: numpy array, gridded data

    How to use:
        After read in the longitude latitude and data into lon0, lat0, data0
        lon, lat data = grid_by_extent(lon0, lat0, data0, extent=[10, 15, 10, 20])
    """

    if extent is None:
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    if NxNy is None:
        xy = (extent[1]-extent[0])*(extent[3]-extent[2])
        N0 = np.sqrt(lon.size/xy)

        Nx = int(N0*(extent[1]-extent[0]))
        if Nx%2 == 1:
            Nx += 1

        Ny = int(N0*(extent[3]-extent[2]))
        if Ny%2 == 1:
            Ny += 1
    else:
        Nx, Ny = NxNy

    lon_1d0 = np.linspace(extent[0], extent[1], Nx+1)
    lat_1d0 = np.linspace(extent[2], extent[3], Ny+1)

    lon_1d = (lon_1d0[1:]+lon_1d0[:-1])/2.0
    lat_1d = (lat_1d0[1:]+lat_1d0[:-1])/2.0

    lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d)

    points   = np.transpose(np.vstack((lon, lat)))
    data_2d0 = interpolate.griddata(points, data, (lon_2d, lat_2d), method='linear', fill_value=np.nan)

    if method == 'nearest':
        data_2d  = interpolate.griddata(points, data, (lon_2d, lat_2d), method='nearest')
        logic = np.isnan(data_2d0) | np.isnan(data_2d)
        data_2d[logic] = 0.0
        return lon_2d, lat_2d, data_2d
    else:
        logic = np.isnan(data_2d0)
        data_2d0[logic] = 0.0
        return lon_2d, lat_2d, data_2d0



def aggregate(data, dx=10, dy=10):
    """Aggregates higher resolution data to lower resolution data"""

    nx, ny = data.shape

    # calculate the size of the aggregated array
    if nx % dx > 0:
        NX = nx//dx + 1
    else:
        NX = nx//dx

    if ny % dy > 0:
        NY = ny//dy + 1
    else:
        NY = ny//dy

    data_agg = np.zeros((NX, NY), dtype='float64')

    # Loop through and average over finer resolution to result in coarser resolution
    for i in range(NX):

        start_x = i * dx
        end_x   = min((i + 1) * dx, nx)

        for j in range(NY):

            start_y = j * dy
            end_y   = min((j + 1) * dy, ny)

            data_agg[i, j] = np.nanmean(data[start_x:end_x, start_y:end_y])

    return data_agg



def grid_by_lonlat(lon, lat, data, lon_1d=None, lat_1d=None, method='nearest', fill_value=np.nan, Ngrid_limit=1, return_geo=True):

    """
    Grid irregular data into a regular grid by input longitude and latitude
    Input:
        lon: numpy array, input longitude to be gridded
        lat: numpy array, input latitude to be gridded
        data: numpy array, input data to be gridded
        lon_1d=: numpy array, the longitude of the grids
        lat_1d=: numpy array, the latitude of the grids
    Output:
        lon_2d : numpy array, gridded longitude
        lat_2d : numpy array, gridded latitude
        data_2d: numpy array, gridded data
    How to use:
        After read in the longitude latitude and data into lon0, lat0, data0
        lon, lat, data = grid_by_lonlat(lon0, lat0, data0, lon_1d=np.linspace(10.0, 15.0, 100), lat_1d=np.linspace(10.0, 20.0, 100))
    """

    # flatten lon/lat/data
    #/----------------------------------------------------------------------------\#
    lon = np.array(lon).ravel()
    lat = np.array(lat).ravel()
    data = np.array(data).ravel()
    #\----------------------------------------------------------------------------/#

    if lon_1d is None or lat_1d is None:

        extent = [lon.min(), lon.max(), lat.min(), lat.max()]

        xy = (extent[1]-extent[0])*(extent[3]-extent[2])
        N0 = np.sqrt(lon.size/xy)

        Nx = int(N0*(extent[1]-extent[0]))
        if Nx%2 == 1:
            Nx += 1

        Ny = int(N0*(extent[3]-extent[2]))
        if Ny%2 == 1:
            Ny += 1

        lon_1d0 = np.linspace(extent[0], extent[1], Nx+1)
        lat_1d0 = np.linspace(extent[2], extent[3], Ny+1)

        lon_1d = (lon_1d0[1:]+lon_1d0[:-1])/2.0
        lat_1d = (lat_1d0[1:]+lat_1d0[:-1])/2.0

    lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d)

    points   = np.transpose(np.vstack((lon, lat)))

    if method == 'nearest':
        data_2d = find_nearest(lon, lat, data, lon_2d, lat_2d, fill_value=np.nan, Ngrid_limit=Ngrid_limit)
    else:
        data_2d = interpolate.griddata(points, data, (lon_2d, lat_2d), method=method, fill_value=np.nan)

    logic = np.isnan(data_2d)
    data_2d[logic] = fill_value

    if return_geo:
        return lon_2d, lat_2d, data_2d
    else:
        return data_2d




def haversine(lon1, lon2, lat1, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = EARTH_RADIUS * 1e3 # Radius of earth in meters
    return c * r


# def calc_lonlat_arr(extent, width, height, resolution):
#     west, east, south, north = extent
#     lon2_arr = np.zeros((width))
#     lat2_arr = np.zeros((height))
#     lon1, lat1 = west, south
#     for i in range(width):
#         lon2, lat2 = get_lon2lat2(lon1, lat1, bearing=90, distance=resolution)
#         lon2_arr[i] = lon2
#         lon1, lat1 = lon2, lat2

#     lon1, lat1 = west, south
#     for j in range(height):
#         lon2, lat2 = get_lon2lat2(lon1, lat1, bearing=0, distance=resolution)
#         lat2_arr[j] = lat2
#         lon1, lat1 = lon2, lat2

#     return lon2_arr, lat2_arr

def calc_lonlat_arr(west, south, width, height, resolution):
    # west, east, south, north = extent
    lon2_arr = np.zeros((width))
    lat2_arr = np.zeros((height))
    lon2_arr[0], lat2_arr[0] = west, south # initialize first index with starting coords

    bearing = 90. # initial bearing to go east along same lat (approximate)
    for i in range(width - 1):
        lon2_arr[i+1], _ = get_lon2lat2(lon2_arr[i], lat2_arr[0], bearing=bearing, distance=resolution)
        bearing = calc_bearing(lat2_arr[0], lat2_arr[0], lon2_arr[i+1] - lon2_arr[i], reference='NS')

    bearing = 0. # initial bearing to go north along same lon (approximate)
    for j in range(height - 1):
        _, lat2_arr[j+1] = get_lon2lat2(lon2_arr[0], lat2_arr[j], bearing=bearing, distance=resolution)
        bearing = calc_bearing(lat2_arr[j], lat2_arr[j+1], 0., reference='NS')

    return lon2_arr, lat2_arr


def calc_bearing(lat1, lat2, delta_lon, reference="north-south"):
    """
    Args:
        - lat1: float, latitude of the first point in degrees
        - lat2: float, latitude of the second point in degrees
        - delta_lon: float, difference of the longitudes in degrees
                     (positive=north, east; negative=south, west)
        - reference: str, reference plane/axis. For meteorological coords,
                     use reference="north-south" (also accepts "N-S" or "NS").
                     For this case, +bearing = northward/eastward,
                                    -bearing = westward/southward
                     otherwise, reference axis will be west-east.
                     For this case, -180 to 180 = west to east

    Returns:
        - bearing: float, bearing angle in degrees with respect to reference

    Reference: http://www.movable-type.co.uk/scripts/latlong.html
    """
    lat1, lat2, delta_lon = map(math.radians, [lat1, lat2, delta_lon]) # convert to radians
    x = math.sin(delta_lon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))
    bearing = math.atan2(y, x)
    if reference.lower() == "north-south" or reference.upper() == "N-S" or reference.upper() == "NS":
        return 90 - bearing * 180./math.pi

    return bearing * 180./math.pi # normalize to 0-360 by ((bearing * 180./math.pi) + 360.) % 360.



def get_lon2lat2(lon1, lat1, bearing, distance):
    """
    Get lat/lon of the destination point given the initial point,
    bearing, and the distance between the points.
    Args:
        - lon1: float, longitude of the first point in degrees
        - lat1: float, latitude of the first point in degrees
        - bearing: float, angle of trajectory.
                   For instance, 0 is true north, 90 is east, 180 is south, 270 is west.
        - distance: float, distance between the points in meters
    Returns:
        - lon2: float, longitude of the second point in degrees
        - lat2: float, latitude of the second point in degrees

    Reference: http://www.movable-type.co.uk/scripts/latlong.html
    """
    lon1, lat1, bearing = map(math.radians, [lon1, lat1, bearing]) # convert to radians
    delta = distance/(EARTH_RADIUS * 1e3) # delta = d/R
    lat2 = math.asin(math.sin(lat1) * math.cos(delta) + math.cos(lat1) * math.sin(delta) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(delta) * math.cos(lat1),
                            math.cos(delta) - math.sin(lat1) * math.sin(lat2))
    return lon2 * 180./math.pi, lat2 * 180./math.pi


def get_extent(fdir):
    with open(os.path.join(fdir, "metadata.txt"), "r") as f:
        meta = f.readlines()

    extent_str = meta[1][8:-2]
    extent     = [float(idx) for idx in extent_str.split(', ')]
    return extent


def get_doy_tag(date, day_interval=8):

    """
    Get day of year tag, e.g., 078, for a given day

    Input:
        date: datetime/date object, e.g., datetime.datetime(2000, 1, 1)

    Output:
        doy_tag: string, closest day of the year, e.g. '097'

    """

    doy = date.timetuple().tm_yday

    day_total = datetime.datetime(date.year, 12, 31).timetuple().tm_yday

    doys = np.arange(1, day_total+1, day_interval)

    doy_tag = '%3.3d' % doys[np.argmin(np.abs(doys-doy))]

    return doy_tag


def check_equal(a, b, threshold=1.0e-6):

    """
    Check if two values are equal (or close to each other)

    Input:
        a: integer or float, value of a
        b: integer or float, value of b

    Output:
        boolean, true or false
    """

    if abs(a-b) >= threshold:
        return False
    else:
        return True


def upscale_modis_lonlat(lon_in, lat_in, scale=5, extra_grid=True):

    """
    Upscaling MODIS geolocation from 5km/1km/1km to 1km/250m/500nm.

    Details can be found at
    http://www.icare.univ-lille1.fr/tutorials/MODIS_geolocation

    Input:
        lon_in: numpy array, input longitude
        lat_in: numpy array, input latitude
        scale=: integer, upscaling factor, e.g., 5km to 1km (scale=5), 1km to 250m (scale=4), 1km to 500nm (scale=2)
        extra_grid=: boolen, for MOD/MYD 05, 06 data, extra_grid=True, for other dataset, extra_grid=False

    Output:
        lon_out: numpy array, upscaled longitude
        lat_out: numpy array, upscaled latitude

    How to use:
        # After read in the longitude latitude from MODIS L2 06 data, e.g., lon0, lat0
        lon, lat = upscale_modis_lonlat(lon0, lat0, scale=5, extra_grid=True)
    """

    try:
        import cartopy.crs as ccrs
    except ImportError:
        msg = 'Error   [upscale_modis_lonlat]: To use \'upscale_modis_lonlat\', \'cartopy\' needs to be installed.'
        raise ImportError(msg)

    offsets_dict = {
            4: {'along_scan':0, 'along_track':1.5},
            5: {'along_scan':2, 'along_track':2},
            2: {'along_scan':0, 'along_track':0.5}
            }
    offsets = offsets_dict[scale]

    lon_in[lon_in>180.0] -= 360.0

    # +
    # find center lon, lat
    proj_lonlat = ccrs.PlateCarree()

    lon0 = np.array([lon_in[0, 0], lon_in[-1, 0], lon_in[-1, -1], lon_in[0, -1], lon_in[0, 0]])
    lat0 = np.array([lat_in[0, 0], lat_in[-1, 0], lat_in[-1, -1], lat_in[0, -1], lat_in[0, 0]])

    if (abs(lon0[0]-lon0[1])>180.0) | (abs(lon0[0]-lon0[2])>180.0) | \
       (abs(lon0[0]-lon0[3])>180.0) | (abs(lon0[1]-lon0[2])>180.0) | \
       (abs(lon0[1]-lon0[3])>180.0) | (abs(lon0[2]-lon0[3])>180.0):

        lon0[lon0<0.0] += 360.0

    center_lon_tmp = lon0[:-1].mean()
    center_lat_tmp = lat0[:-1].mean()

    proj_tmp  = ccrs.Orthographic(central_longitude=center_lon_tmp, central_latitude=center_lat_tmp)
    xy_tmp    = proj_tmp.transform_points(proj_lonlat, lon0, lat0)[:, [0, 1]]
    center_x  = xy_tmp[:, 0].mean()
    center_y  = xy_tmp[:, 1].mean()
    center_lon, center_lat = proj_lonlat.transform_point(center_x, center_y, proj_tmp)
    # -

    proj_xy  = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    xy_in = proj_xy.transform_points(proj_lonlat, lon_in, lat_in)[:, :, [0, 1]]

    y_in, x_in = lon_in.shape
    XX_in = offsets['along_scan']  + np.arange(x_in) * scale   # along scan
    YY_in = offsets['along_track'] + np.arange(y_in) * scale   # along track

    x  = x_in * scale
    y  = y_in * scale

    if scale==5 and extra_grid:
        XX = np.arange(x+4)
    else:
        XX = np.arange(x)

    YY = np.arange(y)

    f_x = interpolate.interp2d(XX_in, YY_in, xy_in[:, :, 0], kind='linear', fill_value=None)
    f_y = interpolate.interp2d(XX_in, YY_in, xy_in[:, :, 1], kind='linear', fill_value=None)

    lonlat = proj_lonlat.transform_points(proj_xy, f_x(XX, YY), f_y(XX, YY))[:, :, [0, 1]]

    return lonlat[:, :, 0], lonlat[:, :, 1]


def get_data_h4(hdf_dset, replace_fill_value=np.nan):

    attrs = hdf_dset.attributes()
    data  = hdf_dset[:]

    if 'add_offset' in attrs:
        data = data - attrs['add_offset']

    if 'scale_factor' in attrs:
        data = data * attrs['scale_factor']

    if '_FillValue' in attrs and replace_fill_value is not None:
        _FillValue = np.float64(attrs['_FillValue'])
        data[data == _FillValue] = replace_fill_value

    return data


def get_data_nc(nc_dset, replace_fill_value=np.nan):

    nc_dset.set_auto_maskandscale(True)
    data  = nc_dset[:]

    if replace_fill_value is not None:
        data = data.astype('float64')
        data.filled(fill_value=replace_fill_value)

    return data





# reader for MODIS (Moderate Resolution Imaging Spectroradiometer)
#/-----------------------------------------------------------------------------\

class modis_l1b:

    """
    Read MODIS Level 1B file into an object `modis_l1b`

    Input:
        fnames=     : keyword argument, default=None, Python list of the file path of the original HDF4 files
        extent=     : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        verbose=    : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['lon']
                ['lat']
                ['wvl']
                ['rad']
                ['ref']
                ['cnt']
                ['uct']
    """


    ID = 'MODIS Level 1b Calibrated Radiance'


    def __init__(self, \
                 fnames    = None, \
                 f03       = None, \
                 extent    = None, \
                 bands     = None, \
                 verbose   = False):

        self.fnames     = fnames      # Python list of the file path of the original HDF4 files
        self.f03        = f03         # geolocation class object created using the `modis_03` reader
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.bands      = bands       # Python list of bands that need to be extracted
        self.verbose    = verbose     # verbose tag


        filename = os.path.basename(fnames[0]).lower()
        if 'qkm' in filename:
            self.resolution = 0.25
            if bands is None:
                self.bands = list(MODIS_L1B_QKM_BANDS.keys())

            elif (bands is not None) and not (set(bands).issubset(set(MODIS_L1B_QKM_BANDS.keys()))):

                msg = 'Error [modis_l1b]: Bands must be one or more of %s' % list(MODIS_L1B_QKM_BANDS.keys())
                raise KeyError(msg)

        elif 'hkm' in filename:
            self.resolution = 0.5
            if bands is None:
                self.bands = list(MODIS_L1B_HKM_1KM_BANDS.keys())

            elif (bands is not None) and not (set(bands).issubset(set(MODIS_L1B_HKM_1KM_BANDS.keys()))):
                msg = 'Error [modis_l1b]: Bands must be one or more of %s' % list(MODIS_L1B_HKM_1KM_BANDS.keys())
                raise KeyError(msg)

        elif '1km' in filename:
            self.resolution = 1.0
            if bands is None:
                self.bands = list(MODIS_L1B_HKM_1KM_BANDS.keys())

            elif (bands is not None) and not (set(bands).issubset(set(MODIS_L1B_HKM_1KM_BANDS.keys()))):
                msg = 'Error [modis_l1b]: Bands must be one or more of %s' % list(MODIS_L1B_HKM_1KM_BANDS.keys())
                raise KeyError(msg)

        else:
            sys.exit('Error [modis_l1b]: Currently, only QKM (0.25km), HKM (0.5km), and 1KM products are supported.')

        for fname in self.fnames:
            self.read(fname)


    def _get_250_500_attrs(self, hdf_dset_250, hdf_dset_500):
        rad_off = hdf_dset_250.attributes()['radiance_offsets']         + hdf_dset_500.attributes()['radiance_offsets']
        rad_sca = hdf_dset_250.attributes()['radiance_scales']          + hdf_dset_500.attributes()['radiance_scales']
        ref_off = hdf_dset_250.attributes()['reflectance_offsets']      + hdf_dset_500.attributes()['reflectance_offsets']
        ref_sca = hdf_dset_250.attributes()['reflectance_scales']       + hdf_dset_500.attributes()['reflectance_scales']
        cnt_off = hdf_dset_250.attributes()['corrected_counts_offsets'] + hdf_dset_500.attributes()['corrected_counts_offsets']
        cnt_sca = hdf_dset_250.attributes()['corrected_counts_scales']  + hdf_dset_500.attributes()['corrected_counts_scales']
        return rad_off, rad_sca, ref_off, ref_sca, cnt_off, cnt_sca


    def _get_250_500_uct(self, hdf_uct_250, hdf_uct_500):
        uct_spc = hdf_uct_250.attributes()['specified_uncertainty'] + hdf_uct_500.attributes()['specified_uncertainty']
        uct_sca = hdf_uct_250.attributes()['scaling_factor']        + hdf_uct_500.attributes()['scaling_factor']
        return uct_spc, uct_sca


    def read(self, fname):

        """
        Read radiance/reflectance/corrected counts along with their uncertainties from the MODIS L1B data
        self.data
                ['lon']
                ['lat']
                ['wvl']
                ['rad']
                ['ref']
                ['cnt']
                ['uct']
        """

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_l1b]: To use \'modis_l1b\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        f     = SD(fname, SDC.READ)

        # when resolution equals to 250 m
        if check_equal(self.resolution, 0.25):
            if self.f03 is not None:
                lon0  = self.f03.data['lon']['data']
                lat0  = self.f03.data['lat']['data']
            else:
                lat0  = f.select('Latitude')
                lon0  = f.select('Longitude')

            lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=4, extra_grid=False)
            raw0      = f.select('EV_250_RefSB')
            uct0      = f.select('EV_250_RefSB_Uncert_Indexes')
            # wvl       = np.array([650, 860], dtype='uint16') # QKM bands

            # if self.extent is None:
            #     if 'actual_range' in lon0.attributes().keys():
            #         lon_range = lon0.attributes()['actual_range']
            #         lat_range = lat0.attributes()['actual_range']
            #     elif 'valid_range' in lon0.attributes().keys():
            #         lon_range = lon0.attributes()['valid_range']
            #         lat_range = lat0.attributes()['valid_range']
            #     else:
            #         lon_range = [-180.0, 180.0]
            #         lat_range = [-90.0 , 90.0]

            if self.extent is None:
                lon_range = [-180.0, 180.0]
                lat_range = [-90.0 , 90.0]

            else:
                lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
                lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

            logic     = (lon >= lon_range[0]) & (lon <= lon_range[1]) & (lat >= lat_range[0]) & (lat <= lat_range[1])
            lon       = lon[logic]
            lat       = lat[logic]

            # save offsets and scaling factors
            rad_off = raw0.attributes()['radiance_offsets']
            rad_sca = raw0.attributes()['radiance_scales']

            ref_off = raw0.attributes()['reflectance_offsets']
            ref_sca = raw0.attributes()['reflectance_scales']

            cnt_off = raw0.attributes()['corrected_counts_offsets']
            cnt_sca = raw0.attributes()['corrected_counts_scales']

            uct_spc = uct0.attributes()['specified_uncertainty']
            uct_sca = uct0.attributes()['scaling_factor']
            do_region = True

        # when resolution equals to 500 m
        elif check_equal(self.resolution, 0.5):
            if self.f03 is not None:
                lon0  = self.f03.data['lon']['data']
                lat0  = self.f03.data['lat']['data']
            else:
                lat0  = f.select('Latitude')
                lon0  = f.select('Longitude')


            lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=2, extra_grid=False)
            raw0_250  = f.select('EV_250_Aggr500_RefSB')
            uct0_250  = f.select('EV_250_Aggr500_RefSB_Uncert_Indexes')
            raw0_500  = f.select('EV_500_RefSB')
            uct0_500  = f.select('EV_500_RefSB_Uncert_Indexes')

            # save offsets and scaling factors (from both QKM and HKM bands)
            rad_off, rad_sca, ref_off, ref_sca, cnt_off, cnt_sca = self._get_250_500_attrs(raw0_250, raw0_500)
            uct_spc, uct_sca                                     = self._get_250_500_uct(uct0_250, uct0_500)

            # combine QKM and HKM bands
            raw0      = np.vstack([raw0_250, raw0_500])
            uct0      = np.vstack([uct0_250, uct0_500])

            # wvl       = np.array([470, 555, 1240, 1640, 2130], dtype='uint16') # HKM bands
            do_region = True

            # if self.extent is None:
            #     if 'actual_range' in lon0.attributes().keys():
            #         lon_range = lon0.attributes()['actual_range']
            #         lat_range = lat0.attributes()['actual_range']
            #     elif 'valid_range' in lon0.attributes().keys():
            #         lon_range = lon0.attributes()['valid_range']
            #         lat_range = lat0.attributes()['valid_range']
            #     else:
            #         lon_range = [-180.0, 180.0]
            #         lat_range = [-90.0 , 90.0]

            if self.extent is None:
                lon_range = [-180.0, 180.0]
                lat_range = [-90.0 , 90.0]

            else:
                lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
                lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

            logic     = (lon >= lon_range[0]) & (lon <= lon_range[1]) & (lat >= lat_range[0]) & (lat <= lat_range[1])
            lon       = lon[logic]
            lat       = lat[logic]

        # when resolution equals to 1000 m
        elif check_equal(self.resolution, 1.0):
            if self.f03 is not None:
                raw0_250  = f.select('EV_250_Aggr1km_RefSB')
                uct0_250  = f.select('EV_250_Aggr1km_RefSB_Uncert_Indexes')
                raw0_500  = f.select('EV_500_Aggr1km_RefSB')
                uct0_500  = f.select('EV_500_Aggr1km_RefSB_Uncert_Indexes')

                # save offsets and scaling factors (from both QKM and HKM bands)
                rad_off, rad_sca, ref_off, ref_sca, cnt_off, cnt_sca = self._get_250_500_attrs(raw0_250, raw0_500)
                uct_spc, uct_sca                                     = self._get_250_500_uct(uct0_250, uct0_500)

                # combine QKM and HKM bands
                raw0      = np.vstack([raw0_250, raw0_500])
                uct0      = np.vstack([uct0_250, uct0_500])
                # wvl       = np.array([650, 860, 470, 555, 1240, 1640, 2130], dtype='uint16')
                do_region = False
                lon       = self.f03.data['lon']['data']
                lat       = self.f03.data['lat']['data']
                logic     = self.f03.logic[find_fname_match(fname, self.f03.logic.keys())]['1km']
            else:
                sys.exit('Error   [modis_l1b]: 1KM product reader has not been implemented without geolocation file being specified.')

        else:
            sys.exit('Error   [modis_l1b]: \'resolution=%f\' has not been implemented.' % self.resolution)


        # Calculate 1. radiance, 2. reflectance, 3. corrected counts from the raw data
        #/----------------------------------------------------------------------------\#
        raw = raw0[:][:, logic]
        rad = np.zeros((len(self.bands), raw.shape[1]), dtype=np.float64)
        ref = np.zeros((len(self.bands), raw.shape[1]), dtype=np.float64)
        cnt = np.zeros((len(self.bands), raw.shape[1]), dtype=np.float64)

        # Calculate uncertainty
        uct     = uct0[:][:, logic]
        uct_pct = np.zeros((len(self.bands), raw.shape[1]), dtype=np.float64)

        wvl = np.zeros(len(self.bands), dtype='uint16')

        band_counter = 0
        for i in self.bands:
            band_idx                     = i - 1 # band indexing in Python starts from 0
            rad0                         = (raw[band_idx, ...] - rad_off[band_idx]) * rad_sca[band_idx]
            rad[band_counter, ...]       = rad0/1000.0 # convert to W/m^2/nm/sr
            ref[band_counter, ...]       = (raw[band_idx, ...] - ref_off[band_idx]) * ref_sca[band_idx]
            cnt[band_counter, ...]       = (raw[band_idx, ...] - cnt_off[band_idx]) * cnt_sca[band_idx]
            uct_pct[band_counter, ...]   = uct_spc[band_idx] * np.exp(uct[band_idx] / uct_sca[band_idx]) # convert to percentage
            wvl[band_counter]            = MODIS_L1B_HKM_1KM_BANDS[self.bands[band_counter]]
            band_counter                += 1

        f.end()
        # -------------------------------------------------------------------------------------------------



        if hasattr(self, 'data'):
            if do_region:
                self.data['lon'] = dict(name='Longitude'           , data=np.hstack((self.data['lon']['data'], lon)),     units='degrees')
                self.data['lat'] = dict(name='Latitude'            , data=np.hstack((self.data['lat']['data'], lat)),     units='degrees')

            self.data['rad'] = dict(name='Radiance'                , data=np.hstack((self.data['rad']['data'], rad)),     units='W/m^2/nm/sr')
            self.data['ref'] = dict(name='Reflectance (x cos(SZA))', data=np.hstack((self.data['ref']['data'], ref)),     units='N/A')
            self.data['cnt'] = dict(name='Corrected Counts'        , data=np.hstack((self.data['cnt']['data'], cnt)),     units='N/A')
            self.data['uct'] = dict(name='Uncertainty Percentage'  , data=np.hstack((self.data['uct']['data'], uct_pct)), units='N/A')

        else:

            self.data = {}
            self.data['lon'] = dict(name='Longitude'               , data=lon,     units='degrees')
            self.data['lat'] = dict(name='Latitude'                , data=lat,     units='degrees')
            self.data['wvl'] = dict(name='Wavelength'              , data=wvl,     units='nm')
            self.data['rad'] = dict(name='Radiance'                , data=rad,     units='W/m^2/nm/sr')
            self.data['ref'] = dict(name='Reflectance (x cos(SZA))', data=ref,     units='N/A')
            self.data['cnt'] = dict(name='Corrected Counts'        , data=cnt,     units='N/A')
            self.data['uct'] = dict(name='Uncertainty Percentage'  , data=uct_pct, units='N/A')


    def save_h5(self, fname):

        f = h5py.File(fname, 'w')
        for key in self.data.keys():
            f[key] = self.data[key]['data']
        f.close()



class modis_l2:

    """
    Read MODIS level 2 cloud product

    Input:
        fnames=   : keyword argument, default=None, Python list of the file path of the original HDF4 file
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['lon']
                ['lat']
                ['cot']
                ['cer']
    """


    ID = 'MODIS Level 2 Cloud Product'


    def __init__(self, \
                 fnames    = None,  \
                 f03       = None,  \
                 extent    = None,  \
                 verbose   = False):

        self.fnames     = fnames      # file name of the pickle file
        self.f03        = f03
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag

        for fname in self.fnames:

            self.read(fname)


    def read(self, fname):

        """
        Read cloud optical properties

        self.data
            ['lon']
            ['lat']
            ['cot']
            ['cer']
            ['pcl']
            ['lon_5km']
            ['lat_5km']

        self.logic
        self.logic_5km
        """

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_l2]: To use \'modis_l2\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        f          = SD(fname, SDC.READ)

        # lon lat
        lat0       = f.select('Latitude')
        lon0       = f.select('Longitude')

        # 5km retrievals
        cth        = f.select('Cloud_Top_Height')
        ctt        = f.select('Cloud_Top_Temperature')

        # 1km retrievals
        cot0       = f.select('Cloud_Optical_Thickness')
        cer0       = f.select('Cloud_Effective_Radius')
        cwp0       = f.select('Cloud_Water_Path')

        # 1km retrievals for ice/snow/bright surfaces
        cot_1621   = f.select('Cloud_Optical_Thickness_1621')
        cer_1621   = f.select('Cloud_Effective_Radius_1621')
        cwp_1621   = f.select('Cloud_Water_Path_1621')

        # PCL retrievals
        cot_pcl     = f.select('Cloud_Optical_Thickness_PCL')
        cer_pcl     = f.select('Cloud_Effective_Radius_PCL')
        cwp_pcl     = f.select('Cloud_Water_Path_PCL')

        # 1621 PCL retrievals
        cot_1621_pcl = f.select('Cloud_Optical_Thickness_1621_PCL')
        cer_1621_pcl = f.select('Cloud_Effective_Radius_1621_PCL')
        cwp_1621_pcl = f.select('Cloud_Water_Path_1621_PCL')

        # other important variables
        ctp         = f.select('Cloud_Phase_Optical_Properties')
#         cld_layer   = f.select('Cloud_Multi_Layer_Flag')

        # uncertainties
        cot_uct     = f.select('Cloud_Optical_Thickness_Uncertainty')
        cer_uct     = f.select('Cloud_Effective_Radius_Uncertainty')
        cwp_uct     = f.select('Cloud_Water_Path_Uncertainty')


        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        # ----------------------------------------------------------------------------


        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]
        else:
            lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
            lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

        if self.f03 is None:
            lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=5, extra_grid=True)
            logic_1km = (lon >= lon_range[0]) & (lon <= lon_range[1]) & (lat >= lat_range[0]) & (lat <= lat_range[1])
            lon       = lon[logic_1km]
            lat       = lat[logic_1km]
        else:
            lon       = self.f03.data['lon']['data']
            lat       = self.f03.data['lat']['data']
            logic_1km = self.f03.logic[find_fname_match(fname, self.f03.logic.keys())]['1km']


        lon_5km   = lon0[:]
        lat_5km   = lat0[:]
        logic_5km = (lon_5km >= lon_range[0]) & (lon_5km <= lon_range[1]) & (lat_5km >= lat_range[0]) & (lat_5km <= lat_range[1])
        lon_5km   = lon_5km[logic_5km]
        lat_5km   = lat_5km[logic_5km]
        # -------------------------------------------------------------------------------------------------


        # Calculate 1. ctt, 2. cth, 3. cot, 4. cer, 5. ctp
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        cth           = get_data_h4(cth)[logic_5km]
        ctt           = get_data_h4(ctt)[logic_5km]
        ctp           = get_data_h4(ctp)[logic_1km]

        cot0_data     = get_data_h4(cot0)[logic_1km]
        cer0_data     = get_data_h4(cer0)[logic_1km]
        cwp0_data     = get_data_h4(cwp0)[logic_1km]

        cot_1621_data  = get_data_h4(cot_1621)[logic_1km]
        cer_1621_data  = get_data_h4(cer_1621)[logic_1km]
        cwp_1621_data  = get_data_h4(cwp_1621)[logic_1km]

        cot_pcl_data  = get_data_h4(cot_pcl)[logic_1km]
        cer_pcl_data  = get_data_h4(cer_pcl)[logic_1km]
        cwp_pcl_data  = get_data_h4(cwp_pcl)[logic_1km]

        cot_1621_pcl_data  = get_data_h4(cot_1621_pcl)[logic_1km]
        cer_1621_pcl_data  = get_data_h4(cer_1621_pcl)[logic_1km]
        cwp_1621_pcl_data  = get_data_h4(cwp_1621_pcl)[logic_1km]

        cot_uct = get_data_h4(cot_uct)[logic_1km]
        cer_uct = get_data_h4(cer_uct)[logic_1km]
        cwp_uct = get_data_h4(cwp_uct)[logic_1km]

        cot     = cot0_data.copy()
        cer     = cer0_data.copy()
        cwp     = cwp0_data.copy()

        cot_1621 = cot_1621_data.copy()
        cer_1621 = cer_1621_data.copy()
        cwp_1621 = cwp_1621_data.copy()

        logic_invalid       = (cot0_data < 0.0)      | (cer0_data < 0.0)      | (cwp0_data < 0.0)
        logic_1621_invalid  = (cot_1621_data < 0.0)  | (cer_1621_data < 0.0)  | (cwp_1621_data < 0.0)

        cot[logic_invalid]      = 0.0
        cer[logic_invalid]      = 0.0
        cwp[logic_invalid]      = 0.0

        cot_1621[logic_1621_invalid] = 0.0
        cer_1621[logic_1621_invalid] = 0.0
        cwp_1621[logic_1621_invalid] = 0.0

        cot_uct[logic_invalid]  = 0.0
        cer_uct[logic_invalid]  = 0.0
        cwp_uct[logic_invalid]  = 0.0

        logic_clear      = (cot0_data == 0.0)     | (cer0_data == 0.0)     | (cwp0_data == 0.0)     | (ctp == 1)
        logic_1621_clear = (cot_1621_data == 0.0) | (cer_1621_data == 0.0) | (cwp_1621_data == 0.0) | (ctp == 1)

        cot[logic_clear]     = 0.0
        cer[logic_clear]     = 0.0
        cwp[logic_clear]     = 0.0

        cot_1621[logic_1621_clear] = 0.0
        cer_1621[logic_1621_clear] = 0.0
        cwp_1621[logic_1621_clear] = 0.0

        logic_pcl = ((cot0_data == 0.0)   | (cer0_data == 0.0)   | (cwp0_data == 0.0)) &\
                    ((cot_pcl_data > 0.0) & (cer_pcl_data > 0.0) & (cwp_pcl_data > 0.0))

        logic_1621_pcl = ((cot_1621_data == 0.0)    | (cer_1621_data == 0.0)    | (cwp_1621_data == 0.0)) &\
                         ((cot_1621_pcl_data > 0.0) & (cer_1621_pcl_data > 0.0) & (cwp_1621_pcl_data > 0.0))

        cot[logic_pcl] = cot_pcl_data[logic_pcl]
        cer[logic_pcl] = cer_pcl_data[logic_pcl]
        cwp[logic_pcl] = cwp_pcl_data[logic_pcl]

        cot_1621[logic_1621_pcl] = cot_1621_pcl_data[logic_1621_pcl]
        cer_1621[logic_1621_pcl] = cer_1621_pcl_data[logic_1621_pcl]
        cwp_1621[logic_1621_pcl] = cwp_1621_pcl_data[logic_1621_pcl]

        f.end()
        # -------------------------------------------------------------------------------------------------


        if hasattr(self, 'data'):

            self.logic[fname] = {'1km':logic_1km, '5km':logic_5km}

            self.data['lon']      = dict(name='Longitude',                           data=np.hstack((self.data['lon']['data'], lon)),                   units='degrees')
            self.data['lat']      = dict(name='Latitude',                            data=np.hstack((self.data['lat']['data'], lat)),                   units='degrees')
            self.data['ctp']      = dict(name='Cloud phase optical proprties',       data=np.hstack((self.data['ctp']['data'], ctp)),                   units='N/A')
            self.data['cth']      = dict(name='Cloud top height',                    data=np.hstack((self.data['cth']['data'], cth)),                   units='m')
            self.data['ctt']      = dict(name='Cloud top temperature',               data=np.hstack((self.data['ctt']['data'], ctt)),                   units='K')
            self.data['cot']      = dict(name='Cloud optical thickness',             data=np.hstack((self.data['cot']['data'], cot)),                   units='N/A')
            self.data['cer']      = dict(name='Cloud effective radius',              data=np.hstack((self.data['cer']['data'], cer)),                   units='micron')
            self.data['cwp']      = dict(name='Cloud water path',                    data=np.hstack((self.data['cwp']['data'], cwp)),                   units='g/m^2')
            self.data['cot_1621'] = dict(name='Cloud optical thickness (1621)',      data=np.hstack((self.data['cot_1621']['data'], cot_1621)),         units='N/A')
            self.data['cer_1621'] = dict(name='Cloud effective radius (1621)',       data=np.hstack((self.data['cer_1621']['data'], cer_1621)),         units='micron')
            self.data['cwp_1621'] = dict(name='Cloud water path (1621)',             data=np.hstack((self.data['cwp_1621']['data'], cwp_1621)),         units='g/m^2')
            self.data['cot_uct']  = dict(name='Cloud optical thickness uncertainty', data=np.hstack((self.data['cot_uct']['data'], cot*cot_uct/100.0)), units='N/A')
            self.data['cer_uct']  = dict(name='Cloud effective radius uncertainty',  data=np.hstack((self.data['cer_uct']['data'], cer*cer_uct/100.0)), units='micron')
            self.data['cwp_uct']  = dict(name='Cloud water path uncertainty',        data=np.hstack((self.data['cwp_uct']['data'], cwp*cwp_uct/100.0)), units='g/m^2')
            self.data['lon_5km']  = dict(name='Longitude at 5km',                    data=np.hstack((self.data['lon_5km']['data'], lon_5km)),           units='degrees')
            self.data['lat_5km']  = dict(name='Latitude at 5km',                     data=np.hstack((self.data['lat_5km']['data'], lat_5km)),           units='degrees')

        else:
            self.logic = {}
            self.logic[fname] = {'1km':logic_1km, '5km':logic_5km}

            self.data  = {}


            self.data['lon']      = dict(name='Longitude',                           data=lon,                   units='degrees')
            self.data['lat']      = dict(name='Latitude',                            data=lat,                   units='degrees')
            self.data['ctp']      = dict(name='Cloud phase optical proprties',       data=ctp,                   units='N/A')
            self.data['cth']      = dict(name='Cloud top height',                    data=cth,                   units='m')
            self.data['ctt']      = dict(name='Cloud top temperature',               data=ctt,                   units='K')
            self.data['cot']      = dict(name='Cloud optical thickness',             data=cot,                   units='N/A')
            self.data['cer']      = dict(name='Cloud effective radius',              data=cer,                   units='micron')
            self.data['cwp']      = dict(name='Cloud water path',                    data=cwp,                   units='g/m^2')
            self.data['cot_1621'] = dict(name='Cloud optical thickness (1621)',      data=cot_1621,              units='N/A')
            self.data['cer_1621'] = dict(name='Cloud effective radius (1621)',       data=cer_1621,              units='micron')
            self.data['cwp_1621'] = dict(name='Cloud water path (1621)',             data=cwp_1621,              units='g/m^2')
            self.data['cot_uct']  = dict(name='Cloud optical thickness uncertainty', data=cot*cot_uct/100.0,     units='N/A')
            self.data['cer_uct']  = dict(name='Cloud effective radius uncertainty',  data=cer*cer_uct/100.0,     units='micron')
            self.data['cwp_uct']  = dict(name='Cloud water path uncertainty',        data=cwp*cwp_uct/100.0,     units='g/m^2')



class modis_35_l2:

    """
    Read MODIS level 2 cloud mask product

    Note: We currently only support processing of the cloud mask bytes at a 1 km resolution only.

    Input:
        fnames=   : keyword argument, default=None, Python list of the file path of the original HDF4 file
        f03=      : keyword argument, default=None, Python list of the corresponding geolocation files to fnames
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['lon']
                ['lat']
                ['use_qa']          => 0: not useful (discard), 1: useful
                ['confidence_qa']   => 0: no confidence (do not use), 1: low confidence, 2, ... 7: very high confidence
                ['cloud_mask_flag'] => 0: not determined, 1: determined
                ['fov_qa_cat']      => 0: cloudy, 1: uncertain, 2: probably clear, 3: confident clear
                ['day_night_flag']  => 0: night, 1: day
                ['sunglint_flag']   => 0: not in sunglint path, 1: in sunglint path
                ['snow_ice_flag']   => 0: no snow/ice processing, 1: snow/ice processing path
                ['land_water_cat']  => 0: water, 1: coastal, 2: desert, 3: land
                ['lon_5km']
                ['lat_5km']

    References: (Product Page) https://atmosphere-imager.gsfc.nasa.gov/products/cloud-mask
                (ATBD)         https://atmosphere-imager.gsfc.nasa.gov/sites/default/files/ModAtmo/MOD35_ATBD_Collection6_1.pdf
                (User Guide)   http://cimss.ssec.wisc.edu/modis/CMUSERSGUIDE.PDF
    """


    ID = 'MODIS Level 2 Cloud Mask Product'


    def __init__(self, \
                 fnames,  \
                 f03       = None,  \
                 extent    = None,  \
                 verbose   = False):

        self.fnames     = fnames      # file name of the pickle file
        self.f03        = f03         # geolocation file
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag

        for fname in self.fnames:
            self.read(fname)


    def extract_data(self, dbyte, byte=0):
        """
        Extract cloud mask (in byte format) flags and categories
        """
        if dbyte.dtype != 'uint8':
            dbyte = dbyte.astype('uint8')

        data = np.unpackbits(dbyte, bitorder='big', axis=1) # convert to binary

        if byte == 0:
            # extract flags and categories (*_cat) bit by bit
            land_water_cat  = 2 * data[:, 0] + 1 * data[:, 1] # convert to a value between 0 and 3
            snow_ice_flag   = data[:, 2]
            sunglint_flag   = data[:, 3]
            day_night_flag  = data[:, 4]
            fov_qa_cat      = 2 * data[:, 5] + 1 * data[:, 6] # convert to a value between 0 and 3
            cloud_mask_flag = data[:, 7]
            return cloud_mask_flag, day_night_flag, sunglint_flag, snow_ice_flag, land_water_cat, fov_qa_cat


    def quality_assurance(self, dbyte, byte=0):
        """
        Extract cloud mask QA data to determine confidence
        """
        if dbyte.dtype != 'uint8':
            dbyte = dbyte.astype('uint8')

        data = np.unpackbits(dbyte, bitorder='big', axis=1)

        # process qa flags
        if byte == 0:
            # Byte 0 only has 4 bits of useful information, other 4 are always 0
            confidence_qa = 4 * data[:, 4] + 2 * data[:, 5] + 1 * data[:, 6] # convert to a value between 0 and 7 confidence
            useful_qa = data[:, 7] # usefulness QA flag
            return useful_qa, confidence_qa



    def read(self, fname):

        """
        Read cloud mask flags and tests/categories

        self.data
            ['lon']
            ['lat']
            ['use_qa']          => 0: not useful (discard), 1: useful
            ['confidence_qa']   => 0: no confidence (do not use), 1: low confidence, 2, ... 7: very high confidence
            ['cloud_mask_flag'] => 0: not determined, 1: determined
            ['fov_qa_cat']      => 0: cloudy, 1: uncertain, 2: probably clear, 3: confident clear
            ['day_night_flag']  => 0: night, 1: day
            ['sunglint_flag']   => 0: not in sunglint path, 1: in sunglint path
            ['snow_ice_flag']   => 0: no snow/ice in background, 1: possible snow/ice in background
            ['land_water_cat']  => 0: water, 1: coastal, 2: desert, 3: land
            ['lon_5km']
            ['lat_5km']

        self.logic
        self.logic_5km
        """

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_35_l2]: To use \'modis_35_l2\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        f          = SD(fname, SDC.READ)

        # lon lat
        lat0       = f.select('Latitude')
        lon0       = f.select('Longitude')
        cld_msk0   = f.select('Cloud_Mask')
        qa0        = f.select('Quality_Assurance')


        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data

        #/----------------------------------------------------------------------------\#
        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]
        else:
            lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
            lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

        # Attempt to get lat/lon from geolocation file
        #/----------------------------------------------------------------------------\#
        if self.f03 is None:
            lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=5, extra_grid=True)
            logic_1km = (lon >= lon_range[0]) & (lon <= lon_range[1]) & (lat >= lat_range[0]) & (lat <= lat_range[1])
            lon       = lon[logic_1km]
            lat       = lat[logic_1km]
        else:
            lon       = self.f03.data['lon']['data']
            lat       = self.f03.data['lat']['data']
            logic_1km = self.f03.logic[find_fname_match(fname, self.f03.logic.keys())]['1km']
        #/----------------------------------------------------------------------------\#

        lon_5km   = lon0[:]
        lat_5km   = lat0[:]
        logic_5km = (lon_5km>=lon_range[0]) & (lon_5km<=lon_range[1]) & (lat_5km>=lat_range[0]) & (lat_5km<=lat_range[1])
        lon_5km   = lon_5km[logic_5km]
        lat_5km   = lat_5km[logic_5km]

        # -------------------------------------------------------------------------------------------------

        # Get cloud mask and flag fields
        #/-----------------------------\#
        cm0_data = get_data_h4(cld_msk0)
        qa0_data = get_data_h4(qa0)
        cm = cm0_data.copy()
        qa = qa0_data.copy()

        cm = cm[0, :, :] # read only the first of 6 bytes; rest will be supported in the future if needed
        cm = np.array(cm[logic_1km], dtype='uint8')
        cm = cm.reshape((cm.size, 1))
        cloud_mask_flag, day_night_flag, sunglint_flag, snow_ice_flag, land_water_cat, fov_qa_cat = self.extract_data(cm)


        qa = qa[:, :, 0] # read only the first byte for confidence (indexed differently from cloud mask SDS)
        qa = np.array(qa[logic_1km], dtype='uint8')
        qa = qa.reshape((qa.size, 1))
        use_qa, confidence_qa = self.quality_assurance(qa)

        f.end()
        # -------------------------------------------------------------------------------------------------

        if hasattr(self, 'data'):

            self.logic[fname] = {'1km':logic_1km, '5km':logic_5km}

            self.data['lon']               = dict(name='Longitude',            data=np.hstack((self.data['lon']['data'], lon)),                         units='degrees')
            self.data['lat']               = dict(name='Latitude',             data=np.hstack((self.data['lat']['data'], lat)),                         units='degrees')
            self.data['use_qa']            = dict(name='QA useful',            data=np.hstack((self.data['use_qa']['data'], use_qa)),                   units='N/A')
            self.data['confidence_qa']     = dict(name='QA Mask confidence',   data=np.hstack((self.data['confidence_qa']['data'], confidence_qa)),     units='N/A')
            self.data['cloud_mask_flag']   = dict(name='Cloud mask flag',      data=np.hstack((self.data['cloud_mask_flag']['data'], cloud_mask_flag)), units='N/A')
            self.data['fov_qa_cat']        = dict(name='FOV quality cateogry', data=np.hstack((self.data['fov_qa_cat']['data'], fov_qa_cat)),           units='N/A')
            self.data['day_night_flag']    = dict(name='Day/night flag',       data=np.hstack((self.data['day_night_flag']['data'], day_night_flag)),   units='N/A')
            self.data['sunglint_flag']     = dict(name='Sunglint flag',        data=np.hstack((self.data['sunglint_flag']['data'], sunglint_flag)),     units='N/A')
            self.data['snow_ice_flag']     = dict(name='Snow/ice flag',        data=np.hstack((self.data['snow_flag']['data'], snow_ice_flag)),         units='N/A')
            self.data['land_water_cat']    = dict(name='Land/water flag',      data=np.hstack((self.data['land_water_cat']['data'], land_water_cat)),   units='N/A')
            self.data['lon_5km']           = dict(name='Longitude at 5km',     data=np.hstack((self.data['lon_5km']['data'], lon_5km)),                 units='degrees')
            self.data['lat_5km']           = dict(name='Latitude at 5km',      data=np.hstack((self.data['lat_5km']['data'], lat_5km)),                 units='degrees')

        else:
            self.logic = {}
            self.logic[fname] = {'1km':logic_1km, '5km':logic_5km}

            self.data  = {}
            self.data['lon']             = dict(name='Longitude',            data=lon,             units='degrees')
            self.data['lat']             = dict(name='Latitude',             data=lat,             units='degrees')
            self.data['use_qa']          = dict(name='QA useful',            data=use_qa,          units='N/A')
            self.data['confidence_qa']   = dict(name='QA Mask confidence',   data=confidence_qa,   units='N/A')
            self.data['cloud_mask_flag'] = dict(name='Cloud mask flag',      data=cloud_mask_flag, units='N/A')
            self.data['fov_qa_cat']      = dict(name='FOV quality category', data=fov_qa_cat,      units='N/A')
            self.data['day_night_flag']  = dict(name='Day/night flag',       data=day_night_flag,  units='N/A')
            self.data['sunglint_flag']   = dict(name='Sunglint flag',        data=sunglint_flag,   units='N/A')
            self.data['snow_ice_flag']   = dict(name='Snow/ice flag',        data=snow_ice_flag,   units='N/A')
            self.data['land_water_cat']  = dict(name='Land/water category',  data=land_water_cat,  units='N/A')
            self.data['lon_5km']         = dict(name='Longitude at 5km',     data=lon_5km,         units='degrees')
            self.data['lat_5km']         = dict(name='Latitude at 5km',      data=lat_5km,         units='degrees')



class modis_03:

    """
    Read MODIS 03 geolocation data

    Input:
        fnames=   : keyword argument, default=None, Python list of the file path of the original HDF4 file
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        vnames=   : keyword argument, default=[], additional variable names to be read in to self.data
        overwrite=: keyword argument, default=False, whether to overwrite or not
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['lon']
                ['lat']
                ['sza']
                ['saa']
                ['vza']
                ['vaa']
    """


    ID = 'MODIS 03 Geolocation Product'


    def __init__(self, \
                 fnames    = None,  \
                 extent    = None,  \
                 vnames    = [],    \
                 verbose   = False):

        self.fnames     = fnames      # file name of the pickle file
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag

        for fname in self.fnames:

            self.read(fname)

            if len(vnames) > 0:
                self.read_vars(fname, vnames=vnames)


    def read(self, fname):

        """
        Read solar and sensor angles

        self.data
            ['lon']
            ['lat']
            ['sza']
            ['saa']
            ['vza']
            ['vaa']

        self.logic
        """

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_03]: To use \'modis_03\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        f     = SD(fname, SDC.READ)

        # lon lat
        lat0       = f.select('Latitude')
        lon0       = f.select('Longitude')

        sza0       = f.select('SolarZenith')
        saa0       = f.select('SolarAzimuth')
        vza0       = f.select('SensorZenith')
        vaa0       = f.select('SensorAzimuth')


        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        lon = lon0[:]
        lat = lat0[:]

        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]
        else:
            lon_range = [self.extent[0], self.extent[1]]
            lat_range = [self.extent[2], self.extent[3]]

        logic     = (lon >= lon_range[0]) & (lon <= lon_range[1]) & (lat >= lat_range[0]) & (lat <= lat_range[1])
        lon       = lon[logic]
        lat       = lat[logic]
        # -------------------------------------------------------------------------------------------------


        # Calculate 1. sza, 2. saa, 3. vza, 4. vaa
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sza0_data = get_data_h4(sza0)
        saa0_data = get_data_h4(saa0)
        vza0_data = get_data_h4(vza0)
        vaa0_data = get_data_h4(vaa0)

        sza = sza0_data[logic]
        saa = saa0_data[logic]
        vza = vza0_data[logic]
        vaa = vaa0_data[logic]

        f.end()
        # -------------------------------------------------------------------------------------------------

        if hasattr(self, 'data'):

            self.logic[fname] = {'1km':logic}

            self.data['lon']   = dict(name='Longitude'                 , data=np.hstack((self.data['lon']['data'], lon    )), units='degrees')
            self.data['lat']   = dict(name='Latitude'                  , data=np.hstack((self.data['lat']['data'], lat    )), units='degrees')
            self.data['sza']   = dict(name='Solar Zenith Angle'        , data=np.hstack((self.data['sza']['data'], sza    )), units='degrees')
            self.data['saa']   = dict(name='Solar Azimuth Angle'       , data=np.hstack((self.data['saa']['data'], saa    )), units='degrees')
            self.data['vza']   = dict(name='Sensor Zenith Angle'       , data=np.hstack((self.data['vza']['data'], vza    )), units='degrees')
            self.data['vaa']   = dict(name='Sensor Azimuth Angle'      , data=np.hstack((self.data['vaa']['data'], vaa    )), units='degrees')

        else:
            self.logic = {}
            self.logic[fname] = {'1km':logic}

            self.data  = {}
            self.data['lon']   = dict(name='Longitude'                 , data=lon    , units='degrees')
            self.data['lat']   = dict(name='Latitude'                  , data=lat    , units='degrees')
            self.data['sza']   = dict(name='Solar Zenith Angle'        , data=sza    , units='degrees')
            self.data['saa']   = dict(name='Solar Azimuth Angle'       , data=saa    , units='degrees')
            self.data['vza']   = dict(name='Sensor Zenith Angle'       , data=vza    , units='degrees')
            self.data['vaa']   = dict(name='Sensor Azimuth Angle'      , data=vaa    , units='degrees')


    def read_vars(self, fname, vnames=[]):

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_03]: To use \'modis_03\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        logic = self.logic[fname]['1km']

        f     = SD(fname, SDC.READ)

        for vname in vnames:

            data0 = f.select(vname)
            data  = get_data_h4(data0)[logic]
            if vname.lower() in self.data.keys():
                self.data[vname.lower()] = dict(name=vname, data=np.hstack((self.data[vname.lower()]['data'], data)), units=data0.attributes()['units'])
            else:
                self.data[vname.lower()] = dict(name=vname, data=data, units=data0.attributes()['units'])

        f.end()


class viirs_03:

    """
    Read VIIRS 03 geolocation data

    Input:
        fnames=   : keyword argument, default=None, Python list of the file path of the original netCDF file
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        vnames=   : keyword argument, default=[], additional variable names to be read in to self.data
        overwrite=: keyword argument, default=False, whether to overwrite or not
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['lon']
                ['lat']
                ['sza']
                ['saa']
                ['vza']
                ['vaa']
    """


    ID = 'VIIRS 03 Geolocation Product'


    def __init__(self, \
                 fnames    = None,  \
                 extent    = None,  \
                 vnames    = [],    \
                 overwrite = False, \
                 verbose   = False):

        self.fnames     = fnames      # file name of the raw netCDF files
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag

        for fname in self.fnames:

            self.read(fname)

            if len(vnames) > 0:
                self.read_vars(fname, vnames=vnames)

    def read(self, fname):

        """
        Read solar and sensor angles

        self.data
            ['lon']
            ['lat']
            ['sza']
            ['saa']
            ['vza']
            ['vaa']

        self.logic
        """

        # placeholder for xarray
        #/-----------------------------------------------------------------------------\
        # if er3t.common.has_xarray:
        #     import xarray as xr
        #     with xr.open_dataset(fname, group='geolocation_data') as f:
        #         lon0 = f.longitude
        #         lat0 = f.latitude
        #         sza0 = f.solar_zenith
        #         saa0 = f.solar_azimuth
        #         vza0 = f.sensor_zenith
        #         vaa0 = f.sensor_azimuth
        #\-----------------------------------------------------------------------------/

        from netCDF4 import Dataset
        f     = Dataset(fname, 'r')

        # geolocation
        #/-----------------------------------------------------------------------------\
        lat0 = f.groups['geolocation_data'].variables['latitude']
        lon0 = f.groups['geolocation_data'].variables['longitude']
        #\-----------------------------------------------------------------------------/

        # only crop necessary data
        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        #/-----------------------------------------------------------------------------\
        if self.extent is None:

            if 'valid_min' in lon0.ncattrs():
                lon_range = [lon0.getncattr('valid_min'), lon0.getncattr('valid_max')]
                lat_range = [lon0.getncattr('valid_min'), lon0.getncattr('valid_max')]
            else:
                lon_range = [-180.0, 180.0]
                lat_range = [-90.0 , 90.0]

        else:

            lon_range = [self.extent[0]-0.01, self.extent[1]+0.01]
            lat_range = [self.extent[2]-0.01, self.extent[3]+0.01]

        lon = get_data_nc(lon0)
        lat = get_data_nc(lat0)

        logic = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])
        lon   = lon[logic]
        lat   = lat[logic]
        #\-----------------------------------------------------------------------------/

        # solar geometries
        #/-----------------------------------------------------------------------------\
        sza0 = f.groups['geolocation_data'].variables['solar_zenith']
        saa0 = f.groups['geolocation_data'].variables['solar_azimuth']
        #\-----------------------------------------------------------------------------/

        # sensor geometries
        #/-----------------------------------------------------------------------------\
        vza0 = f.groups['geolocation_data'].variables['sensor_zenith']
        vaa0 = f.groups['geolocation_data'].variables['sensor_azimuth']
        #\-----------------------------------------------------------------------------/

        # Calculate 1. sza, 2. saa, 3. vza, 4. vaa
        #/-----------------------------------------------------------------------------\
        sza0_data = get_data_nc(sza0)
        saa0_data = get_data_nc(saa0)
        vza0_data = get_data_nc(vza0)
        vaa0_data = get_data_nc(vaa0)

        sza = sza0_data[logic]
        saa = saa0_data[logic]
        vza = vza0_data[logic]
        vaa = vaa0_data[logic]

        f.close()
        #\-----------------------------------------------------------------------------/

        if hasattr(self, 'data'):

            self.logic[get_fname_pattern(fname)] = {'mask':logic}

            self.data['lon'] = dict(name='Longitude'           , data=np.hstack((self.data['lon']['data'], lon)), units='degrees')
            self.data['lat'] = dict(name='Latitude'            , data=np.hstack((self.data['lat']['data'], lat)), units='degrees')
            self.data['sza'] = dict(name='Solar Zenith Angle'  , data=np.hstack((self.data['sza']['data'], sza)), units='degrees')
            self.data['saa'] = dict(name='Solar Azimuth Angle' , data=np.hstack((self.data['saa']['data'], saa)), units='degrees')
            self.data['vza'] = dict(name='Sensor Zenith Angle' , data=np.hstack((self.data['vza']['data'], vza)), units='degrees')
            self.data['vaa'] = dict(name='Sensor Azimuth Angle', data=np.hstack((self.data['vaa']['data'], vaa)), units='degrees')

        else:
            self.logic = {}
            self.logic[get_fname_pattern(fname)] = {'mask':logic}

            self.data  = {}
            self.data['lon'] = dict(name='Longitude'           , data=lon, units='degrees')
            self.data['lat'] = dict(name='Latitude'            , data=lat, units='degrees')
            self.data['sza'] = dict(name='Solar Zenith Angle'  , data=sza, units='degrees')
            self.data['saa'] = dict(name='Solar Azimuth Angle' , data=saa, units='degrees')
            self.data['vza'] = dict(name='Sensor Zenith Angle' , data=vza, units='degrees')
            self.data['vaa'] = dict(name='Sensor Azimuth Angle', data=vaa, units='degrees')

    def read_vars(self, fname, vnames=[]):

        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Error [viirs_03]: Please install <netCDF4> to proceed.'
            raise ImportError(msg)

        logic = self.logic[get_fname_pattern(fname)]['mask']

        f = Dataset(fname, 'r')

        for vname in vnames:

            data0 = f.groups['geolocation_data'].variables[vname]
            data  = get_data_nc(data0)
            if vname.lower() in self.data.keys():
                self.data[vname.lower()] = dict(name=vname.lower().title(), data=np.vstack((self.data[vname.lower()]['data'], data)), units=data0.getncattr('units'))
            else:
                self.data[vname.lower()] = dict(name=vname.lower().title(), data=data, units=data0.getncattr('units'))

        f.close()




class viirs_l1b:

    """
    Read VIIRS Level 1b file, e.g., VNP02MOD, into an object <viirs_l1b>

    Input:
        fnames=     : keyword argument, default=None, Python list of the file path of the original netCDF files
        f03=        : keyword argument, default=None, class object obtained from `viirs_03` reader for geolocation
        verbose=    : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['wvl']
                ['rad']
                ['ref']
    """


    ID = 'VIIRS Level 1b Calibrated Radiance'


    def __init__(self, \
                 fnames     = None,  \
                 f03        = None,  \
                 extent     = None,  \
                 bands      = None,  \
                 verbose    = False):

        self.fnames     = fnames      # Python list of netCDF filenames
        self.f03        = f03         # geolocation class object created using the `viirs_03` reader
        self.bands      = bands       # Python list of bands to extract information


        filename = os.path.basename(fnames[0]).lower()
        if '02img' in filename:
            self.resolution = 0.375
            if bands is None:
                self.bands = list(VIIRS_L1B_IMG_BANDS.keys())
                if verbose:
                    msg = 'Message [viirs_l1b]: Data will be extracted for the following bands %s' % VIIRS_L1B_IMG_BANDS

            elif (bands is not None) and not (set(bands).issubset(set(VIIRS_L1B_IMG_BANDS.keys()))):

                msg = 'Error [viirs_l1b]: Bands must be one or more of %s' % list(VIIRS_L1B_IMG_BANDS.keys())
                raise KeyError(msg)

        elif ('02mod' in filename) or ('02dnb' in filename):
            self.resolution = 0.75
            if bands is None:
                self.bands = list(VIIRS_L1B_MOD_BANDS.keys())
                if verbose:
                    msg = 'Message [viirs_l1b]: Data will be extracted for the following bands %s' % VIIRS_L1B_MOD_BANDS

            elif (bands is not None) and not (set(bands).issubset(set(VIIRS_L1B_MOD_BANDS.keys()))):

                msg = 'Error [viirs_l1b]: Bands must be one or more of %s' % list(VIIRS_L1B_MOD_BANDS.keys())
                raise KeyError(msg)
        else:
            msg = 'Error [viirs_l1b]: Currently, only IMG (0.375km) and MOD (0.75km) products are supported.'
            raise ValueError(msg)


        if extent is not None and verbose:
            msg = '\nMessage [viirs_l1b]: The `extent` argument will be ignored as it is only available for consistency.\n' \
                  'If only region of interest is needed, please use `viirs_03` reader and pass the class object here via `f03=`.\n'
            print(msg)

        if f03 is None and verbose:
            msg = '\nMessage [viirs_l1b]: Geolocation data not provided. File will be read without geolocation.\n'
            print(msg)

        for i in range(len(fnames)):
            self.read(fnames[i])


    def _remove_flags(self, nc_dset, fill_value=np.nan):
        """
        Method to remove all flags without masking.
        This could remove a significant portion of the image.
        """
        nc_dset.set_auto_maskandscale(False)
        data = nc_dset[:]
        data = data.astype('float') # convert to float to use nan
        flags = nc_dset.getncattr('flag_values')

        for flag in flags:
            data[data == flag] = fill_value

        return data


    def _mask_flags(self, nc_dset, fill_value=np.nan):
        """
        Method to keep all flags by masking them with NaN.
        This retains the full image but artifacts exist at extreme swath edges.
        """
        nc_dset.set_auto_scale(False)
        nc_dset.set_auto_mask(True)
        data = nc_dset[:]
        data = np.ma.masked_array(data.data, data.mask, fill_value=fill_value, dtype='float')
        flags = nc_dset.getncattr('flag_values')

        for flag in flags:
            data = np.ma.masked_equal(data.data, flag, copy=False)

        data.filled(fill_value=fill_value)
        return data


    def read(self, fname):

        """
        Read radiance and reflectance from the VIIRS L1b data
        self.data
            ['wvl']
            ['rad']
            ['ref']
        """

        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Error [viirs_l1b]: Please install <netCDF4> to proceed.'
            raise ImportError(msg)


        f   = Dataset(fname, 'r')

        # Calculate 1. radiance, 2. reflectance from the raw data
        #/-----------------------------------------------------------------------------\

        if self.f03 is not None:
            mask = self.f03.logic[get_fname_pattern(fname)]['mask']
            rad  = np.zeros((len(self.bands), mask[mask==True].size))
            ref  = np.zeros((len(self.bands), mask[mask==True].size))

        else:
            rad = np.zeros((len(self.bands),
                           f.groups['observation_data'].variables[self.bands[0]].shape[0],
                           f.groups['observation_data'].variables[self.bands[0]].shape[1]))
            ref = np.zeros((len(self.bands),
                           f.groups['observation_data'].variables[self.bands[0]].shape[0],
                           f.groups['observation_data'].variables[self.bands[0]].shape[1]))

        wvl = np.zeros(len(self.bands), dtype='uint16')

        ## Calculate 1. radiance, 2. reflectance from the raw data
        #\-----------------------------------------------------------------------------/
        for i in range(len(self.bands)):

            nc_dset = f.groups['observation_data'].variables[self.bands[i]]
            data = self._remove_flags(nc_dset)
            if self.f03 is not None:
                data = data[mask]

            # apply scaling, offset, and unit conversions
            # add_offset is usually 0. for VIIRS solar bands
            rad0 = (data - nc_dset.getncattr('radiance_add_offset')) * nc_dset.getncattr('radiance_scale_factor')

            # if nc_dset.getncattr('radiance_units').endswith('micrometer'):
            rad0 /= 1000. # from <per micron> to <per nm>
            rad[i] = rad0

            ref[i] = (data - nc_dset.getncattr('add_offset')) * nc_dset.getncattr('scale_factor')
            wvl[i] = VIIRS_ALL_BANDS[self.bands[i]]

        f.close()
        #\-----------------------------------------------------------------------------/
        if hasattr(self, 'data'):
            self.data['wvl'] = dict(name='Wavelengths', data=np.hstack((self.data['wvl']['data'], wvl)), units='nm')
            self.data['rad'] = dict(name='Radiance'   , data=np.hstack((self.data['rad']['data'], rad)), units='W/m^2/nm/sr')
            self.data['ref'] = dict(name='Reflectance', data=np.hstack((self.data['ref']['data'], ref)), units='N/A')

        else:
            self.data = {}
            self.data['wvl'] = dict(name='Wavelengths', data=wvl       , units='nm')
            self.data['rad'] = dict(name='Radiance'   , data=rad       , units='W/m^2/nm/sr')
            self.data['ref'] = dict(name='Reflectance', data=ref       , units='N/A')


    def save_h5(self, fname):

        f = h5py.File(fname, 'w')
        for key in self.data.keys():
            f[key] = self.data[key]['data']
        f.close()





class viirs_cldprop_l2:
    """
    Read VIIRS Level 2 cloud properties/mask file, e.g., CLDPROP_L2_VIIRS_SNPP..., into an object <viirs_cldprop>

    Input:
        fnames=     : keyword argument, Python list of the file path of the original netCDF files
        extent=     : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        maskvars=   : keyword argument, default=False, extracts optical properties by default; set to False to get cloud mask data

    Output:
        self.data
                ['lon']
                ['lat']
                ['ctp']
                ['cth']
                ['cot']
                ['cer']
                ['cwp']
                ['cot_uct']
                ['cer_uct']
                ['cer_uct']
                ['pcl']

                or
        self.data
                ['lon']
                ['lat']
                ['ret_std_conf_qa'] => 0: no confidence (do not use), 1: marginal, 2: good, 3: very good
                ['cld_type_qa']     => 0: no cloud mask, 1: no cloud, 2: water cloud, 3: ice cloud, 4: unknown cloud
                ['bowtie_qa']       => 0: normal pixel, 1: bowtie pixel
                ['cloud_mask_flag'] => 0: not determined, 1: determined
                ['fov_qa_cat']      => 0: cloudy, 1: uncertain, 2: probably clear, 3: confident clear
                ['day_night_flag']  => 0: night, 1: day
                ['sunglint_flag']   => 0: in sunglint path, 1: not in sunglint path
                ['snow_ice_flag']   => 0: snow/ice background processing, 1: no snow/ice processing path
                ['land_water_cat']  => 0: water, 1: coastal, 2: desert, 3: land
                ['lon_5km']
                ['lat_5km']

    References: (Product Page) https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/CLDPROP_L2_VIIRS_NOAA20
                (User Guide)   https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/Document%20Archive/Science%20Data%20Product%20Documentation/L2_Cloud_Properties_UG_v1.2_March_2021.pdf

    """


    ID = 'VIIRS Level 2 Cloud Properties and Mask'

    def __init__(self, fnames, f03=None, extent=None, maskvars=False):

        self.fnames = fnames
        self.f03    = f03
        self.extent = extent

        for fname in self.fnames:
            if maskvars:
                self.read_mask(fname)
            else:
                self.read_cop(fname)


    def extract_data(self, dbyte, byte=0):
        """
        Extract cloud mask (in byte format) flags and categories
        """
        if dbyte.dtype != 'uint8':
            dbyte = dbyte.astype('uint8')

        data = np.unpackbits(dbyte, bitorder='big', axis=1) # convert to binary

        if byte == 0:
            # extract flags and categories (*_cat) bit by bit
            land_water_cat  = 2 * data[:, 0] + 1 * data[:, 1] # convert to a value between 0 and 3
            snow_ice_flag   = data[:, 2]
            sunglint_flag   = data[:, 3]
            day_night_flag  = data[:, 4]
            fov_qa_cat      = 2 * data[:, 5] + 1 * data[:, 6] # convert to a value between 0 and 3
            cloud_mask_flag = data[:, 7]
            return cloud_mask_flag, day_night_flag, sunglint_flag, snow_ice_flag, land_water_cat, fov_qa_cat


    def quality_assurance(self, dbyte, byte=0):
        """
        Extract cloud mask QA data to determine quality

        Reference: VIIRS CLDPROP User Guide, Version 2.1, March 2021
        Filespec:  https://atmosphere-imager.gsfc.nasa.gov/sites/default/files/ModAtmo/dump_CLDPROP_L2_V011.txt
        """
        if dbyte.dtype != 'uint8':
            dbyte = dbyte.astype('uint8')

        data = np.unpackbits(dbyte, bitorder='big', axis=1)

        # process qa flags
        if byte == 0: # byte 0 has spectral retrieval QA

            # 1.6-2.1 retrieval QA
            ret_1621      = data[:, 0]
            ret_1621_conf = 2 * data[:, 1] + 1 * data[:, 2] # convert to a value between 0 and 3 confidence
            ret_1621_data = data[:, 3]

            # VNSWIR-2.1 or Standard (std) Retrieval QA
            ret_std      = data[:, 4]
            ret_std_conf = 2 * data[:, 5] + 1 * data[:, 6] # convert to a value between 0 and 3 confidence
            ret_std_data = data[:, 7]

            return ret_std, ret_std_conf, ret_std_data, ret_1621, ret_1621_conf, ret_1621_data

        elif byte == 1: # byte 1 has cloud QA

            bowtie            = data[:, 0] # bow tie effect
            cot_oob           = data[:, 1] # cloud optical thickness out of bounds
            cot_bands         = 2 * data[:, 2] + 1 * data[:, 3] # convert to a value between 0 and 3
            rayleigh          = data[:, 4] # whether rayleigh correction was applied
            cld_type_process  = 4 * data[:, 5] + 2 * data[:, 6] + 1 * data[:, 7]

            return cld_type_process, rayleigh, cot_bands, cot_oob, bowtie


    def read_mask(self, fname):
        """
        Function to extract cloud mask variables from the file
        """
        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Error [viirs_cldprop_l2]: Please install <netCDF4> to proceed.'
            raise ImportError(msg)

        # ------------------------------------------------------------------------------------ #
        f = Dataset(fname, 'r')

        cld_msk0       = f.groups['geophysical_data'].variables['Cloud_Mask']
        qua_assurance0 = f.groups['geophysical_data'].variables['Quality_Assurance']

        #/----------------------------------------------------------------------------\#
        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]

        else:
            lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
            lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

        # Select required region only
        if self.f03 is None:

            lat           = f.groups['geolocation_data'].variables['latitude'][...]
            lon           = f.groups['geolocation_data'].variables['longitude'][...]
            logic_extent  = (lon >= lon_range[0]) & (lon <= lon_range[1]) & \
                            (lat >= lat_range[0]) & (lat <= lat_range[1])
            lon           = lon[logic_extent]
            lat           = lat[logic_extent]

        else:
            lon          = self.f03.data['lon']['data']
            lat          = self.f03.data['lat']['data']
            logic_extent = self.f03.logic[get_fname_pattern(fname)]['mask']

        # Get cloud mask and flag fields
        #/-----------------------------\#
        cm_data = get_data_nc(cld_msk0)
        qa_data = get_data_nc(qua_assurance0)
        cm = cm_data.copy()
        qa = qa_data.copy()

        cm0 = cm[:, :, 0] # read only the first byte; rest will be supported in the future if needed
        cm0 = np.array(cm0[logic_extent], dtype='uint8')
        cm0 = cm0.reshape((cm0.size, 1))
        cloud_mask_flag, day_night_flag, sunglint_flag, snow_ice_flag, land_water_cat, fov_qa_cat = self.extract_data(cm0)

        qa0 = qa[:, :, 0] # read byte 0 for confidence QA (note that indexing is different from MODIS)
        qa0 = np.array(qa0[logic_extent], dtype='uint8')
        qa0 = qa0.reshape((qa0.size, 1))
        _, ret_std_conf_qa, _, _, ret_1621_conf_qa, _ = self.quality_assurance(qa0, byte=0) # only get confidence

        qa1 = qa[:, :, 1] # read byte 1 for confidence QA (note that indexing is different from MODIS)
        qa1 = np.array(qa1[logic_extent], dtype='uint8')
        qa1 = qa1.reshape((qa1.size, 1))
        cld_type_qa, _, _, _, bowtie_qa = self.quality_assurance(qa1, byte=1)

        f.close()
        # -------------------------------------------------------------------------------------------------

        # save the data
        if hasattr(self, 'data'):

            self.logic[fname] = {'0.75km':logic_extent}

            self.data['lon']               = dict(name='Longitude',                        data=np.hstack((self.data['lon']['data'], lon)),                                 units='degrees')
            self.data['lat']               = dict(name='Latitude',                         data=np.hstack((self.data['lat']['data'], lat)),                                 units='degrees')
            self.data['ret_std_conf_qa']   = dict(name='QA standard retrieval confidence', data=np.hstack((self.data['ret_std_conf_qa']['data'], ret_std_conf_qa)),         units='N/A')
            self.data['ret_1621_conf_qa']  = dict(name='QA 1.6-2.1 retrieval confidence',  data=np.hstack((self.data['ret_1621_conf_qa']['data'], ret_1621_conf_qa)),       units='N/A')
            self.data['cld_type_qa']       = dict(name='QA cloud type processing path',    data=np.hstack((self.data['cld_type_qa']['data'], cld_type_qa)),                 units='N/A')
            self.data['bowtie_qa']         = dict(name='QA bowtie pixel',                  data=np.hstack((self.data['bowtie_qa']['data'], bowtie_qa)),                     units='N/A')
            self.data['cloud_mask_flag']   = dict(name='Cloud mask flag',                  data=np.hstack((self.data['cloud_mask_flag']['data'], cloud_mask_flag)),         units='N/A')
            self.data['fov_qa_cat']        = dict(name='FOV quality cateogry',             data=np.hstack((self.data['fov_qa_cat']['data'], fov_qa_cat)),                   units='N/A')
            self.data['day_night_flag']    = dict(name='Day/night flag',                   data=np.hstack((self.data['day_night_flag']['data'], day_night_flag)),           units='N/A')
            self.data['sunglint_flag']     = dict(name='Sunglint flag',                    data=np.hstack((self.data['sunglint_flag']['data'], sunglint_flag)),             units='N/A')
            self.data['snow_ice_flag']     = dict(name='Snow/ice flag',                    data=np.hstack((self.data['snow_flag']['data'], snow_ice_flag)),                 units='N/A')
            self.data['land_water_cat']    = dict(name='Land/water flag',                  data=np.hstack((self.data['land_water_cat']['data'], land_water_cat)),           units='N/A')

        else:
            self.logic = {}
            self.logic[fname] = {'0.75km':logic_extent}
            self.data  = {}

            self.data['lon']              = dict(name='Longitude',                        data=lon,              units='degrees')
            self.data['lat']              = dict(name='Latitude',                         data=lat,              units='degrees')
            self.data['ret_std_conf_qa']  = dict(name='QA standard retrieval confidence', data=ret_std_conf_qa,  units='N/A')
            self.data['ret_1621_conf_qa'] = dict(name='QA 1.6-2.1 retrieval confidence',  data=ret_1621_conf_qa, units='N/A')
            self.data['cld_type_qa']      = dict(name='QA cloud type processing path',    data=cld_type_qa,      units='N/A')
            self.data['bowtie_qa']        = dict(name='QA bowtie pixel',                  data=bowtie_qa,        units='N/A')
            self.data['cloud_mask_flag']  = dict(name='Cloud mask flag',                  data=cloud_mask_flag,  units='N/A')
            self.data['fov_qa_cat']       = dict(name='FOV quality category',             data=fov_qa_cat,       units='N/A')
            self.data['day_night_flag']   = dict(name='Day/night flag',                   data=day_night_flag,   units='N/A')
            self.data['sunglint_flag']    = dict(name='Sunglint flag',                    data=sunglint_flag,    units='N/A')
            self.data['snow_ice_flag']    = dict(name='Snow/ice flag',                    data=snow_ice_flag,    units='N/A')
            self.data['land_water_cat']   = dict(name='Land/water category',              data=land_water_cat,   units='N/A')


    def read_cop(self, fname):
        """
        Function to extract cloud optical properties from the file
        """
        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Error [viirs_cldprop_l2]: Please install <netCDF4> to proceed.'
            raise ImportError(msg)

        # ------------------------------------------------------------------------------------ #
        f = Dataset(fname, 'r')

        #------------------------------------Cloud variables------------------------------------#
        ctp  = f.groups['geophysical_data'].variables['Cloud_Phase_Optical_Properties']
        cth  = f.groups['geophysical_data'].variables['Cloud_Top_Height']


        # TODO
        # Support for cloud mask             (byte format)
        #--------------------------------------Retrievals---------------------------------------#
        cot0 = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness']
        cer0 = f.groups['geophysical_data'].variables['Cloud_Effective_Radius']
        cwp0 = f.groups['geophysical_data'].variables['Cloud_Water_Path']

        #-------------------------------------PCL variables-------------------------------------#
        cot_pcl = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness_PCL']
        cer_pcl = f.groups['geophysical_data'].variables['Cloud_Effective_Radius_PCL']
        cwp_pcl = f.groups['geophysical_data'].variables['Cloud_Water_Path_PCL']

        #----------------------------------1621 Retrievals--------------------------------------#
        cot_1621 = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness_1621']
        cer_1621 = f.groups['geophysical_data'].variables['Cloud_Effective_Radius_1621']
        cwp_1621 = f.groups['geophysical_data'].variables['Cloud_Water_Path_1621']

        #---------------------------------1621 PCL variables------------------------------------#
        cot_1621_pcl = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness_1621_PCL']
        cer_1621_pcl = f.groups['geophysical_data'].variables['Cloud_Effective_Radius_1621_PCL']
        cwp_1621_pcl = f.groups['geophysical_data'].variables['Cloud_Water_Path_1621_PCL']

        #-------------------------------------Uncertainties-------------------------------------#
        cot_uct0 = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness_Uncertainty']
        cer_uct0 = f.groups['geophysical_data'].variables['Cloud_Effective_Radius_Uncertainty']
        cwp_uct0 = f.groups['geophysical_data'].variables['Cloud_Water_Path_Uncertainty']


        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]
        else:
            lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
            lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

        # Select required region only
        if self.f03 is None:

            lat           = f.groups['geolocation_data'].variables['latitude'][...]
            lon           = f.groups['geolocation_data'].variables['longitude'][...]
            logic_extent  = (lon >= lon_range[0]) & (lon <= lon_range[1]) & \
                            (lat >= lat_range[0]) & (lat <= lat_range[1])
            lon           = lon[logic_extent]
            lat           = lat[logic_extent]

        else:
            lon          = self.f03.data['lon']['data']
            lat          = self.f03.data['lat']['data']
            logic_extent = self.f03.logic[get_fname_pattern(fname)]['mask']


        # Retrieve 1. ctp, 2. cth, 3. cot, 4. cer, 5. cwp, and select regional extent
        ctp           = get_data_nc(ctp, replace_fill_value=None)[logic_extent]
        cth           = get_data_nc(cth, replace_fill_value=None)[logic_extent]

        cot0_data     = get_data_nc(cot0)[logic_extent]
        cer0_data     = get_data_nc(cer0)[logic_extent]
        cwp0_data     = get_data_nc(cwp0)[logic_extent]
        cot_pcl_data  = get_data_nc(cot_pcl)[logic_extent]
        cer_pcl_data  = get_data_nc(cer_pcl)[logic_extent]
        cwp_pcl_data  = get_data_nc(cwp_pcl)[logic_extent]

        cot_1621_data     = get_data_nc(cot_1621)[logic_extent]
        cer_1621_data     = get_data_nc(cer_1621)[logic_extent]
        cwp_1621_data     = get_data_nc(cwp_1621)[logic_extent]
        cot_1621_pcl_data = get_data_nc(cot_1621_pcl)[logic_extent]
        cer_1621_pcl_data = get_data_nc(cer_1621_pcl)[logic_extent]
        cwp_1621_pcl_data = get_data_nc(cwp_1621_pcl)[logic_extent]

        cot_uct = get_data_nc(cot_uct0)[logic_extent]
        cer_uct = get_data_nc(cer_uct0)[logic_extent]
        cwp_uct = get_data_nc(cwp_uct0)[logic_extent]

        # Make copies to modify
        cot     = cot0_data.copy()
        cer     = cer0_data.copy()
        cwp     = cwp0_data.copy()

        cot_1621 = cot_1621_data.copy()
        cer_1621 = cer_1621_data.copy()
        cwp_1621 = cwp_1621_data.copy()

        # make invalid pixels clear-sky
        logic_invalid       = (cot0_data < 0.0)     | (cer0_data < 0.0)     | (cwp0_data < 0.0) | (ctp == 0)
        logic_1621_invalid  = (cot_1621_data < 0.0) | (cer_1621_data < 0.0) | (cwp_1621_data < 0.0) | (ctp == 0)
        cot[logic_invalid]     = 0.0
        cer[logic_invalid]     = 0.0
        cwp[logic_invalid]     = 0.0

        cot_uct[logic_invalid] = 0.0
        cer_uct[logic_invalid] = 0.0
        cwp_uct[logic_invalid] = 0.0

        cot_1621[logic_1621_invalid] = 0.0
        cer_1621[logic_1621_invalid] = 0.0
        cwp_1621[logic_1621_invalid] = 0.0

        logic_clear      = (cot0_data == 0.0)     | (cer0_data == 0.0)     | (cwp0_data == 0.0)     | (ctp == 1)
        logic_1621_clear = (cot_1621_data == 0.0) | (cer_1621_data == 0.0) | (cwp_1621_data == 0.0) | (ctp == 1)

        cot[logic_clear]     = 0.0
        cer[logic_clear]     = 0.0
        cwp[logic_clear]     = 0.0

        cot_1621[logic_1621_clear] = 0.0
        cer_1621[logic_1621_clear] = 0.0
        cwp_1621[logic_1621_clear] = 0.0

        # use the partially cloudy data to fill in potential missed clouds
        logic_pcl = ((cot0_data == 0.0)    | (cer0_data == 0.0)    | (cwp0_data == 0.0)) &\
                    ((cot_pcl_data > 0.0)  & (cer_pcl_data > 0.0)  & (cwp_pcl_data > 0.0))
        logic_1621_pcl = ((cot_1621_data == 0.0)    | (cer_1621_data == 0.0)    | (cwp_1621_data == 0.0)) &\
                         ((cot_1621_pcl_data > 0.0) & (cer_1621_pcl_data > 0.0) & (cwp_1621_pcl_data > 0.0))

        cot[logic_pcl] = cot_pcl_data[logic_pcl]
        cer[logic_pcl] = cer_pcl_data[logic_pcl]
        cwp[logic_pcl] = cwp_pcl_data[logic_pcl]

        cot_1621[logic_1621_pcl] = cot_1621_pcl_data[logic_1621_pcl]
        cer_1621[logic_1621_pcl] = cer_1621_pcl_data[logic_1621_pcl]
        cwp_1621[logic_1621_pcl] = cwp_1621_pcl_data[logic_1621_pcl]


        f.close()
        # ------------------------------------------------------------------------------------ #

        # save the data
        if hasattr(self, 'data'):

            self.logic[fname] = {'0.75km':logic_extent}

            self.data['lon']      = dict(name='Longitude',                           data=np.hstack((self.data['lon']['data'], lon)),                   units='degrees')
            self.data['lat']      = dict(name='Latitude',                            data=np.hstack((self.data['lat']['data'], lat)),                   units='degrees')
            self.data['ctp']      = dict(name='Cloud phase optical proprties',       data=np.hstack((self.data['ctp']['data'], ctp)),                   units='N/A')
            self.data['cth']      = dict(name='Cloud top height',                    data=np.hstack((self.data['cth']['data'], cth)),                   units='m')
            self.data['ctt']      = dict(name='Cloud top temperature',               data=np.hstack((self.data['ctt']['data'], ctt)),                   units='K')
            self.data['cot']      = dict(name='Cloud optical thickness',             data=np.hstack((self.data['cot']['data'], cot)),                   units='N/A')
            self.data['cer']      = dict(name='Cloud effective radius',              data=np.hstack((self.data['cer']['data'], cer)),                   units='micron')
            self.data['cwp']      = dict(name='Cloud water path',                    data=np.hstack((self.data['cwp']['data'], cwp)),                   units='g/m^2')
            self.data['cot_1621'] = dict(name='Cloud optical thickness (1621)',      data=np.hstack((self.data['cot_1621']['data'], cot_1621)),              units='N/A')
            self.data['cer_1621'] = dict(name='Cloud effective radius (1621)',       data=np.hstack((self.data['cer_1621']['data'], cer_1621)),              units='micron')
            self.data['cwp_1621'] = dict(name='Cloud water path (1621)',             data=np.hstack((self.data['cwp_1621']['data'], cwp_1621)),              units='g/m^2')
            self.data['cot_uct']  = dict(name='Cloud optical thickness uncertainty', data=np.hstack((self.data['cot_uct']['data'], cot*cot_uct/100.0)), units='N/A')
            self.data['cer_uct']  = dict(name='Cloud effective radius uncertainty',  data=np.hstack((self.data['cer_uct']['data'], cer*cer_uct/100.0)), units='micron')
            self.data['cwp_uct']  = dict(name='Cloud water path uncertainty',  data=np.hstack((self.data['cwp_uct']['data'], cwp*cwp_uct/100.0)), units='g/m^2')


        else:
            self.logic = {}
            self.logic[fname] = {'0.75km':logic_extent}
            self.data  = {}

            self.data['lon']      = dict(name='Longitude',                           data=lon,               units='degrees')
            self.data['lat']      = dict(name='Latitude',                            data=lat,               units='degrees')
            self.data['ctp']      = dict(name='Cloud phase optical properties',      data=ctp,               units='N/A')
            self.data['cth']      = dict(name='Cloud top height',                    data=cth,               units='m')
            self.data['cot']      = dict(name='Cloud optical thickness',             data=cot,               units='N/A')
            self.data['cer']      = dict(name='Cloud effective radius',              data=cer,               units='micron')
            self.data['cwp']      = dict(name='Cloud water path',                    data=cwp,               units='g/m^2')
            self.data['cot_1621'] = dict(name='Cloud optical thickness (1621)',      data=cot_1621,          units='N/A')
            self.data['cer_1621'] = dict(name='Cloud effective radius (1621)',       data=cer_1621,          units='micron')
            self.data['cwp_1621'] = dict(name='Cloud water path (1621)',             data=cwp_1621,          units='g/m^2')
            self.data['cot_uct']  = dict(name='Cloud optical thickness uncertainty', data=cot*cot_uct/100.0, units='N/A')
            self.data['cer_uct']  = dict(name='Cloud effective radius uncertainty',  data=cer*cer_uct/100.0, units='micron')
            self.data['cwp_uct']  = dict(name='Cloud water path uncertainty',        data=cwp*cwp_uct/100.0, units='g/m^2')



def grid_by_dxdy(lon, lat, data, extent=None, dx=None, dy=None, method='nearest', fill_value=0.0, Ngrid_limit=1):

    """
    Grid irregular data into a regular xy grid by input 'extent' (westmost, eastmost, southmost, northmost)
    Input:
        lon: numpy array, input longitude to be gridded
        lat: numpy array, input latitude to be gridded
        data: numpy array, input data to be gridded
        extent=: Python list, [westmost, eastmost, southmost, northmost]
        dx=: float, zonal spatial resolution in meter
        dy=: float, meridional spatial resolution in meter
    Output:
        lon_2d : numpy array, gridded longitude
        lat_2d : numpy array, gridded latitude
        data_2d: numpy array, gridded data
    How to use:
        After read in the longitude latitude and data into lon0, lat0, data0
        lon, lat, data = grid_by_dxdy(lon0, lat0, data0, dx=250.0, dy=250.0)
    """

    # flatten lon/lat/data
    #/----------------------------------------------------------------------------\#
    lon = np.array(lon).ravel()
    lat = np.array(lat).ravel()
    data = np.array(data).ravel()
    #\----------------------------------------------------------------------------/#


    # get extent
    #/----------------------------------------------------------------------------\#
    if extent is None:
        extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
    #\----------------------------------------------------------------------------/#


    # dist_x and dist_y
    #/----------------------------------------------------------------------------\#
    lon0 = [extent[0], extent[0]]
    lat0 = [extent[2], extent[3]]
    lon1 = [extent[1], extent[1]]
    lat1 = [extent[2], extent[3]]
    dist_x = cal_geodesic_dist(lon0, lat0, lon1, lat1).min()

    lon0 = [extent[0], extent[1]]
    lat0 = [extent[2], extent[2]]
    lon1 = [extent[0], extent[1]]
    lat1 = [extent[3], extent[3]]
    dist_y = cal_geodesic_dist(lon0, lat0, lon1, lat1).min()
    #\----------------------------------------------------------------------------/#


    # get Nx/Ny and dx/dy
    #/----------------------------------------------------------------------------\#
    if dx is None or dy is None:

        # Nx and Ny
        #/----------------------------------------------------------------------------\#
        xy = (extent[1]-extent[0])*(extent[3]-extent[2])
        N0 = np.sqrt(lon.size/xy)
        Nx = int(N0*(extent[1]-extent[0]))
        Ny = int(N0*(extent[3]-extent[2]))
        #\----------------------------------------------------------------------------/#

        # dx and dy
        #/----------------------------------------------------------------------------\#
        dx = dist_x / Nx
        dy = dist_y / Ny
        #\----------------------------------------------------------------------------/#

    else:

        Nx = int(dist_x // dx)
        Ny = int(dist_y // dy)
    #\----------------------------------------------------------------------------/#


    # get west-most lon_1d/lat_1d
    #/----------------------------------------------------------------------------\#
    lon_1d = np.repeat(extent[0], Ny)
    lat_1d = np.repeat(extent[2], Ny)
    for i in range(1, Ny):
        lon_1d[i], lat_1d[i] = cal_geodesic_lonlat(lon_1d[i-1], lat_1d[i-1], dy, 0.0)
    #\----------------------------------------------------------------------------/#


    # get lon_2d/lat_2d
    #/----------------------------------------------------------------------------\#
    lon_2d = np.zeros((Nx, Ny), dtype=np.float64)
    lat_2d = np.zeros((Nx, Ny), dtype=np.float64)
    lon_2d[0, :] = lon_1d
    lat_2d[0, :] = lat_1d
    for i in range(1, Nx):
        lon_2d[i, :], lat_2d[i, :] = cal_geodesic_lonlat(lon_2d[i-1, :], lat_2d[i-1, :], dx, 90.0)
    #\----------------------------------------------------------------------------/#


    # gridding
    #/----------------------------------------------------------------------------\#
    points   = np.transpose(np.vstack((lon, lat)))

    if method == 'nearest':
        data_2d = find_nearest(lon, lat, data, lon_2d, lat_2d, fill_value=np.nan, Ngrid_limit=Ngrid_limit)
    else:
        data_2d = interpolate.griddata(points, data, (lon_2d, lat_2d), method=method, fill_value=np.nan)

    logic = np.isnan(data_2d)
    data_2d[logic] = fill_value

    return lon_2d, lat_2d, data_2d


def find_nearest(x_raw, y_raw, data_raw, x_out, y_out, Ngrid_limit=1, fill_value=np.nan):

    """
    Use scipy.spatial.KDTree to perform fast nearest gridding

    References:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html

    Inputs:
        x_raw: x position of raw data
        y_raw: y position of raw data
        data_raw: value of raw data
        x_out: x position of the data (e.g., x of data to be gridded)
        y_out: y position of the data (e.g., y of data to be gridded)
        Ngrid_limit=<1>=: number of grids for defining "too far"
        fill_value=<np.nan>: fill-in value for the data that is "too far" away from raw data

    Output:
        data_out: gridded data
    """

    try:
        from scipy.spatial import KDTree
    except ImportError:
        msg = 'Error [find_nearest]: `scipy` installation is required.'
        raise ImportError(msg)

    # only support output at maximum dimension of 2
    #/----------------------------------------------------------------------------\#
    if x_out.ndim > 2:
        msg = '\nError [find_nearest]: Only supports <x_out.ndim<=2> and <y_out.ndim<=2>.'
        raise ValueError(msg)
    #\----------------------------------------------------------------------------/#


    # preprocess raw data
    #/----------------------------------------------------------------------------\#
    x = np.array(x_raw).ravel()
    y = np.array(y_raw).ravel()
    data = np.array(data_raw).ravel()

    logic_valid = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(data))
    x = x[logic_valid]
    y = y[logic_valid]
    data = data[logic_valid]
    #\----------------------------------------------------------------------------/#


    # create KDTree
    #/----------------------------------------------------------------------------\#
    points = np.transpose(np.vstack((x, y)))
    tree_xy = KDTree(points)
    #\----------------------------------------------------------------------------/#


    # search KDTree for the nearest neighbor
    #/----------------------------------------------------------------------------\#
    points_query = np.transpose(np.vstack((x_out.ravel(), y_out.ravel())))
    dist_xy, indices_xy = tree_xy.query(points_query, workers=-1)

    dist_out = dist_xy.reshape(x_out.shape)
    data_out = data[indices_xy].reshape(x_out.shape)
    #\----------------------------------------------------------------------------/#


    # use fill value to fill in grids that are "two far"* away from raw data
    #   * by default 1 grid away is defined as "too far"
    #/----------------------------------------------------------------------------\#
    if Ngrid_limit is None:

        logic_out = np.repeat(False, data_out.size).reshape(x_out.shape)

    else:

        dx = np.zeros_like(x_out, dtype=np.float64)
        dy = np.zeros_like(y_out, dtype=np.float64)

        dx[1:, ...] = x_out[1:, ...] - x_out[:-1, ...]
        dx[0, ...]  = dx[1, ...]

        dy[..., 1:] = y_out[..., 1:] - y_out[..., :-1]
        dy[..., 0]  = dy[..., 1]

        dist_limit = np.sqrt((dx*Ngrid_limit)**2+(dy*Ngrid_limit)**2)
        logic_out = (dist_out>dist_limit)

    logic_out = logic_out | (indices_xy.reshape(data_out.shape)==indices_xy.size)
    data_out[logic_out] = fill_value
    #\----------------------------------------------------------------------------/#

    return data_out



def cal_geodesic_dist(lon0, lat0, lon1, lat1):

    try:
        import cartopy.geodesic as cg
    except ImportError:
        msg = '\nError [cal_geodesic_dist]: Please install <cartopy> to proceed.'
        raise ImportError(msg)

    lon0 = np.array(lon0).ravel()
    lat0 = np.array(lat0).ravel()
    lon1 = np.array(lon1).ravel()
    lat1 = np.array(lat1).ravel()

    geo0 = cg.Geodesic()

    points0 = np.transpose(np.vstack((lon0, lat0)))

    points1 = np.transpose(np.vstack((lon1, lat1)))

    output = np.squeeze(np.asarray(geo0.inverse(points0, points1)))

    dist = output[..., 0]

    return dist


def cal_geodesic_lonlat(lon0, lat0, dist, azimuth):

    try:
        import cartopy.geodesic as cg
    except ImportError:
        msg = '\nError [cal_geodesic_lonlat]: Please install <cartopy> to proceed.'
        raise ImportError(msg)

    lon0 = np.array(lon0).ravel()
    lat0 = np.array(lat0).ravel()
    dist = np.array(dist).ravel()
    azimuth = np.array(azimuth).ravel()

    points = np.transpose(np.vstack((lon0, lat0)))

    geo0 = cg.Geodesic()

    output = np.squeeze(np.asarray(geo0.direct(points, azimuth, dist)))

    lon1 = output[..., 0]
    lat1 = output[..., 1]

    return lon1, lat1


def find_fname_match(fname0, fnames, index_s=1, index_e=3):

    filename0 = os.path.basename(fname0)
    pattern  = '.'.join(filename0.split('.')[index_s:index_e+1])

    fname_match = None
    for fname in fnames:
        if pattern in fname:
            fname_match = fname

    return fname_match


# VIIRS tools
#/---------------------------------------------------------------------------\

def get_fname_pattern(fname, index_s=1, index_e=2):

    filename = os.path.basename(fname)
    pattern  = '.'.join(filename.split('.')[index_s:index_e+1])

    return pattern

#\---------------------------------------------------------------------------/
