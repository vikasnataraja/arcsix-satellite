import os
import sys
import h5py
import datetime
import matplotlib
import warnings
import numpy as np
# from pyhdf.SD import SD, SDC

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from tqdm import tqdm

import cartopy.crs as ccrs

import cmasher as cmr

from argparse import ArgumentParser, RawTextHelpFormatter

import platform
import matplotlib.font_manager as font_manager
uname = platform.uname()

# Add every font at the specified location
if 'macbook' in uname.node.lower() and 'darwin' in uname.system.lower():
    font_dir = ['/Users/vikas/Downloads/EB_Garamond/static/']
    mpl_style = '/Users/vikas/.matplotlib/whiteseaborn.mplstyle'
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)
elif 'linux' in uname.system.lower():
    # Add every font at the specified location
    font_dir = ['/projects/viha4393/software/EB_Garamond/static/']
    mpl_style = '/projects/viha4393/software/matplotlib/whiteseaborn.mplstyle'
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)
else:
    mpl_style = 'ggplot'

# Set font family globally
plt.rc('font',**{'family':'serif','serif':['EB Garamond']})

SZA_LIMIT = 81.36

base_cmap = cmr.arctic
n_colors = 256
dark_colors = 130
dark = np.linspace(0.15, 0.6, dark_colors)
bright = np.linspace(0.6, 1.0, n_colors - dark_colors)
cmap_arr = np.hstack([dark, bright])
arctic_cmap = matplotlib.colors.ListedColormap(base_cmap(cmap_arr))

dark_colors = 0.1
medium_colors = 0.3
dark = np.linspace(0.0, 0.3, int(dark_colors*n_colors))
medium = np.linspace(0.3, 0.8, int(medium_colors*n_colors))
bright = np.linspace(0.8, 1.0, n_colors - int(dark_colors*n_colors) - int(medium_colors*n_colors))
cmap_arr = np.hstack([dark, medium, bright])
arctic_alt_cmap = matplotlib.colors.ListedColormap(base_cmap(cmap_arr))

n_colors = 256

# cmap_name = 'autumn'
# base_cmap = plt.get_cmap(cmap_name)
# dark_colors = 0.2
# medium_colors = 0.2
# dark = np.linspace(0.0, 0.2, int(dark_colors*n_colors))
# medium = np.linspace(0.2, 0.4, int(medium_colors*n_colors))
# bright = np.linspace(0.4, 1.0, n_colors - int(dark_colors*n_colors) - int(medium_colors*n_colors))
# cmap_arr = np.hstack([dark, medium, bright])
# newcolors = base_cmap(cmap_arr)
# newcolors[0, :] = np.array([203/n_colors, 24/n_colors, 50/n_colors, 1])
# newcolors[255, :] = np.array([1, 1, 1, 1])
# arctic_cloud_cmap = matplotlib.colors.ListedColormap(newcolors)
# arctic_cloud_cmap.set_bad(color='black')

# newcolors_alt = base_cmap(np.linspace(0, 1, n_colors))
# newcolors_alt[:2, :] = np.array([203/n_colors, 24/n_colors, 50/n_colors, 1])
# newcolors_alt[254:, :] = np.array([1, 1, 1, 1])
# arctic_cloud_alt_cmap = matplotlib.colors.ListedColormap(newcolors_alt)
# arctic_cloud_alt_cmap.set_bad(color='black')
arctic_cloud_cmap = 'RdBu_r'
arctic_cloud_alt_cmap = 'RdBu_r'


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def scale_255(arr):
    return ((arr - np.nanmin(arr)) * (1/(1e-6 + (np.nanmax(arr) - np.nanmin(arr))) * 255)).astype('uint8')

def scale_table(arr):
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


def doy_2_date_str(acq_dt):
    acq_dt = acq_dt[1:]
    date = datetime.datetime.strptime(acq_dt, '%Y%j.%H%M')
    return date.strftime('%Y-%m-%d-%H%M%SZ')

def format_acq_dt(acq_dt):
    """format acquisition datetime for filename """
    year, doy, hours, minutes = acq_dt[1:5], acq_dt[5:8], acq_dt[9:11], acq_dt[11:13]
    date = datetime.datetime.strptime(year+doy, '%Y%j').date()
    return date.strftime('%Y-%m-%d') + '-{}{}Z'.format(hours, minutes)


def normalize_data(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


def get_instrument(satellite):
    if (satellite == 'Aqua') or (satellite == 'Terra'):
        instrument = 'MODIS'
    elif (satellite == 'Suomi-NPP') or (satellite == 'NOAA-20/JPSS-1') or (satellite == 'NOAA-21/JPSS-2'):
        instrument = 'VIIRS'
    else:
        instrument = 'Unknown'

    return instrument



def mask_geojson(geojson_fpath, lon_2d, lat_2d, dat, proj_plot, proj_data):

    if geojson_fpath is not None:
        import json

    with open(geojson_fpath, 'r') as f:
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



param_list = ['ref', 'cot', 'cot_1621', 'cwp', 'cwp_1621', 'all']
wvl_list = [470, 555, 650, 860, 1640, 2130]


def read_multiple_h5(fdir, h5filename, test, time_range, ndir_recent, param=None, wvl=None, nan_pct=None):
    """ Reader for the processed and gridded .h5 files"""

    # if time_range is not None:
    #     # get current UTC time
    #     today_dt    = datetime.datetime.now(datetime.timezone.utc)
    #     today_dt    = today_dt.replace(tzinfo=None)
    #     oldest_dt   = today_dt - datetime.timedelta(hours=time_range)

    #     subs = []
    #     for f in os.listdir(fdir):
    #         if os.path.isdir(os.path.join(fdir, f)):
    #             start_dt, end_dt = f.split('_')
    #             start_dt = datetime.datetime.strptime(start_dt, '%Y-%m-%d-%H%M')
    #             end_dt   = datetime.datetime.strptime(end_dt, '%Y-%m-%d-%H%M')
    #             if (start_dt >= oldest_dt) or (end_dt <= today_dt):
    #                 subs.append(f)

    #     if len(subs) == 0:
    #         raise OSError("No HDF5 files found in {} for the desired time range between now and {} hours ago".format(fdir, time_range))
    #     else:
    #         subs = sorted(subs)

    # else:
        # subs = sorted([f for f in os.listdir(fdir) if os.path.isdir(os.path.join(fdir, f))]) # get sub-directories i.e., dates

    # subs = sorted([f for f in os.listdir(fdir) if os.path.isdir(os.path.join(fdir, f))]) # get sub-directories i.e., dates
    subs = []
    for f in os.listdir(args.fdir):
        if os.path.isdir(os.path.join(args.fdir, f)) and (len(os.listdir(os.path.join(args.fdir, f))) > 0):
            subs.append(f)

    subs = sorted(subs)

    if (ndir_recent is not None) and (len(subs) >= ndir_recent):
        subs = subs[-ndir_recent:]

    if len(subs) == 0:
        print("No training data found...exiting...\n")
        sys.exit()

    test_counter = 0
    overpasses = [] # use to maintain a record of already compiled data to remove duplicate processing
    lat, lon, sat_name = {}, {}, {}
    if (param in param_list) and (param == 'all'):

        dat_field = "ref_{}".format(str(wvl))
        ref, cot, cwp, cot_1621, cwp_1621, ctp = {}, {}, {}, {}, {}, {}

        for subdir in subs:

            h5file = os.path.join(fdir, subdir, h5filename)
            if not os.path.isfile(h5file):
                print("File {} does not exist but this error should have been caught earlier\n".format(h5file))
                continue

            with h5py.File(h5file, 'r') as f:

                keys = list(f.keys())

                for overpass in keys:
                    ref_no_norm        = np.array(f[overpass][dat_field])

                    if (nan_pct is not None) and (np.count_nonzero(np.isnan(ref_no_norm))*100/ref_no_norm.size > nan_pct):
                        continue

                    if overpass not in overpasses:
                        overpasses.append(overpass)
                    else:
                        print("Message [read_multiple_h5]: Skipping duplicate of {}".format(overpass))
                        continue

                    sza                = np.array(f[overpass]['sza_2d'])
                    sza[sza>SZA_LIMIT] = SZA_LIMIT
                    ref[overpass]      = ref_no_norm/np.cos(np.deg2rad(sza))
                    cot[overpass]      = np.array(f[overpass]['cot_2d'])
                    cwp[overpass]      = np.array(f[overpass]['cwp_2d'])
                    cot_1621[overpass] = np.array(f[overpass]['cot_1621'])
                    cwp_1621[overpass] = np.array(f[overpass]['cwp_1621'])
                    ctp[overpass]      = np.array(f[overpass]['ctp'])
                    lat[overpass]      = np.array(f[overpass]['lat_1d'])
                    lon[overpass]      = np.array(f[overpass]['lon_1d'])
                    # sat_name[overpass] = "hello"
                    sat_name[overpass] = str(np.char.decode(f[overpass]['satellite']))

                    test_counter += 1

                    if test is not None and test_counter == test:
                        return lat, lon, sat_name, ref, cot, cwp, cot_1621, cwp_1621, ctp

        return lat, lon, sat_name, ref, cot, cwp, cot_1621, cwp_1621, ctp

    # reflectance only
    elif (param in param_list) and ((param == 'ref') or (param is None)):
        if wvl not in wvl_list:
            raise KeyError("Data for wavelength {} does not exist. Try using one of [470, 555, 650, 1640, 2130]".format(wvl))

        dat_field = "ref_{}".format(str(wvl))
        ref = {}
        for subdir in subs:

            h5file = os.path.join(fdir, subdir, h5filename)
            if not os.path.isfile(h5file):
                print("File {} does not exist\n".format(h5file))
                continue

            with h5py.File(h5file, 'r') as f:

                keys = list(f.keys())

                for overpass in keys:
                    ref_no_norm        = np.array(f[overpass][dat_field])
                    if (nan_pct is not None) and (np.count_nonzero(np.isnan(ref_no_norm))*100/ref_no_norm.size > nan_pct):
                        continue

                    if overpass not in overpasses:
                        overpasses.append(overpass)
                    else:
                        print("Message [read_multiple_h5]: Skipping duplicate of {}".format(overpass))
                        continue

                    sza                = np.array(f[overpass]['sza_2d'])
                    sza[sza>SZA_LIMIT] = SZA_LIMIT
                    ref[overpass]      = ref_no_norm/np.cos(np.deg2rad(sza))
                    lat[overpass]      = np.array(f[overpass]['lat_1d'])
                    lon[overpass]      = np.array(f[overpass]['lon_1d'])

                    sat_name[overpass] = str(np.char.decode(f[overpass]['satellite']))

                    test_counter += 1

                    if test is not None and test_counter == test:
                        return lat, lon, sat_name, ref

        return lat, lon, sat_name, ref


    # cloud optical thickness (standard retrieval and 1621) only
    elif (param in param_list) and ((param == 'cot') or (param == 'cot_1621')):

        cot, cot_1621 = {}, {}
        for subdir in subs:

            h5file = os.path.join(fdir, subdir, h5filename)
            if not os.path.isfile(h5file):
                print("File {} does not exist\n".format(h5file))
                continue

            with h5py.File(h5file, 'r') as f:

                keys = list(f.keys())

                for overpass in keys:

                    cot[overpass]      = np.array(f[overpass]['cot_2d'])
                    if (nan_pct is not None) and (np.count_nonzero(np.isnan(cot[overpass]))*100/cot[overpass].size > nan_pct):
                        continue

                    if overpass not in overpasses:
                        overpasses.append(overpass)
                    else:
                        print("Message [read_multiple_h5]: Skipping duplicate of {}".format(overpass))
                        continue

                    cot_1621[overpass] = np.array(f[overpass]['cot_1621'])
                    lat[overpass]      = np.array(f[overpass]['lat_1d'])
                    lon[overpass]      = np.array(f[overpass]['lon_1d'])

                    sat_name[overpass] = str(np.char.decode(f[overpass]['satellite']))

                    test_counter += 1

                    if test is not None and test_counter == test:
                        return lat, lon, sat_name, cot, cot_1621

        return lat, lon, sat_name, cot, cot_1621


    # cloud water path (standard retrieval and 1621) only
    elif (param in param_list) and ((param == 'cwp') or (param == 'cwp_1621') or (param == 'ctp')):

        cwp, cwp_1621, ctp = {}, {}, {}
        for subdir in subs:

            h5file = os.path.join(fdir, subdir, h5filename)
            if not os.path.isfile(h5file):
                print("File {} does not exist\n".format(h5file))
                continue

            with h5py.File(h5file, 'r') as f:

                keys = list(f.keys())

                for overpass in keys:

                    cwp[overpass]      = np.array(f[overpass]['cwp_2d'])
                    if (nan_pct is not None) and (np.count_nonzero(np.isnan(cwp[overpass]))*100/cwp[overpass].size > nan_pct):
                        continue

                    if overpass not in overpasses:
                        overpasses.append(overpass)
                    else:
                        print("Message [read_multiple_h5]: Skipping duplicate of {}".format(overpass))
                        continue

                    cwp_1621[overpass] = np.array(f[overpass]['cwp_1621'])
                    ctp[overpass]      = np.array(f[overpass]['ctp'])
                    lat[overpass]      = np.array(f[overpass]['lat_1d'])
                    lon[overpass]      = np.array(f[overpass]['lon_1d'])

                    sat_name[overpass] = str(np.char.decode(f[overpass]['satellite']))

                    test_counter += 1

                    if test is not None and test_counter == test:
                        return lat, lon, sat_name, cwp, cwp_1621, ctp

        return lat, lon, sat_name, cwp, cwp_1621, ctp





def create_false_color_367_imagery(fdir, h5filename, outdir, time_range, ndir_recent, test, nan_pct, scale_factors=[1., 1.75, 2.]):
    ref = {}
    lats, lons, sat_name, ref['470'] = read_multiple_h5(fdir, h5filename, test=test, time_range=time_range, ndir_recent=ndir_recent, param='ref', wvl=470,  nan_pct=nan_pct)
    _, _, _, ref['1640']             = read_multiple_h5(fdir, h5filename, test=test, time_range=time_range, ndir_recent=ndir_recent,param='ref', wvl=1640, nan_pct=nan_pct)
    _, _, _, ref['2130']             = read_multiple_h5(fdir, h5filename, test=test, time_range=time_range, ndir_recent=ndir_recent,param='ref', wvl=2130, nan_pct=nan_pct)

    acq_dts = list(ref['470'].keys())
    if len(acq_dts) == 0:
        print("Could not find any HDF5 files.\n")
        return 0

    acq_dts = sorted(acq_dts, key = lambda x: x[1:])

    lon_1d = lons[acq_dts[0]]
    lat_1d = lats[acq_dts[0]]
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    extent = [np.float16(lon_1d.min()), np.float16(lon_1d.max()), np.float16(lat_1d.min()), np.float16(lat_1d.max())]

    proj_plot = ccrs.PlateCarree()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_dir = outdir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    counter = 0
    for acq_dt in acq_dts:

        satellite = sat_name[acq_dt]
        if satellite == 'Aqua': # band 6 is broken on Aqua so skip
            print('\nSkipping {} as 3-6-7 imagery cannot be created due to issues with Aqua MODIS Band 6\n'.format(acq_dt))
            continue

        dt_str = doy_2_date_str(acq_dt)
        instrument = get_instrument(satellite)
        sat_fname    = satellite.split('/')[0]
        fname_target = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f}).png".format(instrument.upper(), sat_fname.upper(), "FalseColor367", dt_str, *extent)
        full_fname = os.path.join(save_dir, fname_target)
        if os.path.isfile(full_fname):
            print("Message [satellite_quicklook]: {} skipped since it already exists.".format(full_fname))
            continue

        blue    = np.clip(normalize_data(ref['470'][acq_dt])  * scale_factors[0], 0, 1)
        swir_16 = np.clip(normalize_data(ref['1640'][acq_dt]) * scale_factors[1], 0, 1)
        swir_21 = np.clip(normalize_data(ref['2130'][acq_dt]) * scale_factors[2], 0, 1)
        img_fci = np.stack([blue, swir_16, swir_21], axis=-1)


        fig = plt.figure(figsize=(12, 12))
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, img_fci,
                      shading='nearest',
                      zorder=2,
                      vmin=0., vmax=1.,
                      transform=proj_data)

        ax.set_aspect("auto")
        ax.set_extent(extent, proj_data)
        ax.set_xticks([])
        ax.set_yticks([])

        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig(full_fname, dpi=300, pad_inches=0.0, bbox_inches="tight",  metadata=_metadata)
        plt.close()
        counter += 1
        # break
    return 1


def create_false_color_721_imagery(fdir, h5filename, outdir, time_range, ndir_recent, test, nan_pct, scale_factors=[1.75, 1., 1.5]):
    ref = {}
    lats, lons, sat_name, ref['2130'] = read_multiple_h5(fdir, h5filename, test=test, time_range=time_range, ndir_recent=ndir_recent,param='ref', wvl=2130, nan_pct=nan_pct)
    _, _, _, ref['860']               = read_multiple_h5(fdir, h5filename, test=test, time_range=time_range, ndir_recent=ndir_recent,param='ref', wvl=860,  nan_pct=nan_pct)
    _, _, _, ref['650']               = read_multiple_h5(fdir, h5filename, test=test, time_range=time_range, ndir_recent=ndir_recent,param='ref', wvl=650,  nan_pct=nan_pct)


    acq_dts = list(ref['2130'].keys())
    if len(acq_dts) == 0:
        print("Could not find any HDF5 files.\n")
        return 0

    acq_dts = sorted(acq_dts, key = lambda x: x[1:])

    lon_1d = lons[acq_dts[0]]
    lat_1d = lats[acq_dts[0]]
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    extent = [np.float16(lon_1d.min()), np.float16(lon_1d.max()), np.float16(lat_1d.min()), np.float16(lat_1d.max())]

    proj_plot = ccrs.PlateCarree()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_dir = outdir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    counter = 0
    for acq_dt in acq_dts:

        dt_str = doy_2_date_str(acq_dt)
        satellite = sat_name[acq_dt]
        instrument = get_instrument(satellite)

        sat_fname    = satellite.split('/')[0]
        fname_target = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f}).png".format(instrument.upper(), sat_fname.upper(), "FalseColor721", dt_str, *extent)
        full_fname = os.path.join(save_dir, fname_target)
        if os.path.isfile(full_fname):
            print("Message [satellite_quicklook]: {} skipped since it already exists.".format(full_fname))
            continue

        swir_21 = np.clip(normalize_data(ref['2130'][acq_dt]) * scale_factors[0], 0, 1)
        nir     = np.clip(normalize_data(ref['860'][acq_dt]) * scale_factors[1], 0, 1)
        red     = np.clip(normalize_data(ref['650'][acq_dt]) * scale_factors[2], 0, 1)

        img_fci = np.stack([swir_21, nir, red], axis=-1)

        fig = plt.figure(figsize=(12, 12))
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, img_fci,
                      shading='nearest',
                      zorder=1,
                      vmin=0., vmax=1.,
                      transform=proj_data)
        ax.set_aspect("auto")
        ax.set_extent(extent, proj_data)
        ax.set_xticks([])
        ax.set_yticks([])

        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig(full_fname, dpi=300, pad_inches=0.0, bbox_inches="tight",  metadata=_metadata)
        plt.close()
        counter += 1
        # break
    return 1


def create_true_color_imagery(fdir, h5filename, outdir, time_range, ndir_recent, test, nan_pct):

    ref = {}
    lats, lons, sat_name, ref['650']  = read_multiple_h5(fdir, h5filename, test=test, time_range=time_range, ndir_recent=ndir_recent,param='ref', wvl=650,  nan_pct=nan_pct)
    _, _, _, ref['555']               = read_multiple_h5(fdir, h5filename, test=test, time_range=time_range, ndir_recent=ndir_recent,param='ref', wvl=555,  nan_pct=nan_pct)
    _, _, _, ref['470']               = read_multiple_h5(fdir, h5filename, test=test, time_range=time_range, ndir_recent=ndir_recent,param='ref', wvl=470,  nan_pct=nan_pct)


    acq_dts = list(ref['650'].keys())
    if len(acq_dts) == 0:
        print("Could not find any HDF5 files.\n")
        return 0

    acq_dts = sorted(acq_dts, key = lambda x: x[1:])

    lon_1d = lons[acq_dts[0]]
    lat_1d = lats[acq_dts[0]]
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    extent = [np.float16(lon_1d.min()), np.float16(lon_1d.max()), np.float16(lat_1d.min()), np.float16(lat_1d.max())]

    proj_plot = ccrs.PlateCarree()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_dir = outdir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    counter = 0
    for acq_dt in acq_dts:

        dt_str = doy_2_date_str(acq_dt)
        satellite = sat_name[acq_dt]
        instrument = get_instrument(satellite)

        sat_fname    = satellite.split('/')[0]
        fname_target = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f}).png".format(instrument.upper(), sat_fname.upper(), "TrueColor", dt_str, *extent)
        full_fname = os.path.join(save_dir, fname_target)
        if os.path.isfile(full_fname):
            print("Message [satellite_quicklook]: {} skipped since it already exists.".format(full_fname))
            continue

        red   = np.clip(ref['650'][acq_dt], 0, 1.1)
        red   = scale_255(red)

        green = np.clip(ref['555'][acq_dt], 0, 1.1)
        green = scale_255(green)

        blue  = np.clip(ref['470'][acq_dt], 0, 1.1)
        blue  = scale_255(blue)

        rgb   = np.stack([red, green, blue], axis=-1)
        # so that outliers don't offset colors too much
        rgb   = np.interp(rgb, (np.percentile(rgb, 1), np.percentile(rgb, 99)), (0, 1))

        fig = plt.figure(figsize=(12, 12))
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, rgb,
                      shading='nearest',
                      zorder=2,
                      vmin=0., vmax=1.,
                      transform=proj_data)

        ax.set_aspect("auto")
        ax.set_extent(extent, proj_data)
        ax.set_xticks([])
        ax.set_yticks([])

        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig(full_fname, dpi=300, pad_inches=0.0, bbox_inches="tight",  metadata=_metadata)
        plt.close()
        counter += 1
        # break
    return 0



def plot_liquid_water_paths(fdir, h5filename, outdir, time_range, ndir_recent, test, nan_pct, vmax):

    #     " The values in this SDS are set to mean the following:                              \n",
    #     " 0 -- cloud mask undetermined                                                       \n",
    #     " 1 -- clear sky                                                                     \n",
    #     " 2 -- liquid water cloud                                                            \n",
    #     " 3 -- ice cloud                                                                     \n",
    #     " 4 -- undetermined phase cloud (but retrieval is attempted as  liquid water)        \n",

    lats, lons, sat_name, cwp, cwp_1621, ctp = read_multiple_h5(fdir, h5filename, test=test, time_range=time_range, ndir_recent=ndir_recent,param='cwp', nan_pct=nan_pct)

    acq_dts = list(sat_name.keys())
    if len(acq_dts) == 0:
        print("Could not find any HDF5 files.\n")
        return 0

    acq_dts = sorted(acq_dts, key = lambda x: x[1:])

    lon_1d = lons[acq_dts[0]]
    lat_1d = lats[acq_dts[0]]
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    extent = [np.float16(lon_1d.min()), np.float16(lon_1d.max()), np.float16(lat_1d.min()), np.float16(lat_1d.max())]

    proj_plot = ccrs.PlateCarree()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_dir = outdir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    counter = 0
    for acq_dt in acq_dts:

        satellite = sat_name[acq_dt]
        dt_str  = doy_2_date_str(acq_dt)
        instrument = get_instrument(satellite)
        sat_fname    = satellite.split('/')[0]
        fname_target = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f}).png".format(instrument.upper(), sat_fname.upper(), "LWP", dt_str, *extent)
        full_fname = os.path.join(save_dir, fname_target)

        fname_target_1621 = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f}).png".format(instrument.upper(), sat_fname.upper(), "LWP1621", dt_str, *extent)
        full_fname_1621 = os.path.join(save_dir, fname_target_1621)

        if os.path.isfile(full_fname) or os.path.isfile(full_fname_1621):
            print("Message [satellite_quicklook]: {} skipped since it already exists.".format(full_fname))
            continue

        pha            = ctp[acq_dt]
        liquid_logic   = np.where((pha == 2) | (pha == 4)) # liquid water
        ice_logic      = np.where((pha == 3))
        # undt_logic     = np.where((pha == 0))
        clear_logic    = np.where((pha == 1))

        # standard retrieval liquid water
        cloud_wp       = cwp[acq_dt]
        im_lwp         = np.empty(cloud_wp.shape)
        im_lwp[:]      = np.nan
        im_lwp[liquid_logic] = cloud_wp[liquid_logic]
        im_lwp[clear_logic]  = 0.
        im_lwp[ice_logic]    = 0.

        # 1621 retrieval liquid water
        cloud_wp_1621  = cwp_1621[acq_dt]
        im_lwp_1621    = np.empty(cloud_wp_1621.shape)
        im_lwp_1621[:] = np.nan
        im_lwp_1621[liquid_logic] = cloud_wp_1621[liquid_logic]
        im_lwp_1621[clear_logic]  = 0.
        im_lwp_1621[ice_logic]    = 0.


        if vmax is not None:
            im_lwp[im_lwp > vmax]           = vmax
            im_lwp_1621[im_lwp_1621 > vmax] = vmax
            cmap = arctic_cloud_cmap
        else:
            cmap = arctic_cloud_alt_cmap

        ##############################################################
        fig = plt.figure(figsize=(12, 12))
        gs  = GridSpec(1, 1, figure=fig)
        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_lwp,
                            shading='nearest',
                            zorder=1,
                            cmap=cmap,
                            transform=proj_data)

        ax00.set_extent(extent, proj_data)
        ax00.set_aspect("auto")
        ax00.set_xticks([])
        ax00.set_yticks([])

        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig(full_fname, dpi=300, pad_inches=0.0, bbox_inches="tight",  metadata=_metadata)
        plt.close()

        ##############################################################

        fig = plt.figure(figsize=(12, 12))
        gs  = GridSpec(1, 1, figure=fig)
        ax01 = fig.add_subplot(gs[0], projection=proj_plot)
        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_lwp_1621,
                      shading='nearest',
                      zorder=1,
                      cmap=cmap,
                      transform=proj_data)

        ax01.set_extent(extent, proj_data)
        ax01.set_aspect("auto")
        ax01.set_xticks([])
        ax01.set_yticks([])
        sat_fname    = satellite.split('/')[0]
        fname_target = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f})".format(instrument.upper(), sat_fname.upper(), "LWP1621", dt_str, *extent)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig(os.path.join("{}/{}.png".format(save_dir, fname_target)), dpi=300, pad_inches=0.0, bbox_inches="tight",  metadata=_metadata)
        plt.close()
        ##############################################################

        counter += 1
        # break

    return 1



def plot_ice_water_paths(fdir, h5filename, outdir, time_range, ndir_recent, test, nan_pct, vmax):

    #     " The values in this SDS are set to mean the following:                              \n",
    #     " 0 -- cloud mask undetermined                                                       \n",
    #     " 1 -- clear sky                                                                     \n",
    #     " 2 -- liquid water cloud                                                            \n",
    #     " 3 -- ice cloud                                                                     \n",
    #     " 4 -- undetermined phase cloud (but retrieval is attempted as  liquid water)        \n",

    lats, lons, sat_name, cwp, cwp_1621, ctp = read_multiple_h5(fdir, h5filename, test=test, time_range=time_range, ndir_recent=ndir_recent, param='cwp', nan_pct=nan_pct)


    acq_dts = list(sat_name.keys())
    if len(acq_dts) == 0:
        print("Could not find any HDF5 files.\n")
        return 0

    acq_dts = sorted(acq_dts, key = lambda x: x[1:])

    lon_1d = lons[acq_dts[0]]
    lat_1d = lats[acq_dts[0]]
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    extent = [np.float16(lon_1d.min()), np.float16(lon_1d.max()), np.float16(lat_1d.min()), np.float16(lat_1d.max())]

    proj_plot = ccrs.PlateCarree()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_dir = outdir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Found {} overpasses from {} sub-directories".format(len(acq_dts), len(os.listdir(fdir))))

    counter = 0
    for acq_dt in acq_dts:

        satellite = sat_name[acq_dt]
        dt_str  = doy_2_date_str(acq_dt)
        instrument = get_instrument(satellite)
        sat_fname    = satellite.split('/')[0]
        fname_target = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f}).png".format(instrument.upper(), sat_fname.upper(), "IWP", dt_str, *extent)
        full_fname = os.path.join(save_dir, fname_target)

        fname_target_1621 = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f}).png".format(instrument.upper(), sat_fname.upper(), "IWP1621", dt_str, *extent)
        full_fname_1621 = os.path.join(save_dir, fname_target_1621)

        if os.path.isfile(full_fname) or os.path.isfile(full_fname_1621):
            print("Message [satellite_quicklook]: {} skipped since it already exists.".format(full_fname))
            continue

        pha            = ctp[acq_dt]
        liquid_logic   = np.where((pha == 2) | (pha == 4)) # liquid water
        ice_logic      = np.where((pha == 3))
        # undt_logic     = np.where((pha == 0))
        clear_logic    = np.where((pha == 1))

        # standard retrieval liquid water
        cloud_wp       = cwp[acq_dt]

        # standard retrieval ice logic
        im_iwp         = np.empty(cloud_wp.shape)
        im_iwp[:]      = np.nan
        im_iwp[ice_logic] = cloud_wp[ice_logic]
        im_iwp[clear_logic]  = 0.
        im_iwp[liquid_logic] = 0.


        cloud_wp_1621  = cwp_1621[acq_dt]

        # 1621 retrieval ice logic
        im_iwp_1621         = np.empty(cloud_wp_1621.shape)
        im_iwp_1621[:]      = np.nan
        im_iwp_1621[ice_logic] = cloud_wp_1621[ice_logic]
        im_iwp_1621[clear_logic]  = 0.
        im_iwp_1621[liquid_logic] = 0.

        if vmax is not None:
            im_iwp[im_iwp > vmax]           = vmax
            im_iwp_1621[im_iwp_1621 > vmax] = vmax
            cmap = arctic_cloud_cmap

        else:
            cmap = arctic_cloud_alt_cmap

        ##############################################################

        fig = plt.figure(figsize=(12, 12))
        gs  = GridSpec(1, 1, figure=fig)
        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_iwp,
                      shading='nearest',
                      zorder=1,
                      cmap=cmap,
                      transform=proj_data)

        ax00.set_extent(extent, proj_data)
        ax00.set_aspect("auto")
        ax00.set_xticks([])
        ax00.set_yticks([])
        # sat_fname    = satellite.split('/')[0]
        # fname_target = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f})".format(instrument.upper(), sat_fname.upper(), "IWP", dt_str, *extent)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig(full_fname, dpi=300, pad_inches=0.0, bbox_inches="tight",  metadata=_metadata)
        plt.close()

        ##############################################################
        fig = plt.figure(figsize=(12, 12))
        gs  = GridSpec(1, 1, figure=fig)
        ax01 = fig.add_subplot(gs[0], projection=proj_plot)
        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_iwp_1621,
                      shading='nearest',
                      zorder=1,
                      cmap=cmap,
                      transform=proj_data)
        ax01.set_extent(extent, proj_data)
        ax01.set_aspect("auto")
        ax01.set_xticks([])
        ax01.set_yticks([])

        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig(full_fname_1621, dpi=300, pad_inches=0.0, bbox_inches="tight",  metadata=_metadata)
        plt.close()

        ##############################################################

        counter += 1
        # break

    return 1


def plot_optical_depths(fdir, h5filename, outdir, time_range, ndir_recent, test, nan_pct, vmax):

    lats, lons, sat_name, cot, cot_1621  = read_multiple_h5(fdir, h5filename, param='cot', test=test, time_range=time_range, ndir_recent=ndir_recent, nan_pct=nan_pct)

    acq_dts = list(sat_name.keys())
    if len(acq_dts) == 0:
        print("Could not find any HDF5 files.\n")
        return 0

    acq_dts = sorted(acq_dts, key = lambda x: x[1:])

    lon_1d = lons[acq_dts[0]]
    lat_1d = lats[acq_dts[0]]
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    extent = [np.float16(lon_1d.min()), np.float16(lon_1d.max()), np.float16(lat_1d.min()), np.float16(lat_1d.max())]

    proj_plot = ccrs.PlateCarree()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_dir = outdir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    vmax = 100.
    counter = 0
    for acq_dt in acq_dts:

        satellite = sat_name[acq_dt]
        dt_str  = doy_2_date_str(acq_dt)
        instrument = get_instrument(satellite)
        sat_fname    = satellite.split('/')[0]
        fname_target = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f}).png".format(instrument.upper(), sat_fname.upper(), "COT", dt_str, *extent)
        full_fname = os.path.join(save_dir, fname_target)

        fname_target_1621 = "{}-{}_{}_{}_({:.2f},{:.2f},{:.2f},{:.2f}).png".format(instrument.upper(), sat_fname.upper(), "COT1621", dt_str, *extent)
        full_fname_1621 = os.path.join(save_dir, fname_target_1621)

        if os.path.isfile(full_fname) or os.path.isfile(full_fname_1621):
            print("Message [satellite_quicklook]: {} skipped since it already exists.".format(full_fname))
            continue


        im_cot      = cot[acq_dt]
        im_cot_1621 = cot_1621[acq_dt]

        if vmax is not None:
            im_cot[im_cot > vmax]           = vmax
            im_cot_1621[im_cot_1621 > vmax] = vmax
            cmap = arctic_cloud_cmap
        else:
            cmap = arctic_cloud_alt_cmap

        fig = plt.figure(figsize=(12, 12))
        gs  = GridSpec(1, 1, figure=fig)
        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_cot,
                            shading='nearest',
                            zorder=1,
                            cmap=cmap,
                            transform=proj_data)

        ax00.set_extent(extent, proj_data)
        ax00.set_aspect("auto")
        ax00.set_xticks([])
        ax00.set_yticks([])

        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig(full_fname, dpi=300, pad_inches=0.0, bbox_inches="tight",  metadata=_metadata)
        plt.close()

        ##############################################################

        fig = plt.figure(figsize=(12, 12))
        gs  = GridSpec(1, 1, figure=fig)
        ax01 = fig.add_subplot(gs[0], projection=proj_plot)
        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_cot_1621,
                      shading='nearest',
                      zorder=1,
                      cmap=cmap,
                      transform=proj_data)

        ax01.set_extent(extent, proj_data)
        ax01.set_aspect("auto")
        ax01.set_xticks([])
        ax01.set_yticks([])

        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig(full_fname_1621, dpi=300, pad_inches=0.0, bbox_inches="tight",  metadata=_metadata)
        plt.close()

        counter += 1
        ##############################################################

    return 1


def create_video(fdir, outdir, frame_rate):

    subs = sorted([f for f in os.listdir(fdir) if os.path.isdir(os.path.join(fdir, f))])

    commands = []
    for sub in subs:

        commands.append("ffmpeg -r {} -start_number 0 -i {}/%d.png -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p {}.mp4".format(frame_rate, os.path.join(fdir, sub), outdir))

    for command in commands:
        os.system(command)

    return 0


proj_data = ccrs.PlateCarree()
cfs_alert = (-62.3167, 82.5) # Station Alert
stn_nord  = (-16.6667, 81.6) # Station Nord
thule_pituffik = (-68.703056, 76.531111) # Pituffik Space Base

if __name__ == "__main__":
    parser = ArgumentParser(prog='satellite_quicklook', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--fdir', type=str, metavar='', default='sat-data/',
                        help='Directory where the files have been downloaded\n'\
                        'By default, files are assumed to be in \'sat-data/\'\n \n')
    parser.add_argument('--outdir', type=str, metavar='', help='Directory where files need to be written')
    parser.add_argument('--h5filename', type=str, metavar='', help='HDF5 filename that is common')
    parser.add_argument('--test', type=int, default=None, metavar='', help='Number of test overpasses')
    parser.add_argument('--geojson', type=str, metavar='',
                        help='Path to a geoJSON file containing the extent of interest coordinates\n'\
                        'Example:  --geojson my/path/to/geofile.json\n \n')
    parser.add_argument('--nan_pct', type=float, default=None, metavar='',
                        help='Percentage of NaNs to tolerate. If overpass has > nan_pct, it will be discarded')
    parser.add_argument('--vmax', type=float, default=None, metavar='',
                        help='Maximum value for capping during plotting (only for COT and CWP)')
    parser.add_argument('--param', nargs='+', type=str, metavar='',
                        help='Parameter to plot. One or more of ["fci_367", "fci_721", "tci", "cwp", "cot"]\n')
    parser.add_argument('--mode', type=str, metavar='', default=None, help='One of "normal", "lincoln_sea", "platypus" ')
    parser.add_argument('--range', type=int, metavar='', default=None, help='Time range. For example, if --range 5, then images will be generated between now and 5 hours ago.')
    parser.add_argument('--ndir_recent', type=int, metavar='', default=None, help='Time range in terms of sub-directories. For example, if --ndir_recent 5, then the most recent (based on filenames) 5 sub-directories will be used to generate images')
    parser.add_argument('--buoys', type=str, metavar='', default=None, help='Path to the JSON file containing URLs to the buoys csv data')
    args = parser.parse_args()

    # lon_range = [-100-2, 0+2]
    # lat_range = [75-0.5, 89+0.5]

    if (args.mode is None) or (args.mode.lower() == 'normal') or (args.mode.lower() == 'standard') or ('wide' in args.mode.lower()):
        VIEW_EXTENT = [-130, 50, 76, 89]
        VLON, VLAT = -40, 84

    elif (args.mode.lower() == 'lincoln') or (args.mode.lower() == 'lincoln_sea'):
        # VIEW_EXTENT = [-69.5, 12.4, 79.8, 85.2]
        # VIEW_EXTENT = [-120, 35, 78, 88.8]
        VIEW_EXTENT = [-100, 15, 79, 89.5]
        VLON, VLAT = -40, 84

    elif (args.mode.lower() == 'platypus') or (args.mode.lower() == 'canada') or (args.mode.lower() == 'ca_archipelago'):
        # VIEW_EXTENT =  [-100.5, -85, 76.4, 87.2]
        # VIEW_EXTENT = [-152, -23, 76.5, 88.3]
        VIEW_EXTENT = [-140, -30, 75.5, 89.5]
        VLON, VLAT = -70, 84

    # vlon, vlat = np.mean(lon_range), np.mean(lat_range)
    # proj_plot = ccrs.NorthPolarStereo(central_longitude=vlon)
    return_val = 0
    for param in tqdm(args.param):
        if param == "fci_367":
            return_val += create_false_color_367_imagery(fdir=args.fdir, h5filename=args.h5filename, outdir=args.outdir, test=args.test, nan_pct=args.nan_pct, time_range=args.range, ndir_recent=args.ndir_recent)

        if param == "fci_721":
            return_val += create_false_color_721_imagery(fdir=args.fdir, h5filename=args.h5filename, outdir=args.outdir, test=args.test, nan_pct=args.nan_pct, time_range=args.range, ndir_recent=args.ndir_recent)

        if param == "tci":
            return_val += create_true_color_imagery(fdir=args.fdir, h5filename=args.h5filename, outdir=args.outdir, test=args.test, nan_pct=args.nan_pct, time_range=args.range, ndir_recent=args.ndir_recent)

        if param == "cwp":
            return_val += plot_liquid_water_paths(fdir=args.fdir, h5filename=args.h5filename, outdir=args.outdir, test=args.test, nan_pct=args.nan_pct, time_range=args.range, ndir_recent=args.ndir_recent, vmax=args.vmax)
            return_val += plot_ice_water_paths(fdir=args.fdir, h5filename=args.h5filename, outdir=args.outdir, test=args.test, nan_pct=args.nan_pct, time_range=args.range, ndir_recent=args.ndir_recent, vmax=args.vmax)

        if param == "cot":
            return_val += plot_optical_depths(fdir=args.fdir, h5filename=args.h5filename, outdir=args.outdir, test=args.test, nan_pct=args.nan_pct, time_range=args.range, ndir_recent=args.ndir_recent, vmax=args.vmax)

    print("Message [satellite_quicklook] Created png files for {} types of imagery.\n".format(return_val))
