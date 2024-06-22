import os
import sys
import h5py
import json
import time
import requests
import datetime
import matplotlib
import cartopy
import warnings
import numpy as np
# from pyhdf.SD import SD, SDC

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from tqdm import tqdm

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

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
    year, doy, hours, minutes = acq_dt[1:5], acq_dt[5:8], acq_dt[9:11], acq_dt[11:13]
    date = datetime.datetime.strptime('{} {} {} {}'.format(year, doy, hours, minutes), '%Y %j %H %M')
    return date.strftime('%B %d, %Y: %H%M')

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



def get_buoy_data(json_fpath, API_KEY="kU7vw3YBcIvnxqTH8DlDeR08QTTfNYiZ", days=7):

    import pandas as pd
    fields_str = '?field=time_stamp&field=latitude&field=longitude'
    # for field in fields:
    #     fields_str += 'field={}&'.format(field)

    with open(json_fpath, 'r') as fp:
        urls = json.load(fp)

    dts, lons, lats = {}, {}, {}
    for bname, burl in urls.items():
        url = burl + fields_str
        data = requests.get(url, headers={'Authorization':'Bearer {}'.format(API_KEY)}).json()

        df = pd.DataFrame(data)
        df['time_stamp'] = df['time_stamp'].apply(lambda x: datetime.datetime.fromtimestamp(time.mktime(time.gmtime(x))))
        end_dt = df['time_stamp'].iloc[-1]
        start_dt = df['time_stamp'].iloc[-1] - datetime.timedelta(days=days)
        time_logic = (df['time_stamp'] >= start_dt) & (df['time_stamp'] <= end_dt)
        df = df[time_logic].reset_index(drop=True)

        # drop rows that have erroneous or irrelevant information
        # df = df.drop(df[df.longitude < 70].index)
        df = df.drop(df[df.latitude < 70].index)
        df = df.dropna(subset=['time_stamp', 'longitude', 'latitude'])
        df = df.reset_index(drop=True)

        # df['buoy_id'] = bname

        dts[bname]  = list(df['time_stamp'])
        lons[bname] = list(df['longitude'])
        lats[bname] = list(df['latitude'])

    return dts, lons, lats


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




def add_ancillary(ax, buoys, title=None):
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
        ax.set_title(title, pad=10, fontsize=22, fontweight="bold")

    ax.add_feature(cartopy.feature.OCEAN.with_scale('50m'), zorder=0, facecolor='black', edgecolor='none')
    ax.add_feature(cartopy.feature.LAND.with_scale('50m'), zorder=0, facecolor='black', edgecolor='none')
    ax.add_feature(cartopy.feature.COASTLINE.with_scale('50m'), zorder=2, edgecolor='darkgray', linewidth=1, alpha=1)

    # ax.set_aspect("auto")

    gl = ax.gridlines(linewidth=1.5, color='darkgray',
                draw_labels=True, zorder=2, alpha=1, linestyle=(0, (1, 1)),
                x_inline=False, y_inline=True, crs=proj_data)

    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 20))
    gl.ylocator = mticker.FixedLocator(np.arange(0, 90, 5))
    gl.xlabel_style = {'size': 12, 'color': 'black'}
    gl.ylabel_style = {'size': 12, 'color': 'white'}
    gl.rotate_labels = False
    gl.top_labels    = False
    gl.xpadding = 5
    gl.ypadding = 5
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    if buoys is not None:
        dt, lons, lats = get_buoy_data(buoys)

        # dt_text_box = ''
        buoy_ids = list(dt.keys())
        colors = plt.cm.brg(np.linspace(0, 1, len(buoy_ids)))
        for i, bid in enumerate(buoy_ids):
            ax.scatter(lons[bid][0], lats[bid][0], marker='s', s=30, edgecolor='black', facecolor=colors[i], transform=proj_data, zorder=2, alpha=0.7)
            ax.plot(lons[bid], lats[bid], linewidth=1.5, color=colors[i], transform=proj_data, zorder=2, alpha=0.7)
            ax.scatter(lons[bid][-1], lats[bid][-1], marker='*', s=100, edgecolor='black', facecolor=colors[i], transform=proj_data, zorder=2, alpha=1)
            # text = "{}\n{}".format(bid, dt[bid][-1].strftime("%Y-%m-%d: %H%MZ"))
            text = str(bid)
            ax.text(lons[bid][-1] + 2, lats[bid][-1] - 0.2, text, ha="center", va="center", transform=proj_data, color=colors[i],
                    fontsize=10, fontweight="bold", zorder=2)

        #     dt_text_box += str(bid) + ":" + dt[bid][-1].strftime("%B %d, %Y: %H%MZ")
        #     if i < len(buoy_ids) - 1:
        #         dt_text_box += ",  "

        # ax.text(0.5, 0.02, dt_text_box, color="black", ha="center", va="center", transform=ax.transAxes, fontsize=10,
        #         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.9))




def create_false_color_367_imagery(fdir, h5filename, outdir, geojson_fpath, time_range, ndir_recent, test, nan_pct, buoys, scale_factors=[1., 1.75, 2.]):
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
    # lon_range = [-100-2, 0+2]
    # lat_range = [75-0.5, 89+0.5]
    vlon, vlat = np.mean([lon_1d.min(), lon_1d.max()]), np.mean([lat_1d.min(), lat_1d.max()])
    # proj_plot = ccrs.NorthPolarStereo(central_longitude=vlon)
    proj_plot = ccrs.Orthographic(central_longitude=VLON, central_latitude=VLAT)

    # semicircle boundary
    # lonmin, lonmax, latmin, latmax = -140, 40, 73, 90
    lonmin, lonmax, latmin, latmax = lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()
    vertices = [(lon, int(latmin)) for lon in range(int(lonmin), int(lonmax) + 1, 1)] + \
    [(lon, int(latmax)) for lon in range(int(lonmax), int(lonmin) - 1, -1)]
    boundary = matplotlib.path.Path(vertices)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_dir = os.path.join(outdir, "false_color_367")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    print("Found {} overpasses from {} sub-directories".format(len(acq_dts), len(os.listdir(fdir))))

    counter = 0
    for acq_dt in acq_dts:

        satellite = sat_name[acq_dt]
        if satellite == 'Aqua': # band 6 is broken on Aqua so skip
            print('\nSkipping {} as 3-6-7 imagery cannot be created due to issues with Aqua MODIS Band 6\n'.format(acq_dt))
            continue

        fname_target = format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
            continue

        dt_title = doy_2_date_str(acq_dt) + "Z"
        instrument = get_instrument(satellite)

        blue    = np.clip(normalize_data(ref['470'][acq_dt])  * scale_factors[0], 0, 1)
        swir_16 = np.clip(normalize_data(ref['1640'][acq_dt]) * scale_factors[1], 0, 1)
        swir_21 = np.clip(normalize_data(ref['2130'][acq_dt]) * scale_factors[2], 0, 1)
        img_fci = np.stack([blue, swir_16, swir_21], axis=-1)

        if geojson_fpath is not None:
            img_fci = mask_geojson(geojson_fpath, lon_2d, lat_2d, img_fci, proj_plot, proj_data)

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, img_fci,
                      shading='nearest',
                      zorder=2,
                      vmin=0., vmax=1.,
                      transform=proj_data)
        # ax.set_boundary(boundary, transform=proj_data)

        title = "{} ({}) False Color (3-6-7) - ".format(instrument, satellite) + dt_title
        add_ancillary(ax, buoys, title)

        # view_extent = [lonmin - 5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax.set_extent(VIEW_EXTENT, proj_data)


        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        counter += 1
        # break
    return 1


def create_false_color_721_imagery(fdir, h5filename, outdir, geojson_fpath, time_range, ndir_recent, test, nan_pct, buoys, scale_factors=[1.75, 1., 1.5]):
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
    # lon_range = [-100-2, 0+2]
    # lat_range = [75-0.5, 89+0.5]
    vlon, vlat = np.mean([lon_1d.min(), lon_1d.max()]), np.mean([lat_1d.min(), lat_1d.max()])
    # proj_plot = ccrs.NorthPolarStereo(central_longitude=vlon)
    proj_plot = ccrs.Orthographic(central_longitude=VLON, central_latitude=VLAT)

    # semicircle boundary
    # lonmin, lonmax, latmin, latmax = -140, 40, 73, 90
    lonmin, lonmax, latmin, latmax = lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()
    vertices = [(lon, int(latmin)) for lon in range(int(lonmin), int(lonmax) + 1, 1)] + \
    [(lon, int(latmax)) for lon in range(int(lonmax), int(lonmin) - 1, -1)]
    boundary = matplotlib.path.Path(vertices)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_dir = os.path.join(outdir, "false_color_721")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    print("Found {} overpasses from {} sub-directories".format(len(acq_dts), len(os.listdir(fdir))))

    counter = 0
    for acq_dt in acq_dts:

        dt_title = doy_2_date_str(acq_dt) + "Z"
        satellite = sat_name[acq_dt]
        instrument = get_instrument(satellite)
        fname_target = format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
            continue

        swir_21 = np.clip(normalize_data(ref['2130'][acq_dt]) * scale_factors[0], 0, 1)
        nir     = np.clip(normalize_data(ref['860'][acq_dt]) * scale_factors[1], 0, 1)
        red     = np.clip(normalize_data(ref['650'][acq_dt]) * scale_factors[2], 0, 1)

        img_fci = np.stack([swir_21, nir, red], axis=-1)

        if geojson_fpath is not None:
            img_fci = mask_geojson(geojson_fpath, lon_2d, lat_2d, img_fci, proj_plot, proj_data)

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, img_fci,
                      shading='nearest',
                      zorder=2,
                      vmin=0., vmax=1.,
                      transform=proj_data)
        # ax.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) False Color (7-2-1) - ".format(instrument, satellite) + dt_title
        add_ancillary(ax, buoys, title)

        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax.set_extent(VIEW_EXTENT, proj_data)

        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        counter += 1
        # break
    return 1


def create_true_color_imagery(fdir, h5filename, outdir, geojson_fpath, time_range, ndir_recent, test, nan_pct, buoys):

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


    vlon, vlat = np.mean([lon_1d.min(), lon_1d.max()]), np.mean([lat_1d.min(), lat_1d.max()])
    # proj_plot = ccrs.NorthPolarStereo(central_longitude=vlon)
    proj_plot = ccrs.Orthographic(central_longitude=VLON, central_latitude=VLAT)

    # semicircle boundary
    # lonmin, lonmax, latmin, latmax = -140, 40, 73, 90
    lonmin, lonmax, latmin, latmax = lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()
    vertices = [(lon, int(latmin)) for lon in range(int(lonmin), int(lonmax) + 1, 1)] + \
    [(lon, int(latmax)) for lon in range(int(lonmax), int(lonmin) - 1, -1)]
    boundary = matplotlib.path.Path(vertices)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_dir = os.path.join(outdir, "true_color")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    print("Found {} overpasses from {} sub-directories".format(len(acq_dts), len(os.listdir(fdir))))

    counter = 0
    for acq_dt in acq_dts:

        dt_title = doy_2_date_str(acq_dt) + "Z"
        satellite = sat_name[acq_dt]
        instrument = get_instrument(satellite)
        fname_target = format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
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

        if geojson_fpath is not None:
            rgb = mask_geojson(geojson_fpath, lon_2d, lat_2d, rgb, proj_plot, proj_data)

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, rgb,
                      shading='nearest',
                      zorder=2,
                      vmin=0., vmax=1.,
                      transform=proj_data)
        # ax.set_boundary(boundary, transform=proj_data)

        # title = "MODIS ({}) True Color - ".format(satellite) + title
        title = "{} ({}) True Color  - ".format(instrument, satellite) + dt_title
        add_ancillary(ax, buoys, title)

        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax.set_extent(VIEW_EXTENT, proj_data)

        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        counter += 1
        # break
    return 0


"""
def create_reflectance_imagery(fdir, h5filename, outdir, test, nan_pct):

    ref = {}
    lats, lons, sat_name, ref['2130'] = read_multiple_h5(fdir, h5filename, test=test, param='ref', wvl=2130, nan_pct=nan_pct)
    _, _, _, ref['860']               = read_multiple_h5(fdir, h5filename, test=test, param='ref', wvl=860,  nan_pct=nan_pct)
    _, _, _, ref['650']               = read_multiple_h5(fdir, h5filename, test=test, param='ref', wvl=650,  nan_pct=nan_pct)
    _, _, _, ref['555']               = read_multiple_h5(fdir, h5filename, test=test, param='ref', wvl=555,  nan_pct=nan_pct)
    _, _, _, ref['470']               = read_multiple_h5(fdir, h5filename, test=test, param='ref', wvl=470,  nan_pct=nan_pct)
    _, _, _, ref['1640']              = read_multiple_h5(fdir, h5filename, test=test, param='ref', wvl=1640, nan_pct=nan_pct)


    acq_dts = list(ref['2130'].keys())
    acq_dts = sorted(acq_dts, key = lambda x: x[1:])

    lon_1d = lons[acq_dts[0]]
    lat_1d = lats[acq_dts[0]]
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    # lon_range = [-100-2, 0+2]
    # lat_range = [75-0.5, 89+0.5]
    vlon, vlat = np.mean([lon_1d.min(), lon_1d.max()]), np.mean([lat_1d.min(), lat_1d.max()])
    # proj_plot = ccrs.NorthPolarStereo(central_longitude=vlon)
    proj_plot = ccrs.Orthographic(central_longitude=VLON, central_latitude=VLAT)

    # semicircle boundary
    # lonmin, lonmax, latmin, latmax = -140, 40, 73, 90
    lonmin, lonmax, latmin, latmax = lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()
    vertices = [(lon, int(latmin)) for lon in range(int(lonmin), int(lonmax) + 1, 1)] + \
    [(lon, int(latmax)) for lon in range(int(lonmax), int(lonmin) - 1, -1)]
    boundary = matplotlib.path.Path(vertices)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_dir_721 = os.path.join(outdir, "false_color_721")
    save_dir_367 = os.path.join(outdir, "false_color_367")
    save_dir_true = os.path.join(outdir, "true_color")

    if not os.path.exists(save_dir_721):
        os.makedirs(save_dir_721)

    if not os.path.exists(save_dir_367):
        os.makedirs(save_dir_367)

    if not os.path.exists(save_dir_true):
        os.makedirs(save_dir_true)


    print("Found {} overpasses from {} sub-directories".format(len(acq_dts), len(os.listdir(fdir))))

    counter = 0
    for acq_dt in acq_dts:

        dt_title = doy_2_date_str(acq_dt) + "Z"
        satellite = sat_name[acq_dt]
        instrument = get_instrument(satellite)

        ################################ 7-2-1 Imagery ################################

        scale_factors_721 =[1.75, 1., 1.5]
        swir_21 = np.clip(normalize_data(ref['2130'][acq_dt]) * scale_factors_721[0], 0, 1)
        nir     = np.clip(normalize_data(ref['860'][acq_dt]) * scale_factors_721[1], 0, 1)
        red     = np.clip(normalize_data(ref['650'][acq_dt]) * scale_factors_721[2], 0, 1)

        img_fci = np.stack([swir_21, nir, red], axis=-1)
        # img_fci = normalize_data(img_fci)
        fig = plt.figure(figsize=(20, 20))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, img_fci,
                      shading='nearest',
                      zorder=2,
                      vmin=0., vmax=1.,
                      transform=proj_data)
        # ax.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) False Color (7-2-1) - ".format(instrument, satellite) + dt_title
        add_ancillary(ax, buoys, title)

        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax.set_extent(VIEW_EXTENT, proj_data)

        fname_target = format_acq_dt(acq_dt)
        fig.savefig(os.path.join("{}/{}.png".format(save_dir_721, fname_target)), dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()

        ################################ 3-6-7 Imagery ################################

        scale_factors_367 =[1., 1.75, 2.]
        blue    = np.clip(normalize_data(ref['470'][acq_dt])  * scale_factors_367[0], 0, 1)
        swir_16 = np.clip(normalize_data(ref['1640'][acq_dt]) * scale_factors_367[1], 0, 1)
        swir_21 = np.clip(normalize_data(ref['2130'][acq_dt]) * scale_factors_367[2], 0, 1)
        img_fci = np.stack([blue, swir_16, swir_21], axis=-1)

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, img_fci,
                      shading='nearest',
                      zorder=2,
                      vmin=0., vmax=1.,
                      transform=proj_data)
        # ax.set_boundary(boundary, transform=proj_data)

        title = "{} ({}) False Color (3-6-7) - ".format(instrument, satellite) + dt_title
        add_ancillary(ax, buoys, title)

        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax.set_extent(VIEW_EXTENT, proj_data)

        fig.savefig(os.path.join("{}/{}.png".format(save_dir, counter)), dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()

        ################################ True Color Imagery ################################

        red   = np.clip(ref['650'][acq_dt], 0, 1.1)
        red   = scale_255(red)

        green = np.clip(ref['555'][acq_dt], 0, 1.1)
        green = scale_255(green)

        blue  = np.clip(ref['470'][acq_dt], 0, 1.1)
        blue  = scale_255(blue)

        rgb   = np.stack([red, green, blue], axis=-1)
        # so that outliers don't offset colors too much
        rgb   = np.interp(rgb, (np.percentile(rgb, 1), np.percentile(rgb, 99)), (0, 1))

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, rgb,
                      shading='nearest',
                      zorder=2,
                      vmin=0., vmax=1.,
                      transform=proj_data)
        # ax.set_boundary(boundary, transform=proj_data)

        # title = "MODIS ({}) True Color - ".format(satellite) + title
        title = "{} ({}) True Color  - ".format(instrument, satellite) + dt_title
        add_ancillary(ax, buoys, title)

        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax.set_extent(VIEW_EXTENT, proj_data)

        fname_target = format_acq_dt(acq_dt)
        fig.savefig(os.path.join("{}/{}.png".format(save_dir, fname_target)), dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()

        counter += 1



    return 0
"""


def plot_liquid_water_paths(fdir, h5filename, outdir, geojson_fpath, time_range, ndir_recent, test, nan_pct, vmax, buoys):

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
    # lon_range = [-100-2, 0+2]
    # lat_range = [75-0.5, 89+0.5]
    vlon, vlat = np.mean([lon_1d.min(), lon_1d.max()]), np.mean([lat_1d.min(), lat_1d.max()])
    # proj_plot = ccrs.NorthPolarStereo(central_longitude=vlon)
    proj_plot = ccrs.Orthographic(central_longitude=VLON, central_latitude=VLAT)

    # extent = [lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()]

    # semicircle boundary
    # lonmin, lonmax, latmin, latmax = -140, 40, 73, 90
    lonmin, lonmax, latmin, latmax = lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()
    vertices = [(lon, int(latmin)) for lon in range(int(lonmin), int(lonmax) + 1, 1)] + \
    [(lon, int(latmax)) for lon in range(int(lonmax), int(lonmin) - 1, -1)]
    boundary = matplotlib.path.Path(vertices)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_dir = os.path.join(outdir, "water_path")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Found {} overpasses from {} sub-directories".format(len(acq_dts), len(os.listdir(fdir))))

    counter = 0
    for acq_dt in acq_dts:

        satellite = sat_name[acq_dt]
        dt_title  = doy_2_date_str(acq_dt) + "Z"

        instrument = get_instrument(satellite)
        fname_target = format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
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
            cbar_ticks = np.linspace(0, vmax, 4, dtype='int')
            cmap = arctic_cloud_cmap
            extend = 'max'
        else:
            cbar_ticks = np.linspace(0, np.nanmax([im_lwp, im_lwp_1621]), 4, dtype='int')
            cmap = arctic_cloud_alt_cmap
            extend = 'neither'

        if geojson_fpath is not None:
            im_lwp      = mask_geojson(geojson_fpath, lon_2d, lat_2d, im_lwp, proj_plot, proj_data)
            im_lwp_1621 = mask_geojson(geojson_fpath, lon_2d, lat_2d, im_lwp_1621, proj_plot, proj_data)

        fig = plt.figure(figsize=(20, 10))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 2, figure=fig)
        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_lwp,
                            shading='nearest',
                            zorder=1,
                            cmap=cmap,
                            transform=proj_data)
        # ax00.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) LWP - ".format(instrument, satellite) + dt_title
        add_ancillary(ax00, buoys, title)
        cbar = fig.colorbar(y00, ax=ax00, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.1, shrink=0.35)
        cbar.ax.set_title('$LWP \;\;[g/m^2]$', fontsize=18)
        cbar.ax.tick_params(length=0, labelsize=18)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax00.set_extent(VIEW_EXTENT, proj_data)

        ##############################################################

        ax01 = fig.add_subplot(gs[1], projection=proj_plot)
        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_lwp_1621,
                      shading='nearest',
                      zorder=1,
                      cmap=cmap,
                      transform=proj_data)
        # ax01.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) LWP (1621) - ".format(instrument, satellite) + dt_title
        add_ancillary(ax01, buoys, title)
        cbar = fig.colorbar(y01, ax=ax01, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.1, shrink=0.35)
        cbar.ax.set_title('$LWP \;\;[g/m^2]$', fontsize=18)
        cbar.ax.tick_params(length=0, labelsize=18)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax01.set_extent(VIEW_EXTENT, proj_data)

        ##############################################################

        fig.subplots_adjust(wspace=0.1)


        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        counter += 1
        # break

    return 1



def plot_ice_water_paths(fdir, h5filename, outdir, geojson_fpath, time_range, ndir_recent, test, nan_pct, vmax, buoys):

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
    # lon_range = [-100-2, 0+2]
    # lat_range = [75-0.5, 89+0.5]
    vlon, vlat = np.mean([lon_1d.min(), lon_1d.max()]), np.mean([lat_1d.min(), lat_1d.max()])
    # proj_plot = ccrs.NorthPolarStereo(central_longitude=vlon)
    proj_plot = ccrs.Orthographic(central_longitude=VLON, central_latitude=VLAT)

    # extent = [lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()]

    # semicircle boundary
    # lonmin, lonmax, latmin, latmax = -140, 40, 73, 90
    lonmin, lonmax, latmin, latmax = lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()
    vertices = [(lon, int(latmin)) for lon in range(int(lonmin), int(lonmax) + 1, 1)] + \
    [(lon, int(latmax)) for lon in range(int(lonmax), int(lonmin) - 1, -1)]
    boundary = matplotlib.path.Path(vertices)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_dir = os.path.join(outdir, "ice_path")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Found {} overpasses from {} sub-directories".format(len(acq_dts), len(os.listdir(fdir))))

    counter = 0
    for acq_dt in acq_dts:

        satellite = sat_name[acq_dt]
        dt_title  = doy_2_date_str(acq_dt) + "Z"

        instrument = get_instrument(satellite)
        fname_target = format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
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
            cbar_ticks = np.linspace(0, vmax, 4, dtype='int')
            cmap = arctic_cloud_cmap
            extend = 'max'
        else:
            cbar_ticks = np.linspace(0, np.nanmax([im_iwp_1621, im_iwp_1621]), 4, dtype='int')
            cmap = arctic_cloud_alt_cmap
            extend = 'neither'

        if geojson_fpath is not None:
            im_iwp      = mask_geojson(geojson_fpath, lon_2d, lat_2d, im_iwp, proj_plot, proj_data)
            im_iwp_1621 = mask_geojson(geojson_fpath, lon_2d, lat_2d, im_iwp_1621, proj_plot, proj_data)

        fig = plt.figure(figsize=(20, 10))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 2, figure=fig)
        ##############################################################

        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_iwp,
                      shading='nearest',
                      zorder=1,
                      cmap=cmap,
                      transform=proj_data)
        # ax10.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) IWP - ".format(instrument, satellite) + dt_title
        add_ancillary(ax00, buoys, title)
        cbar = fig.colorbar(y00, ax=ax00, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.1, shrink=0.35)
        cbar.ax.set_title('$IWP \;\;[g/m^2]$', fontsize=18)
        cbar.ax.tick_params(length=0, labelsize=18)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax00.set_extent(VIEW_EXTENT, proj_data)

        ##############################################################

        ax01 = fig.add_subplot(gs[1], projection=proj_plot)
        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_iwp_1621,
                      shading='nearest',
                      zorder=1,
                      cmap=cmap,
                      transform=proj_data)
        # ax11.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) IWP (1621) - ".format(instrument, satellite) + dt_title
        add_ancillary(ax01, buoys, title)
        cbar = fig.colorbar(y01, ax=ax01, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.1, shrink=0.35)
        cbar.ax.set_title('$IWP \;\;[g/m^2]$', fontsize=18)
        cbar.ax.tick_params(length=0, labelsize=18)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax01.set_extent(VIEW_EXTENT, proj_data)

        ##############################################################

        fig.subplots_adjust(wspace=0.1)

        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        counter += 1
        # break

    return 1


def plot_optical_depths(fdir, h5filename, outdir, geojson_fpath, time_range, ndir_recent, test, nan_pct, vmax, buoys):

    lats, lons, sat_name, cot, cot_1621  = read_multiple_h5(fdir, h5filename, param='cot', test=test, time_range=time_range, ndir_recent=ndir_recent, nan_pct=nan_pct)

    acq_dts = list(sat_name.keys())
    if len(acq_dts) == 0:
        print("Could not find any HDF5 files.\n")
        return 0

    acq_dts = sorted(acq_dts, key = lambda x: x[1:])

    lon_1d = lons[acq_dts[0]]
    lat_1d = lats[acq_dts[0]]
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    # lon_range = [-100-2, 0+2]
    # lat_range = [75-0.5, 89+0.5]
    vlon, vlat = np.mean([lon_1d.min(), lon_1d.max()]), np.mean([lat_1d.min(), lat_1d.max()])
    # proj_plot = ccrs.NorthPolarStereo(central_longitude=vlon)
    proj_plot = ccrs.Orthographic(central_longitude=VLON, central_latitude=VLAT)

    # extent = [lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()]

    # semicircle boundary
    # lonmin, lonmax, latmin, latmax = -140, 40, 73, 90
    lonmin, lonmax, latmin, latmax = lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()
    vertices = [(lon, int(latmin)) for lon in range(int(lonmin), int(lonmax) + 1, 1)] + \
    [(lon, int(latmax)) for lon in range(int(lonmax), int(lonmin) - 1, -1)]
    boundary = matplotlib.path.Path(vertices)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_dir = os.path.join(outdir, "optical_thickness")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Found {} overpasses from {} sub-directories".format(len(acq_dts), len(os.listdir(fdir))))

    vmax = 100.
    counter = 0
    for acq_dt in acq_dts:

        satellite = sat_name[acq_dt]
        dt_title  = doy_2_date_str(acq_dt) + "Z"
        instrument = get_instrument(satellite)
        fname_target = format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
            continue

        im_cot      = cot[acq_dt]
        im_cot_1621 = cot_1621[acq_dt]

        if vmax is not None:
            im_cot[im_cot > vmax]           = vmax
            im_cot_1621[im_cot_1621 > vmax] = vmax
            cbar_ticks = np.linspace(0, vmax, 4, dtype='int')
            cmap = arctic_cloud_cmap
            extend = 'max'
        else:
            cbar_ticks = np.linspace(0, np.nanmax([im_cot, im_cot_1621]), 4, dtype='int')
            cmap = arctic_cloud_alt_cmap
            extend = 'neither'

        if geojson_fpath is not None:
            im_cot      = mask_geojson(geojson_fpath, lon_2d, lat_2d, im_cot, proj_plot, proj_data)
            im_cot_1621 = mask_geojson(geojson_fpath, lon_2d, lat_2d, im_cot_1621, proj_plot, proj_data)

        fig = plt.figure(figsize=(20, 10))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 2, figure=fig)
        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_cot,
                            shading='nearest',
                            zorder=1,
                            cmap=cmap,
                            transform=proj_data)
        # ax00.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) COT - ".format(instrument, satellite) + dt_title
        add_ancillary(ax00, buoys, title)
        cbar = fig.colorbar(y00, ax=ax00, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.1, shrink=0.35)
        cbar.ax.set_title('$COT$', fontsize=18)
        cbar.ax.tick_params(length=0, labelsize=18)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax00.set_extent(VIEW_EXTENT, proj_data)

        ##############################################################

        ax01 = fig.add_subplot(gs[1], projection=proj_plot)
        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_cot_1621,
                      shading='nearest',
                      zorder=1,
                      cmap=cmap,
                      transform=proj_data)
        # ax01.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) COT (1621) - ".format(instrument, satellite) + dt_title
        add_ancillary(ax01, buoys, title)
        cbar = fig.colorbar(y01, ax=ax01, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.1, shrink=0.35)
        cbar.ax.set_title('$COT$', fontsize=18)
        cbar.ax.tick_params(length=0, labelsize=18)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax01.set_extent(VIEW_EXTENT, proj_data)


        ##############################################################

        fig.subplots_adjust(wspace=0.1)

        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        counter += 1
        # break

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
    parser = ArgumentParser(prog='timelapse', formatter_class=RawTextHelpFormatter)
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

    elif (args.mode.lower() == 'baffin') or (args.mode.lower() == 'bb') or (args.mode.lower() == 'baffin_bay'):
        VIEW_EXTENT = [-130, -40, 67, 84]
        VLON, VLAT = -60, 84
        
    # vlon, vlat = np.mean(lon_range), np.mean(lat_range)
    # proj_plot = ccrs.NorthPolarStereo(central_longitude=vlon)
    return_val = 0
    for param in tqdm(args.param):
        if param == "fci_367":
            return_val += create_false_color_367_imagery(fdir=args.fdir, h5filename=args.h5filename, outdir=args.outdir, geojson_fpath=args.geojson, test=args.test, nan_pct=args.nan_pct, time_range=args.range, ndir_recent=args.ndir_recent, buoys=args.buoys)

        if param == "fci_721":
            return_val += create_false_color_721_imagery(fdir=args.fdir, h5filename=args.h5filename, outdir=args.outdir, geojson_fpath=args.geojson, test=args.test, nan_pct=args.nan_pct, time_range=args.range, ndir_recent=args.ndir_recent, buoys=args.buoys)

        if param == "tci":
            return_val += create_true_color_imagery(fdir=args.fdir, h5filename=args.h5filename, outdir=args.outdir, geojson_fpath=args.geojson, test=args.test, nan_pct=args.nan_pct, time_range=args.range, ndir_recent=args.ndir_recent, buoys=args.buoys)

        if param == "cwp":
            return_val += plot_liquid_water_paths(fdir=args.fdir, h5filename=args.h5filename, outdir=args.outdir, geojson_fpath=args.geojson, test=args.test, nan_pct=args.nan_pct, vmax=args.vmax, time_range=args.range, ndir_recent=args.ndir_recent, buoys=args.buoys)
            return_val += plot_ice_water_paths(fdir=args.fdir, h5filename=args.h5filename, outdir=args.outdir, geojson_fpath=args.geojson, test=args.test, nan_pct=args.nan_pct, vmax=args.vmax, time_range=args.range, ndir_recent=args.ndir_recent, buoys=args.buoys)

        if param == "cot":
            return_val += plot_optical_depths(fdir=args.fdir, h5filename=args.h5filename, outdir=args.outdir, geojson_fpath=args.geojson, test=args.test, nan_pct=args.nan_pct, vmax=args.vmax, time_range=args.range, ndir_recent=args.ndir_recent, buoys=args.buoys)

    print("Created png files for {} types of imagery.\n".format(return_val))
