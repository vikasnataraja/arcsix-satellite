import os
import json
import time
import requests
import datetime
import matplotlib
import cartopy
import warnings
import numpy as np

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import cmasher as cmr

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

############################## Cloud phase IR colormap ##############################
ctp_ir_cmap_arr = np.array([
                            [0.5, 0.5, 0.5, 1.], # clear
                            [0., 0., 0.55, 1.], # liquid
                            [0.75, 0.85, 0.95, 1.], # ice
                            [0.55, 0.55, 0.95, 1.], # mixed
                            [0., 0.95, 0.95, 1.]])# undet. phase
ctp_ir_cmap_ticklabels = np.array(["clear", "liquid", "ice", "mixed phase", "uncertain"])
ctp_ir_tick_locs = (np.arange(len(ctp_ir_cmap_ticklabels)) + 0.5)*(len(ctp_ir_cmap_ticklabels) - 1)/len(ctp_ir_cmap_ticklabels)
ctp_ir_cmap = matplotlib.colors.ListedColormap(ctp_ir_cmap_arr)
ctp_ir_cmap.set_bad("black", 1)


############################## Cloud phase SWIR/COP colormap ##############################
ctp_swir_cmap_arr = np.array([
                            #   [0, 0, 0, 1], # undet. mask
                              [0.5, 0.5, 0.5, 1.], # clear
                              [0., 0., 0.55, 1.], # liquid
                              [0.75, 0.85, 0.95, 1.], # ice
                              [0., 0.95, 0.95, 1.]])# no phase (liquid)
ctp_swir_cmap_ticklabels = np.array(["clear", "liquid", "ice", "uncertain"])
ctp_swir_tick_locs = (np.arange(len(ctp_swir_cmap_ticklabels)) + 0.5)*(len(ctp_swir_cmap_ticklabels) - 1)/len(ctp_swir_cmap_ticklabels)
ctp_swir_cmap = matplotlib.colors.ListedColormap(ctp_swir_cmap_arr)
# ctp_swir_cmap.set_bad("black", 1)


############################## Cloud top height colormap ##############################
cth_cmap_arr = np.array([[0., 0., 0., 1], # no retrieval
                        [0.5, 0.5, 0.5, 1], # clear
                        [0.05, 0.7, 0.95, 1], # low clouds
                        [0.65, 0.05, 0.3, 1.],  # mid clouds
                        [0.95, 0.95, 0.95, 1.]])    # high clouds
cth_cmap_ticklabels = ["undet.", "clear", "low\n0.1 - 2 km", "mid\n2 - 6 km", "high\n>=6 km"]
cth_tick_locs = (np.arange(len(cth_cmap_ticklabels)) + 0.5)*(len(cth_cmap_ticklabels) - 1)/len(cth_cmap_ticklabels)
cth_cmap = matplotlib.colors.ListedColormap(cth_cmap_arr)

############################## Cloud top temperature colormap ##############################
ctt_cmap_arr = np.array(list(plt.get_cmap('Blues_r')(np.linspace(0, 0.8, 4))) + list(plt.get_cmap('Reds')(np.linspace(0, 1, 4))))
ctt_cmap = matplotlib.colors.ListedColormap(ctt_cmap_arr)


arctic_cloud_cmap = 'RdBu_r'
arctic_cloud_alt_cmap = 'RdBu_r'


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Imagery:

    def __init__(self, verbose=False):

        self.verbose = verbose


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


    def doy_2_date_str(self, acq_dt):
        year, doy, hours, minutes = acq_dt[1:5], acq_dt[5:8], acq_dt[9:11], acq_dt[11:13]
        date = datetime.datetime.strptime('{} {} {} {}'.format(year, doy, hours, minutes), '%Y %j %H %M')
        return date.strftime('%B %d, %Y: %H%M')


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

        vmax = np.nanpercentile(band, 99)
        band = np.clip(band, 0., vmax)
        if scale:
            band_mask = np.ma.masked_where((np.isnan(band)), band)
            band = self.scale_255(band, vmax)
            band = np.ma.masked_where(np.ma.getmask(band_mask), band)
            return band
        else:
            band = (band - np.nanmin(band)) / (vmax - np.nanmin(band))
            band = np.ma.masked_where((np.isnan(band)), band)
            return band


    def create_3d_mask(self, red, green, blue):
        red_mask, green_mask, blue_mask = red.mask, green.mask, blue.mask
        if red_mask.size != red.data.size:
            red_mask = np.full(red.shape, False)
        if green_mask.size != green.data.size:
            green_mask = np.full(green.shape, False)
        if blue_mask.size != blue.data.size:
            blue_mask = np.full(blue.shape, False)

        return np.stack([red_mask, green_mask, blue_mask], axis=-1)


    def get_instrument(self, satellite):
        if (satellite == 'Aqua') or (satellite == 'Terra'):
            instrument = 'MODIS'
        elif (satellite == 'Suomi-NPP') or (satellite == 'NOAA-20/JPSS-1') or (satellite == 'NOAA-21/JPSS-2'):
            instrument = 'VIIRS'
        else:
            instrument = 'Unknown'

        return instrument


    def get_buoy_data(self, json_fpath, API_KEY="kU7vw3YBcIvnxqTH8DlDeR08QTTfNYiZ", days=7, download_threshold_hrs=6, force_download=False):

        import pandas as pd
        # def read_exist_buoy(download_json_fpath, metadata_json_fpath)
        fdir = os.path.dirname(os.path.abspath(json_fpath))
        fields_str = '?field=time_stamp&field=latitude&field=longitude'
        # for field in fields:
        #     fields_str += 'field={}&'.format(field)

        with open(json_fpath, 'r') as fp:
            urls = json.load(fp)

        # read metadata
        buoy_metadata_json = os.path.join(fdir, 'metadata.json')
        meta = None
        if os.path.isfile(buoy_metadata_json):
            with open(buoy_metadata_json, "r") as fm:
                meta = json.load(fm)

        utc_now_dt = datetime.datetime.now(datetime.timezone.utc)
        utc_now_dt = utc_now_dt.replace(tzinfo=None) # so that timedelta does not raise an error
        dts, lons, lats = {}, {}, {}
        # meta_dict = {}
        for bname, burl in urls.items():
            url = burl + fields_str

            buoy_download_json = os.path.join(fdir, 'download_buoy{}.json'.format(bname))

            # essentially force download; read old data only if it's actually available
            check_dt = utc_now_dt - datetime.timedelta(hours=download_threshold_hrs+1)
            if os.path.isfile(buoy_download_json) and (meta is not None):
                check_dt = datetime.datetime.strptime(meta['last_downloaded_buoy{}'.format(bname.upper())], "%Y-%m-%d_%H:%M:%S")

            # if less than time threshold, read old data; otherwise download from server and save to file
            if ((utc_now_dt - check_dt) < datetime.timedelta(hours=download_threshold_hrs)) and (not force_download):
                # print("Message [get_buoy_data]: Using existing data file for: Buoy {}".format(bname))
                with open(buoy_download_json, 'r') as f:
                    data = json.load(f)
                    df   = pd.DataFrame(data)
            else:
                # print("Message [get_buoy_data]: Downloading data for: Buoy {}".format(bname))
                data = requests.get(url, headers={'Authorization':'Bearer {}'.format(API_KEY)}, timeout=(20, 30)).json()
                df   = pd.DataFrame(data)
                with open(buoy_download_json, 'w') as f: # save for next time
                    json.dump(data, f)
                meta['last_downloaded_buoy{}'.format(bname.upper())] = utc_now_dt.strftime("%Y-%m-%d_%H:%M:%S")


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


    def mask_geojson(self, geojson_fpath, lon_2d, lat_2d, dat, proj_plot, proj_data):

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


    def add_ancillary(self, ax, buoys, title=None, scale=1):
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
                    x_inline=False, y_inline=True, crs=proj_data)

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

        if buoys is not None:
            dt, lons, lats = self.get_buoy_data(buoys)

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


    def convert_ir_ctp(self, ctp_ir_arr):
        ctp_ir_arr[ctp_ir_arr==6] = 4
        ctp_ir_arr = np.ma.masked_where(np.isnan(ctp_ir_arr), ctp_ir_arr)
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



    def create_false_color_721_imagery(self, lon_2d, lat_2d, red, green, blue, sza, satellite, acq_dt, outdir, geojson_fpath, buoys, mode):

        proj_plot = ccrs.Orthographic(central_longitude=ccrs_views[mode]['vlon'], central_latitude=ccrs_views[mode]['vlat'])

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        save_dir = os.path.join(outdir, "false_color_721")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_title = self.doy_2_date_str(acq_dt) + "Z"
        instrument = self.get_instrument(satellite)
        fname_target = self.format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
            return 0

        red    = self.preprocess_band(red,   sza=sza)
        green  = self.preprocess_band(green, sza=sza)
        blue   = self.preprocess_band(blue,  sza=sza)

        img_fci = np.stack([red, green, blue], axis=-1)

        if geojson_fpath is not None:
            img_fci = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, img_fci, proj_plot, proj_data)

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, img_fci,
                        shading='nearest',
                        zorder=2,
                        transform=proj_data)
        # ax.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) False Color (7-2-1) - ".format(instrument, satellite) + dt_title
        self.add_ancillary(ax, buoys, title)

        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax.set_extent(ccrs_views[mode]['view_extent'], proj_data)

        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        return 1


    def create_false_color_367_imagery(self, lon_2d, lat_2d, red, green, blue, sza, satellite, acq_dt, outdir, geojson_fpath, buoys, mode):

        proj_plot = ccrs.Orthographic(central_longitude=ccrs_views[mode]['vlon'], central_latitude=ccrs_views[mode]['vlat'])

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        save_dir = os.path.join(outdir, "false_color_367")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_title = self.doy_2_date_str(acq_dt) + "Z"
        instrument = self.get_instrument(satellite)
        fname_target = self.format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
            return 0

        red    = self.preprocess_band(red,   sza=sza)
        green  = self.preprocess_band(green, sza=sza)
        blue   = self.preprocess_band(blue,  sza=sza)

        img_fci = np.stack([red, green, blue], axis=-1)

        if geojson_fpath is not None:
            img_fci = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, img_fci, proj_plot, proj_data)

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, img_fci,
                        shading='nearest',
                        zorder=2,
                        transform=proj_data)
        # ax.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) False Color (3-6-7) - ".format(instrument, satellite) + dt_title
        self.add_ancillary(ax, buoys, title)

        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax.set_extent(ccrs_views[mode]['view_extent'], proj_data)

        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        return 1


    def create_true_color_imagery(self, lon_2d, lat_2d, red, green, blue, sza, satellite, acq_dt, outdir, geojson_fpath, buoys, mode):

        proj_plot = ccrs.Orthographic(central_longitude=ccrs_views[mode]['vlon'], central_latitude=ccrs_views[mode]['vlat'])

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        save_dir = os.path.join(outdir, "true_color")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_title = self.doy_2_date_str(acq_dt) + "Z"
        instrument = self.get_instrument(satellite)
        fname_target = self.format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
            return 0

        red    = self.preprocess_band(red,   sza=sza, scale=True)
        green  = self.preprocess_band(green, sza=sza, scale=True)
        blue   = self.preprocess_band(blue,  sza=sza, scale=True)

        rgb      = np.stack([red, green, blue], axis=-1)
        rgb_mask = self.create_3d_mask(red, green, blue)
        rgb      = np.ma.masked_array(rgb, mask=rgb_mask)
        # rgb = np.interp(rgb, (np.nanpercentile(rgb, 1), np.nanpercentile(rgb, 99)), (0, 1))

        if geojson_fpath is not None:
            rgb = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, rgb, proj_plot, proj_data)

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, rgb,
                        shading='nearest',
                        zorder=2,
                        # vmin=0., vmax=1.,
                        transform=proj_data)
        # ax.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) False Color (3-6-7) - ".format(instrument, satellite) + dt_title
        self.add_ancillary(ax, buoys, title)

        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax.set_extent(ccrs_views[mode]['view_extent'], proj_data)

        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        return 1


    def create_false_color_cirrus_imagery(self, lon_2d, lat_2d, red, green, blue, sza, satellite, acq_dt, outdir, geojson_fpath, buoys, mode):

        proj_plot = ccrs.Orthographic(central_longitude=ccrs_views[mode]['vlon'], central_latitude=ccrs_views[mode]['vlat'])

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        save_dir = os.path.join(outdir, "false_color_cirrus")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_title = self.doy_2_date_str(acq_dt) + "Z"
        instrument = self.get_instrument(satellite)
        fname_target = self.format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
            return 0

        red    = self.preprocess_band(red,   sza=sza)
        green  = self.preprocess_band(green, sza=sza)
        blue   = self.preprocess_band(blue,  sza=sza)

        img_fci = np.stack([red, green, blue], axis=-1)

        if geojson_fpath is not None:
            img_fci = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, img_fci, proj_plot, proj_data)

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, img_fci,
                        shading='nearest',
                        zorder=2,
                        transform=proj_data)
        # ax.set_boundary(boundary, transform=proj_data)
        if instrument == 'MODIS':
            title = "{} ({}) False Color (1.38-1.64-2.13$\;\mu m$) - ".format(instrument, satellite) + dt_title
        else:
            title = "{} ({}) False Color (1.38-1.61-2.25$\;\mu m$) - ".format(instrument, satellite) + dt_title

        self.add_ancillary(ax, buoys, title)

        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax.set_extent(ccrs_views[mode]['view_extent'], proj_data)

        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        return 1


    def create_false_color_ir_imagery(self, lon_2d, lat_2d, red, green, blue, satellite, acq_dt, outdir, geojson_fpath, buoys, mode):

        proj_plot = ccrs.Orthographic(central_longitude=ccrs_views[mode]['vlon'], central_latitude=ccrs_views[mode]['vlat'])

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        save_dir = os.path.join(outdir, "false_color_ir")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_title = self.doy_2_date_str(acq_dt) + "Z"
        instrument = self.get_instrument(satellite)
        fname_target = self.format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
            return 0

        red    = self.preprocess_band(red,   sza=None)
        green  = self.preprocess_band(green, sza=None)
        blue   = self.preprocess_band(blue,  sza=None)

        img_fci = np.stack([red, green, blue], axis=-1)

        if geojson_fpath is not None:
            img_fci = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, img_fci, proj_plot, proj_data)

        fig = plt.figure(figsize=(20, 20))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0], projection=proj_plot)

        ax.pcolormesh(lon_2d, lat_2d, img_fci,
                        shading='nearest',
                        zorder=2,
                        transform=proj_data)
        # ax.set_boundary(boundary, transform=proj_data)
        if instrument == 'MODIS':
            title = "{} ({}) False Color (11.03-1.64-2.13$\;\mu m$) - ".format(instrument, satellite) + dt_title
        else:
            title = "{} ({}) False Color (10.76-1.61-2.25$\;\mu m$) - ".format(instrument, satellite) + dt_title

        self.add_ancillary(ax, buoys, title)

        # view_extent = [lonmin - 2.5, lonmin + 2.5, latmin - 0.5, min(latmax + 0.5, 89)]
        ax.set_extent(ccrs_views[mode]['view_extent'], proj_data)

        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        return 1


    def plot_liquid_water_paths(self, lon_2d, lat_2d, ctp, cwp, cwp_1621, satellite, acq_dt, outdir, geojson_fpath, buoys, mode):

        #     " The values in this SDS are set to mean the following:                              \n",
        #     " 0 -- cloud mask undetermined                                                       \n",
        #     " 1 -- clear sky                                                                     \n",
        #     " 2 -- liquid water cloud                                                            \n",
        #     " 3 -- ice cloud                                                                     \n",
        #     " 4 -- undetermined phase cloud (but retrieval is attempted as  liquid water)        \n",

        proj_plot = ccrs.Orthographic(central_longitude=ccrs_views[mode]['vlon'], central_latitude=ccrs_views[mode]['vlat'])

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        save_dir = os.path.join(outdir, "water_path")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_title  = self.doy_2_date_str(acq_dt) + "Z"
        instrument = self.get_instrument(satellite)
        fname_target = self.format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
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
            cmap = arctic_cloud_cmap
            extend = 'max'
        else:
            cbar_ticks = np.linspace(0, np.nanmax([im_lwp, im_lwp_1621]), 5, dtype='int')
            cmap = arctic_cloud_alt_cmap
            extend = 'neither'

        if geojson_fpath is not None:
            im_lwp      = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, im_lwp, proj_plot, proj_data)
            im_lwp_1621 = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, im_lwp_1621, proj_plot, proj_data)

        ##############################################################

        fig = plt.figure(figsize=(40, 40))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 2, figure=fig)
        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_lwp,
                            shading='nearest',
                            zorder=2,
                            cmap=cmap,
                            transform=proj_data)
        # ax00.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) LWP - ".format(instrument, satellite) + dt_title
        self.add_ancillary(ax00, buoys, title, scale=1.4)
        cbar = fig.colorbar(y00, ax=ax00, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.05, shrink=0.45)
        cbar.ax.set_title('$LWP \;\;[g/m^2]$', fontsize=24)
        cbar.ax.tick_params(length=0, labelsize=24)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        ax00.set_extent(ccrs_views[mode]['view_extent'], proj_data)
        ax00.set_aspect(1.25)

        ##############################################################

        ax01 = fig.add_subplot(gs[1], projection=proj_plot)
        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_lwp_1621,
                        shading='nearest',
                        zorder=1,
                        cmap=cmap,
                        transform=proj_data)
        # ax01.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) LWP (1621) - ".format(instrument, satellite) + dt_title
        self.add_ancillary(ax01, buoys, title, scale=1.4)
        cbar = fig.colorbar(y01, ax=ax01, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.05, shrink=0.45)
        cbar.ax.set_title('$LWP \;\;[g/m^2]$', fontsize=24)
        cbar.ax.tick_params(length=0, labelsize=24)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        ax01.set_extent(ccrs_views[mode]['view_extent'], proj_data)
        ax01.set_aspect(1.25)

        ##############################################################

        fig.subplots_adjust(wspace=0.1)
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        return 1


    def plot_ice_water_paths(self, lon_2d, lat_2d, ctp, cwp, cwp_1621, satellite, acq_dt, outdir, geojson_fpath, buoys, mode):

        #     " The values in this SDS are set to mean the following:                              \n",
        #     " 0 -- cloud mask undetermined                                                       \n",
        #     " 1 -- clear sky                                                                     \n",
        #     " 2 -- liquid water cloud                                                            \n",
        #     " 3 -- ice cloud                                                                     \n",
        #     " 4 -- undetermined phase cloud (but retrieval is attempted as  liquid water)        \n",

        proj_plot = ccrs.Orthographic(central_longitude=ccrs_views[mode]['vlon'], central_latitude=ccrs_views[mode]['vlat'])

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        save_dir = os.path.join(outdir, "ice_path")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_title  = self.doy_2_date_str(acq_dt) + "Z"

        instrument = self.get_instrument(satellite)
        fname_target = self.format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
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
            cmap = arctic_cloud_cmap
            extend = 'max'
        else:
            cbar_ticks = np.linspace(0, np.nanmax([im_iwp_1621, im_iwp_1621]), 5, dtype='int')
            cmap = arctic_cloud_alt_cmap
            extend = 'neither'

        if geojson_fpath is not None:
            im_iwp      = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, im_iwp, proj_plot, proj_data)
            im_iwp_1621 = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, im_iwp_1621, proj_plot, proj_data)

        fig = plt.figure(figsize=(40, 40))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 2, figure=fig)
        ##############################################################

        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_iwp,
                            shading='nearest',
                            zorder=2,
                            cmap=cmap,
                            transform=proj_data)
        title = "{} ({}) IWP - ".format(instrument, satellite) + dt_title
        self.add_ancillary(ax00, buoys, title, scale=1.4)
        cbar = fig.colorbar(y00, ax=ax00, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.05, shrink=0.45)
        cbar.ax.set_title('$IWP \;\;[g/m^2]$', fontsize=24)
        cbar.ax.tick_params(length=0, labelsize=24)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        ax00.set_extent(ccrs_views[mode]['view_extent'], proj_data)
        ax00.set_aspect(1.25)

        ##############################################################

        ax01 = fig.add_subplot(gs[1], projection=proj_plot)
        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_iwp_1621,
                            shading='nearest',
                            zorder=1,
                            cmap=cmap,
                            transform=proj_data)
        title = "{} ({}) IWP (1621) - ".format(instrument, satellite) + dt_title
        self.add_ancillary(ax01, buoys, title, scale=1.4)
        cbar = fig.colorbar(y01, ax=ax01, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.05, shrink=0.45)
        cbar.ax.set_title('$IWP \;\;[g/m^2]$', fontsize=24)
        cbar.ax.tick_params(length=0, labelsize=24)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        ax01.set_extent(ccrs_views[mode]['view_extent'], proj_data)
        ax01.set_aspect(1.25)

        ##############################################################

        fig.subplots_adjust(wspace=0.1)
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        return 1



    def plot_optical_depths(self, lon_2d, lat_2d, cot, cot_1621, satellite, acq_dt, outdir, geojson_fpath, buoys, mode):

        proj_plot = ccrs.Orthographic(central_longitude=ccrs_views[mode]['vlon'], central_latitude=ccrs_views[mode]['vlat'])

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        save_dir = os.path.join(outdir, "optical_thickness")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        vmax = 100.

        dt_title     = self.doy_2_date_str(acq_dt) + "Z"
        instrument   = self.get_instrument(satellite)
        fname_target = self.format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname   = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
            return 0

        im_cot      = cot
        im_cot_1621 = cot_1621

        if vmax is not None:
            im_cot[im_cot > vmax]           = vmax
            im_cot_1621[im_cot_1621 > vmax] = vmax
            cbar_ticks = np.linspace(0, vmax, 5, dtype='int')
            cmap = arctic_cloud_cmap
            extend = 'max'
        else:
            cbar_ticks = np.linspace(0, np.nanmax([im_cot, im_cot_1621]), 5, dtype='int')
            cmap = arctic_cloud_alt_cmap
            extend = 'neither'

        if geojson_fpath is not None:
            im_cot      = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, im_cot, proj_plot, proj_data)
            im_cot_1621 = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, im_cot_1621, proj_plot, proj_data)

        fig = plt.figure(figsize=(40, 40))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 2, figure=fig)
        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_cot,
                            shading='nearest',
                            zorder=2,
                            cmap=cmap,
                            transform=proj_data)
        # ax00.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) COT - ".format(instrument, satellite) + dt_title
        self.add_ancillary(ax00, buoys, title, scale=1.4)
        cbar = fig.colorbar(y00, ax=ax00, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.05, shrink=0.45)
        cbar.ax.set_title('$COT$', fontsize=24)
        cbar.ax.tick_params(length=0, labelsize=24)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        ax00.set_extent(ccrs_views[mode]['view_extent'], proj_data)
        ax00.set_aspect(1.25)

        ##############################################################

        ax01 = fig.add_subplot(gs[1], projection=proj_plot)
        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_cot_1621,
                            shading='nearest',
                            zorder=1,
                            cmap=cmap,
                            transform=proj_data)

        title = "{} ({}) COT (1621) - ".format(instrument, satellite) + dt_title
        self.add_ancillary(ax01, buoys, title, scale=1.4)
        cbar = fig.colorbar(y01, ax=ax01, ticks=cbar_ticks, extend=extend,
                            orientation='horizontal', location='bottom',
                            pad=0.05, shrink=0.45)
        cbar.ax.set_title('$COT$', fontsize=24)
        cbar.ax.tick_params(length=0, labelsize=24)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
        ax01.set_extent(ccrs_views[mode]['view_extent'], proj_data)
        ax01.set_aspect(1.25)

        ##############################################################

        fig.subplots_adjust(wspace=0.1)
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        return 1


    def plot_cloud_phase(self, lon_2d, lat_2d, ctp_swir, ctp_ir, satellite, acq_dt, outdir, geojson_fpath, buoys, mode):

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

        proj_plot = ccrs.Orthographic(central_longitude=ccrs_views[mode]['vlon'], central_latitude=ccrs_views[mode]['vlat'])

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        save_dir = os.path.join(outdir, "cloud_phase")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        patches_legend_swir = []
        patches_legend_ir   = []
        dt_title     = self.doy_2_date_str(acq_dt) + "Z"
        instrument   = self.get_instrument(satellite)
        fname_target = self.format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
            return 0

        im_ctp_swir = ctp_swir
        im_ctp_ir   = ctp_ir

        if geojson_fpath is not None:
            im_ctp_swir = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, im_ctp_swir, proj_plot, proj_data)
            im_ctp_ir   = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, im_ctp_ir, proj_plot, proj_data)

        # im_ctp_swir = np.nan_to_num(im_ctp_swir, 0)
        im_ctp_swir = np.ma.masked_where((np.isnan(im_ctp_swir) | (im_ctp_swir == 0)), im_ctp_swir)

        fig = plt.figure(figsize=(40, 40))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 2, figure=fig)
        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_ctp_swir,
                            shading='nearest',
                            zorder=2,
                            cmap=ctp_swir_cmap,
                            transform=proj_data)
        # ax00.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) Cloud Phase (SWIR/COP) - ".format(instrument, satellite) + dt_title
        self.add_ancillary(ax00, buoys, title, scale=1.4)
        ax00.set_extent(ccrs_views[mode]['view_extent'], proj_data)

        # create legend labels
        for i in range(len(ctp_swir_cmap_ticklabels)):
            patches_legend_swir.append(matplotlib.patches.Patch(color=ctp_swir_cmap_arr[i] , label=ctp_swir_cmap_ticklabels[i]))

        ax00.legend(handles=patches_legend_swir, loc='upper center', bbox_to_anchor=(0.5, -0.05), facecolor='white',
                    ncol=len(patches_legend_swir), fancybox=True, shadow=False, frameon=False, prop={'size': 24})
        ax00.set_aspect(1.25)
        ##############################################################

        ax01 = fig.add_subplot(gs[1], projection=proj_plot)
        im_ctp_ir = self.convert_ir_ctp(im_ctp_ir)
        y01 = ax01.pcolormesh(lon_2d, lat_2d, im_ctp_ir,
                    shading='nearest',
                    zorder=2,
                    cmap=ctp_ir_cmap,
                    transform=proj_data)
        # ax01.set_boundary(boundary, transform=proj_data)
        title = "{} ({}) Cloud Phase (IR) - ".format(instrument, satellite) + dt_title
        self.add_ancillary(ax01, buoys, title, scale=1.4)
        ax01.set_extent(ccrs_views[mode]['view_extent'], proj_data
                        )
        # create legend labels
        for i in range(len(ctp_ir_cmap_ticklabels)):
            patches_legend_ir.append(matplotlib.patches.Patch(color=ctp_ir_cmap_arr[i] , label=ctp_ir_cmap_ticklabels[i]))

        ax01.legend(handles=patches_legend_ir, loc='upper center', bbox_to_anchor=(0.5, -0.05), facecolor='white',
                    ncol=len(patches_legend_ir), fancybox=True, shadow=False, frameon=False, prop={'size': 24})
        ax01.set_aspect(1.25)
        ##############################################################

        fig.subplots_adjust(wspace=0.1)
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        return 1


    def plot_cloud_top(self, lon_2d, lat_2d, cth, ctt, satellite, acq_dt, outdir, geojson_fpath, buoys, mode):

        proj_plot = ccrs.Orthographic(central_longitude=ccrs_views[mode]['vlon'], central_latitude=ccrs_views[mode]['vlat'])

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        save_dir = os.path.join(outdir, "cloud_top_height_temperature")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        patches_legend_cth = []
        patches_legend_ctt = []

        dt_title  = self.doy_2_date_str(acq_dt) + "Z"
        instrument = self.get_instrument(satellite)
        fname_target = self.format_acq_dt(acq_dt)
        sat_fname    = satellite.split('/')[0]
        full_fname = "{}/{}_{}.png".format(save_dir, fname_target, sat_fname)
        if os.path.isfile(full_fname):
            print("Message [timelapse]: {} skipped since it already exists.".format(full_fname))
            return 0

        im_cth = cth
        im_ctt = ctt

        if geojson_fpath is not None:
            im_cth = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, im_cth, proj_plot, proj_data)
            im_ctt = self.mask_geojson(geojson_fpath, lon_2d, lat_2d, im_ctt, proj_plot, proj_data)

        fig = plt.figure(figsize=(40, 40))
        plt.style.use(mpl_style)
        gs  = GridSpec(1, 2, figure=fig)
        ax00 = fig.add_subplot(gs[0], projection=proj_plot)
        im_cth_binned = self.bin_cth_to_class(im_cth)
        y00 = ax00.pcolormesh(lon_2d, lat_2d, im_cth_binned,
                            shading='nearest',
                            zorder=1,
                            cmap=cth_cmap,
                            transform=proj_data)

        title = "{} ({}) Cloud Top Height - ".format(instrument, satellite) + dt_title
        self.add_ancillary(ax00, buoys, title, scale=1.4)

        for i in range(len(cth_cmap_ticklabels)):
            patches_legend_cth.append(matplotlib.patches.Patch(color=cth_cmap_arr[i] , label=cth_cmap_ticklabels[i]))

        ax00.legend(handles=patches_legend_cth, loc='upper center', bbox_to_anchor=(0.5, -0.05), facecolor='white',
                    ncol=len(patches_legend_cth), fancybox=True, shadow=False, frameon=False, prop={'size': 24})
        ax00.set_extent(ccrs_views[mode]['view_extent'], proj_data)
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
                            zorder=1,
                            cmap=ctt_cmap,
                            transform=proj_data)

        title = "{} ({}) Cloud Top Temperature - ".format(instrument, satellite) + dt_title
        self.add_ancillary(ax01, buoys, title, scale=1.4)
        for i in range(len(ctt_cmap_ticklabels)):
            patches_legend_ctt.append(matplotlib.patches.Patch(color=ctt_cmap_arr[i] , label=ctt_cmap_ticklabels[i]))

        ax01.legend(handles=patches_legend_ctt, loc='upper center', bbox_to_anchor=(0.5, -0.025), facecolor='white',
            ncol=int(len(patches_legend_ctt)/2), fancybox=True, shadow=False, frameon=False, prop={'size': 24},
            title='Cloud Top Temperature ($^\circ C$ )', title_fontsize=18)
        ax01.set_extent(ccrs_views[mode]['view_extent'], proj_data)
        ax01.set_aspect(1.25)
        ##############################################################

        fig.subplots_adjust(wspace=0.1)
        fig.savefig(full_fname, dpi=100, pad_inches=0.15, bbox_inches="tight")
        plt.close()
        return 1
    ############################################################################################################################

################################################################################################################

proj_data = ccrs.PlateCarree()
cfs_alert = (-62.3167, 82.5) # Station Alert
stn_nord  = (-16.6667, 81.6) # Station Nord
thule_pituffik = (-68.703056, 76.531111) # Pituffik Space Base

ccrs_views =        {'lincoln': {'view_extent': [-130, 50, 76, 89],
                                'vlon':         -40,
                                'vlat':          84},
                    'platypus': {'view_extent': [-140, -30, 75.5, 89.5],
                                'vlon':         -70,
                                'vlat':          84},
                    'canada':   {'view_extent': [-140, -30, 75.5, 89.5],
                                'vlon':         -70,
                                'vlat':          84},
                    'ca_archipelago':   {'view_extent': [-140, -30, 75.5, 89.5],
                                        'vlon':         -70,
                                        'vlat':          84},
                    'baffin':    {'view_extent': [-130, -40, 67, 84],
                                'vlon':         -60,
                                'vlat':          84},
                }
