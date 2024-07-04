import os
import sys
import argparse
import h5py
import numpy as np
from arctic_gridding_utils import modis_l1b, modis_03, modis_l2, viirs_l1b, viirs_03
import datetime
from tqdm import tqdm
from imagery import Imagery

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

START_TIME = datetime.datetime.now()
EARTH_RADIUS = 6371.009
SZA_LIMIT = 81.36

utf8_type = h5py.string_dtype('utf-8', 30) # for string encoding in HDF5

def get_swath_area(lon, lat):
    swath_area = np.abs(np.abs(lon.max()) - np.abs(lon.min())) * np.abs(np.abs(lat.max()) - np.abs(lat.min()))
    return swath_area

# def calc_lwp(cot, cer, rho_water=1000.):
#     return 2. * cot * cer * 1e-6 * rho_water * 1e3 / 3.



def within_range(fname, start_dt, end_dt):
    fname = os.path.basename(fname)
    yyyydoy, hhmm = fname.split('.')[1], fname.split('.')[2]
    yyyydoy = yyyydoy[1:] # the first character is "A"

    sat_dt = datetime.datetime.strptime(yyyydoy + hhmm, "%Y%j%H%M")

    if start_dt <= sat_dt <= end_dt:
        return True
    else:
        return False


def get_last_opass(sorted_geofiles):

    last_opass_fname = os.path.basename(sorted_geofiles[-1])

    yyyydoy, hhmm = last_opass_fname.split('.')[1], last_opass_fname.split('.')[2]
    yyyydoy = yyyydoy[1:] # the first character is "A"

    last_opass_dt = datetime.datetime.strptime(yyyydoy + hhmm, "%Y%j%H%M")
    return last_opass_dt


def get_satellite_group_name(acq_dt, geo_file, encode=False):
    # satellite name
    if os.path.basename(geo_file)[:3] == "MOD":
        satellite  = "Terra"
        group_name = "T" + acq_dt[1:]
    elif os.path.basename(geo_file)[:3] == "MYD":
        satellite  = "Aqua"
        group_name = "A" + acq_dt[1:]
    elif os.path.basename(geo_file)[:3] == "VNP":
        satellite  = "Suomi-NPP"
        group_name = "S" + acq_dt[1:]
    elif os.path.basename(geo_file)[:3] == "VJ1":
        satellite  = "NOAA-20/JPSS-1"
        group_name = "J" + acq_dt[1:]
    elif os.path.basename(geo_file)[:3] == "VJ2":
        satellite  = "NOAA-21/JPSS-2"
        group_name = "N" + acq_dt[1:]
    else:
        satellite  = "Unknown"
        group_name = "Z" + acq_dt[1:]

    if encode:
        satellite = np.array(satellite.encode("utf-8"), dtype=utf8_type) # encode string for HDF5
    return satellite, group_name
################################################################################################################

def get_all_modis_files(fdir):
    return sorted([f for f in os.listdir(fdir) if f.lower().endswith('.hdf')])

def get_all_viirs_files(fdir):
    return sorted([f for f in os.listdir(fdir) if f.lower().endswith('.nc')])

def get_product_info(fnames, product_id):
    p_fnames, acq_dts = [], []
    for f in fnames:
        if product_id.upper() in f.upper():
            p_fnames.append(f)
            acq_dts.append(f.split(".")[1] + '.' + f.split(".")[2])
    return p_fnames, acq_dts



def get_viirs_ref_geo(fdir):

    # VIIRS files
    viirs_fnames = get_all_viirs_files(fdir)
    _, acq_dts_viirs_fref = get_product_info(viirs_fnames, product_id="02MOD")
    _, acq_dts_viirs_f03 = get_product_info(viirs_fnames, product_id="03MOD")

    viirs_acq_dts_common = list(set.intersection(*map(set,
                                                 [acq_dts_viirs_fref,
                                                  acq_dts_viirs_f03])))

    if len(viirs_acq_dts_common) == 0:
        raise IndexError("Could not find any common date/times among products")

    fnames = viirs_fnames

    fref, f03, = [], []

    for f in fnames:
        filename = os.path.basename(f).upper()
        # acq_dt = filename.split(".")[1] + '.' + filename.split(".")[2]
        pname  = filename.split(".")[0]


        if "02MOD" in pname:
            fref.append(os.path.join(fdir, f))

        elif "03MOD" in pname:
            f03.append(os.path.join(fdir, f))

        else:
            pass

    fref = sorted(fref, key=lambda x: x.split('.')[2])
    f03  = sorted(f03, key=lambda x: x.split('.')[2])
    return fref, f03



################################################################################################################

def get_modis_noref_geo_cld_opt(fdir):

    # MODIS files
    modis_fnames = get_all_modis_files(fdir)
    if len(modis_fnames) == 0:
        print("No MODIS files found in: ", fdir)
        return [], []
    else:
        _, acq_dts_ft03 = get_product_info(modis_fnames, product_id="MOD03")
        _, acq_dts_fa03 = get_product_info(modis_fnames, product_id="MYD03")
        acq_dts_modis_f03 = acq_dts_ft03 + acq_dts_fa03
        # modis_f03 = temp1 + temp2
        _, acq_dts_modis_f06_l2 = get_product_info(modis_fnames, product_id="06_L2")

    modis_acq_dts_common = list(set.intersection(*map(set,
                                                  [acq_dts_modis_f03,
                                                  acq_dts_modis_f06_l2
                                                  ])))

    if len(modis_acq_dts_common) == 0:
        print("IndexError: Could not find any common date/times among products")
        return [], []

    fnames = modis_fnames
    # acq_dts_common = modis_acq_dts_common + viirs_acq_dts_common
    # print(len(fnames), len(acq_dts_common))
    f03, fcld_l2 = [], []

    for f in fnames:
        filename = os.path.basename(f).upper()
        acq_dt = filename.split(".")[1] + '.' + filename.split(".")[2]
        pname  = filename.split(".")[0]

        if acq_dt in modis_acq_dts_common:

            if ("MOD03" in pname) or ("MYD03" in pname):
                f03.append(os.path.join(fdir, f))

            elif "06_L2" in pname:
                fcld_l2.append(os.path.join(fdir, f))

            else:
                pass

    f03  = sorted(f03, key=lambda x: x.split('.')[2])
    fcld_l2 = sorted(fcld_l2, key=lambda x: x.split('.')[2])
    return f03, fcld_l2


def save_to_file_modis_only_geo_cld_opt(fdir, outdir, extent, metadata, geojson_fpath, buoys, start_dt, end_dt, mode):

    f03, fcld_l2 = get_modis_noref_geo_cld_opt(fdir)
    if (len(f03) == 0) or (len(fcld_l2) == 0):
        print("\nMessage [visualize_satellites]: Could not find any common ref + cld products for MODIS\n")
        return 0

    n_obs = len(f03)

    print("Found {} overpasses".format(n_obs))
    # ext_area = get_swath_area(lon_1d, lat_1d)

    valid_count = 0 # just for reporting

    start_dt_str = start_dt.strftime('%Y-%m-%d-%H%M')
    end_dt_str   = end_dt.strftime('%Y-%m-%d-%H%M')
    # outdir_dt =  start_dt_str + '_' + end_dt_str

    # outdir = os.path.join(outdir, outdir_dt)
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)
    l2_dirs = ['water_path', 'ice_path',  'optical_thickness', 'cloud_phase', 'cloud_top_height_temperature']
    exist_acq_dts = []
    sub_outdirs = [f for f in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, f))]
    for sub in sub_outdirs:
        if sub in l2_dirs:
            for f in os.listdir(os.path.join(outdir, sub)):
                if f.endswith('.png'):
                    exist_acq_dt = datetime.datetime.strptime(''.join(f.split('_')[:-1]), '%Y-%m-%d-%H%MZ')
                    exist_acq_dts.append(exist_acq_dt.strftime("%Y%j.%H%M"))

    # purposely change start date to account for latency but not affect filenames
    start_dt = start_dt - datetime.timedelta(hours=2)
    start_dt_str = start_dt.strftime('%Y-%m-%d-%H%M')
    print("Message [visualize_satellites]: Start datetime:{}, End datetime: {}".format(start_dt_str, end_dt_str))
    print("Warning [visualize_satellites]: Start datetime was changed to catch more data and account for latency")

    for i in tqdm(range(n_obs)):
        try:

            # geometries/geolocation file
            geo_file    = f03[i]

            # check if the file has already been processed
            bgeo_file = os.path.basename(geo_file)
            yyyydoy_hhmm = bgeo_file.split('.')[1][1:] + '.' + bgeo_file.split('.')[2]

            # print("Message [modis_cld_geo]: yyyydoy", yyyydoy)
            if yyyydoy_hhmm in exist_acq_dts: # if already processed, then skip
                print("Message [modis_cld_geo]: Skipping {} as it has likely already been processed previously".format(geo_file))
                continue


            if not within_range(geo_file, start_dt, end_dt):
                print("Message [modis_cld_geo]: Skipping {} as it is outside the provided date range: {} to {}".format(geo_file, start_dt.strftime("%Y-%m-%d:%H%M"), end_dt.strftime("%Y-%m-%d:%H%M")))
                continue

            cld_file    = fcld_l2[i]

            acq_dt = os.path.basename(geo_file).split('.')[1] + '.' + os.path.basename(geo_file).split('.')[2]

            if os.path.basename(geo_file).upper().startswith('M'): # modis
                f_geo = modis_03(fnames=[geo_file], extent=extent, keep_dims=True)

        # except HDF4Error:
        #     print("PyHDF error with {}...skipping...\n".format(geo_file))
        #     continue
        # except FileNotFoundError:
        #     print("netcdf error with {}...skipping...\n".format(geo_file))
        #     continue
        except Exception as err:
            print("some error with {}...skipping...\nError {}\n".format(geo_file, err))
            continue

        lon2d_1km = f_geo.data['lon']['data'].T
        lat2d_1km = f_geo.data['lat']['data'].T
        if len(lon2d_1km) == 0 or len(lat2d_1km) == 0:
            # print('Lat/lon not valid')
            continue

        try:
            # cloud product
            if os.path.basename(cld_file).upper().startswith(('MOD06_L2', 'MYD06_L2')): # modis
                f_cld = modis_l2(fnames=[cld_file], f03=f_geo, extent=extent, keep_dims=True)

            lon2d_5km = f_cld.data['lon_5km']['data'].T
            lat2d_5km = f_cld.data['lat_5km']['data'].T

            ctp      = f_cld.data['ctp']['data'].T
            cot_2d   = f_cld.data['cot']['data'].T
            cwp_2d   = f_cld.data['cwp']['data'].T
            cot_1621 = f_cld.data['cot_1621']['data'].T
            cwp_1621 = f_cld.data['cwp_1621']['data'].T
            ctp_ir   = f_cld.data['ctp_ir']['data'].T
            cth      = f_cld.data['cth']['data'].T
            ctt      = f_cld.data['ctt']['data'].T

            # satellite name
            satellite, group_name = get_satellite_group_name(acq_dt, geo_file, encode=False)
            # satellite = np.array(satellite.encode("utf-8"), dtype=utf8_type)

            arcsix_l2 = Imagery() # initialize class object

            _ = arcsix_l2.plot_liquid_water_paths(lon_2d=lon2d_1km, lat_2d=lat2d_1km, ctp=ctp, cwp=cwp_2d, cwp_1621=cwp_1621,
                                                  satellite=satellite, acq_dt=group_name, outdir=outdir, geojson_fpath=geojson_fpath,
                                                  buoys=buoys, mode=mode)
            _ = arcsix_l2.plot_ice_water_paths(lon_2d=lon2d_1km, lat_2d=lat2d_1km, ctp=ctp, cwp=cwp_2d, cwp_1621=cwp_1621,
                                               satellite=satellite, acq_dt=group_name, outdir=outdir, geojson_fpath=geojson_fpath,
                                               buoys=buoys, mode=mode)
            _ = arcsix_l2.plot_optical_depths(lon_2d=lon2d_1km, lat_2d=lat2d_1km, cot=cot_2d, cot_1621=cot_1621,
                                              satellite=satellite, acq_dt=group_name, outdir=outdir, geojson_fpath=geojson_fpath,
                                              buoys=buoys, mode=mode)
            _ = arcsix_l2.plot_cloud_phase(lon_2d=lon2d_1km, lat_2d=lat2d_1km, ctp_swir=ctp, ctp_ir=ctp_ir,
                                           satellite=satellite, acq_dt=group_name, outdir=outdir, geojson_fpath=geojson_fpath,
                                           buoys=buoys, mode=mode)
            _ = arcsix_l2.plot_cloud_top(lon_2d=lon2d_5km, lat_2d=lat2d_5km, cth=cth, ctt=ctt,
                                         satellite=satellite, acq_dt=group_name, outdir=outdir, geojson_fpath=geojson_fpath,
                                         buoys=buoys, mode=mode)

            # save data
            # g1 = f1.create_group(group_name)
            # _ = g1.create_dataset('ctp',         data=sdata['ctp'].T, compression='gzip', compression_opts=9)
            # _ = g1.create_dataset('ctp_ir',      data=sdata['ctp_ir'].T, compression='gzip', compression_opts=9)
            # _ = g1.create_dataset('ctt',         data=sdata['ctt'].T, compression='gzip', compression_opts=9)
            # _ = g1.create_dataset('cth',         data=sdata['cth'].T, compression='gzip', compression_opts=9)
            # _ = g1.create_dataset('cot_2d',      data=sdata['cot_2d'].T, compression='gzip', compression_opts=9)
            # _ = g1.create_dataset('cwp_2d',      data=sdata['cwp_2d'].T, compression='gzip', compression_opts=9)
            # _ = g1.create_dataset('cot_1621',    data=sdata['cot_1621'].T, compression='gzip', compression_opts=9)
            # _ = g1.create_dataset('cwp_1621',    data=sdata['cwp_1621'].T, compression='gzip', compression_opts=9)
            # _ = g1.create_dataset('lon2d_1km',   data=lon2d_1km.T, compression='gzip', compression_opts=9)
            # _ = g1.create_dataset('lat2d_1km',   data=lat2d_1km.T, compression='gzip', compression_opts=9)
            # _ = g1.create_dataset('lon2d_5km',   data=lon2d_5km.T, compression='gzip', compression_opts=9)
            # _ = g1.create_dataset('lat2d_5km',   data=lat2d_5km.T, compression='gzip', compression_opts=9)
            # # _ = g1.create_dataset('lon_1d',      data=lon_1d, compression='gzip', compression_opts=9)
            # # _ = g1.create_dataset('lat_1d',      data=lat_1d, compression='gzip', compression_opts=9)
            # _ = g1.create_dataset('metadata',    data=metadata)
            # _ = g1.create_dataset('satellite',   data=satellite)

            valid_count += 1

        except Exception as err:
            print(err)
            continue

    if valid_count > 0:
        return 1
    else:
        return 0



def get_modis_viirs_ref_geo(fdir):

    # MODIS files
    modis_fnames = get_all_modis_files(fdir)
    if len(modis_fnames) == 0:
        modis_acq_dts_common = []
    else:
        _, acq_dts_modis_fref = get_product_info(modis_fnames, product_id="1KM")
        _, acq_dts_ft03 = get_product_info(modis_fnames, product_id="MOD03")
        _, acq_dts_fa03 = get_product_info(modis_fnames, product_id="MYD03")
        acq_dts_modis_f03 = acq_dts_ft03 + acq_dts_fa03
        # modis_f03 = temp1 + temp2

        modis_acq_dts_common = list(set.intersection(*map(set,
                                                    [acq_dts_modis_fref,
                                                    acq_dts_modis_f03])))

    # if len(modis_acq_dts_common) == 0:
    #     raise IndexError("Could not find any common date/times among products")

    # VIIRS files
    viirs_fnames = get_all_viirs_files(fdir)
    if len(viirs_fnames) == 0:
        viirs_acq_dts_common = []
    else:
        _, acq_dts_viirs_fref = get_product_info(viirs_fnames, product_id="02MOD")
        _, acq_dts_viirs_f03 = get_product_info(viirs_fnames, product_id="03MOD")

        viirs_acq_dts_common = list(set.intersection(*map(set,
                                                    [acq_dts_viirs_fref,
                                                    acq_dts_viirs_f03])))


    if (len(viirs_acq_dts_common) == 0) and (len(modis_acq_dts_common) == 0):
        print("IndexError: Could not find any common date/times among products")
        return [], []

    fnames = modis_fnames + viirs_fnames

    fref, f03, = [], []

    for f in fnames:
        filename = os.path.basename(f).upper()
        acq_dt = filename.split(".")[1] + '.' + filename.split(".")[2]
        pname  = filename.split(".")[0]

        # use separate if statements to avoid discrepancy in file numbers and names

        if acq_dt in modis_acq_dts_common:
            if "1KM" in pname:
                fref.append(os.path.join(fdir, f))

            elif ("MOD03" in pname) or ("MYD03" in pname):
                f03.append(os.path.join(fdir, f))

            else:
                pass

        elif acq_dt in viirs_acq_dts_common:

            if "02MOD" in pname:
                fref.append(os.path.join(fdir, f))

            elif "03MOD" in pname:
                f03.append(os.path.join(fdir, f))

            else:
                pass

    fref = sorted(fref, key=lambda x: x.split('.')[2])
    f03  = sorted(f03,  key=lambda x: x.split('.')[2])
    return fref, f03


def save_to_file_modis_viirs_ref_geo(fdir, outdir, extent, metadata, geojson_fpath, buoys, start_dt, end_dt, mode):

    fref, f03 = get_modis_viirs_ref_geo(fdir)
    if (len(fref) == 0) or (len(f03) == 0):
        print("Message [visualize_satellites]: Could not find any common products for both MODIS and VIIRS")
        return 0

    n_obs = len(f03)

    print("Found {} overpasses".format(n_obs))

    valid_count = 0 # just for reporting

    start_dt_str = start_dt.strftime('%Y-%m-%d-%H%M')
    end_dt_str   = end_dt.strftime('%Y-%m-%d-%H%M')
    print("Message [modis_viirs_ref_geo]: Start datetime:{}, End datetime: {}".format(start_dt_str, end_dt_str))

    img_dirs = ['true_color', 'false_color_721', 'false_color_367', 'false_color_ir', 'false_color_cirrus']
    exist_acq_dts = []
    sub_outdirs = [f for f in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, f))]

    if len(sub_outdirs) > 0:
        for sub in sub_outdirs:
            if sub in img_dirs:
                for f in os.listdir(os.path.join(outdir, sub)):
                    if f.endswith('.png'):
                        exist_acq_dt = datetime.datetime.strptime(''.join(f.split('_')[:-1]), '%Y-%m-%d-%H%MZ')
                        exist_acq_dts.append(exist_acq_dt.strftime("%Y%j.%H%M"))

    for i in tqdm(range(n_obs)):
        try:
            # geometries/geolocation file
            geo_file    = f03[i]

            # check if the file has already been processed
            bgeo_file = os.path.basename(geo_file)
            yyyydoy_hhmm = bgeo_file.split('.')[1][1:] + '.' + bgeo_file.split('.')[2]

            # print("Message [modis_viirs_ref_geo]: yyyydoy", yyyydoy)
            if yyyydoy_hhmm in exist_acq_dts: # if already processed, then skip
                print("Message [modis_viirs_ref_geo]: Skipping {} as it has likely already been processed previously".format(geo_file))
                continue


            ########################################################################################################################
            # # add an extra hour of latency if necessary as they can sometimes be slow
            if os.path.basename(geo_file).startswith('V'):

                if (end_dt - start_dt) < datetime.timedelta(hours=2):
                    viirs_start_dt = start_dt - datetime.timedelta(hours=1)
                elif ((end_dt - start_dt) > datetime.timedelta(hours=3)):
                    viirs_start_dt = end_dt - datetime.timedelta(hours=3)
                    print("Message [visualize_satellites]: Start time and end times were too far apart...limiting to 3 hour gap.")
                else:
                    viirs_start_dt = start_dt

                if not within_range(geo_file, viirs_start_dt, end_dt):
                    # print("Message [modis_viirs_ref_geo]: VIIRS: Skipping {} as it is outside the provided date range: {} to {}".format(geo_file, viirs_start_dt.strftime("%Y-%m-%d:%H%M"), end_dt.strftime("%Y-%m-%d:%H%M")))
                    continue

            else: # for MODIS keep latency as is
                if not within_range(geo_file, start_dt, end_dt):
                    # print("Message [modis_viirs_ref_geo]: MODIS: Skipping {} as it is outside the provided date range: {} to {}".format(geo_file, start_dt.strftime("%Y-%m-%d:%H%M"), end_dt.strftime("%Y-%m-%d:%H%M")))
                    continue

            ########################################################################################################################

            ref_file    = fref[i]
            acq_dt = os.path.basename(geo_file).split('.')[1] + '.' + os.path.basename(geo_file).split('.')[2]

            if os.path.basename(geo_file).upper().startswith('M'): # modis
                f_geo = modis_03(fnames=[geo_file], extent=extent, keep_dims=True)
            elif os.path.basename(geo_file).upper().startswith('V'): # viirs
                f_geo = viirs_03(fnames=[geo_file], extent=extent, keep_dims=True)
            else:
                raise NotImplementedError("\nOnly VIIRS and MODIS products are supported\n")
        except Exception as err:
            print("some error: with {}...skipping...\nError: {}\n".format(geo_file, err))
            continue

        lon2d_1km = f_geo.data['lon']['data'].T
        lat2d_1km = f_geo.data['lat']['data'].T
        if len(lon2d_1km) == 0 or len(lat2d_1km) == 0:
            # print('Lat/lon not valid')
            continue

        sza_2d = f_geo.data['sza']['data'].T
        sza_2d[sza_2d > SZA_LIMIT] = SZA_LIMIT

        # reflectance file
        try:
            if os.path.basename(ref_file).upper().startswith('M'): # modis
                f_vis = modis_l1b(fnames=[ref_file],
                                    extent=extent,
                                    f03=f_geo,
                                    keep_dims=True,
                                    bands=[1, 2, 3, 4, 6, 7, 26, 31])

            elif os.path.basename(ref_file).upper().startswith('V'): # viirs
                f_vis = viirs_l1b(fnames=[ref_file],
                                    extent=extent,
                                    f03=f_geo,
                                    keep_dims=True,
                                    bands=["M05", "M07", "M02", "M04", "M10", "M11", "M09", "M15"])
            else:
                print("\nOnly VIIRS and MODIS products are supported. Instead, file was: {}\n".format(ref_file))
                continue

            ref_650  = f_vis.data['ref']['data'][0].T
            ref_860  = f_vis.data['ref']['data'][1].T
            ref_470  = f_vis.data['ref']['data'][2].T
            ref_555  = f_vis.data['ref']['data'][3].T
            ref_1640 = f_vis.data['ref']['data'][4].T
            ref_2130 = f_vis.data['ref']['data'][5].T
            ref_1380 = f_vis.data['ref']['data'][6].T

            rad_1640  = f_vis.data['rad']['data'][4].T
            rad_2130  = f_vis.data['rad']['data'][5].T
            rad_11000 = f_vis.data['rad']['data'][7].T


            # satellite name
            satellite, group_name = get_satellite_group_name(acq_dt, geo_file, encode=False)
            # satellite = np.array(satellite.encode("utf-8"), dtype=utf8_type)

            arcsix_imagery = Imagery() # initialize class object

            _ = arcsix_imagery.create_true_color_imagery(lon_2d=lon2d_1km, lat_2d=lat2d_1km, red=ref_650, green=ref_555, blue=ref_470, sza=sza_2d, satellite=satellite, acq_dt=group_name, outdir=outdir, geojson_fpath=geojson_fpath, buoys=buoys, mode=mode)

            _ = arcsix_imagery.create_false_color_721_imagery(lon_2d=lon2d_1km, lat_2d=lat2d_1km, red=ref_2130, green=ref_860, blue=ref_650, sza=sza_2d, satellite=satellite, acq_dt=group_name, outdir=outdir, geojson_fpath=geojson_fpath, buoys=buoys, mode=mode)

            if satellite != 'Aqua': # band 6 at 1640 is broken for Aqua

                _ = arcsix_imagery.create_false_color_367_imagery(lon_2d=lon2d_1km, lat_2d=lat2d_1km, red=ref_470, green=ref_1640, blue=ref_2130, sza=sza_2d, satellite=satellite, acq_dt=group_name, outdir=outdir, geojson_fpath=geojson_fpath, buoys=buoys, mode=mode)

                # 1.38-1.6-2.1 reflectance
                _ = arcsix_imagery.create_false_color_cirrus_imagery(lon_2d=lon2d_1km, lat_2d=lat2d_1km, red=ref_1380, green=ref_1640, blue=ref_2130, sza=sza_2d, satellite=satellite, acq_dt=group_name, outdir=outdir, geojson_fpath=geojson_fpath, buoys=buoys, mode=mode)

                # 11micron-1.6-2.1 radiance
                _ = arcsix_imagery.create_false_color_ir_imagery(lon_2d=lon2d_1km, lat_2d=lat2d_1km, red=rad_11000, green=rad_1640, blue=rad_2130, satellite=satellite, acq_dt=group_name, outdir=outdir, geojson_fpath=geojson_fpath, buoys=buoys, mode=mode)

            valid_count += 1

        except Exception as err:
            print(err)
            continue

    if valid_count > 0:
        return 1
    else:
        return 0

########################################################################################################################


def get_metadata(fdir):
    """
    Get date and extent from metadata file
    """
    with open(os.path.join(fdir, "metadata.txt"), "r") as f:
        meta = f.readlines()

    extent_str = meta[1][9:-2]
    extent     = [float(idx) for idx in extent_str.split(', ')]
    ymd = meta[0][6:-1]
    dt = datetime.datetime.strptime(ymd, '%Y-%m-%d')

    return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, extent


def get_start_end_dates_metadata(fdir):
    """
    Get start and end from metadata file
    """
    with open(os.path.join(fdir, "metadata.txt"), "r") as f:
        meta = f.readlines()

    start_dt = meta[-2][12:-1]
    end_dt   = meta[-1][12:-1]

    start_dt = datetime.datetime.strptime(start_dt, '%Y-%m-%d-%H%M')
    end_dt = datetime.datetime.strptime(end_dt, '%Y-%m-%d-%H%M')

    return start_dt, end_dt


def get_extent(fdir):
    with open(os.path.join(fdir, "metadata.txt"), "r") as f:
        meta = f.readlines()

    extent_str = meta[1][9:-2]
    extent     = [float(idx) for idx in extent_str.split(', ')]
    return extent


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir",  default=None,  type=str, help="Path to directory containing raw MODIS/VIIRS HDF files")
    parser.add_argument("--outdir", type=str, default=None, help="Path to directory where images will be saved")
    parser.add_argument("--start_date", type=str, metavar='', default=None,
                        help='The start date of the range of dates for which you would like to grid data. '\
                        'Use yyyymmdd or yyyymmddhhmm format.\n'\
                        'Example: --start_date 20210404\n \n')
    parser.add_argument("--end_date", type=str, metavar='', default=None,
                        help='The end date of the range of dates for which you would like to grid data. '\
                        'Use yyyymmdd or yyyymmddhhmm format.\n'\
                        'Example: --end_date 20210414\n \n')
    parser.add_argument('--ndir_recent', type=int, metavar='', default=None, help='Time range in terms of sub-directories. For example, if --ndir_recent 5, then the 5 most recent (based on filenames) datetime ranges will be used to create data')
    parser.add_argument('--max_hours', type=int, metavar='', default=None, help='Maximum time range in hours. For example, if --max_hours 5, then the 5 most recent hours from the last available satellite time are used')
    parser.add_argument("--nrt", action='store_true', help="Enable --nrt to process VIIRS L1b and 03 products, and MODIS L1b, 03, and clds")
    parser.add_argument('--geojson', type=str, metavar='',
                        help='Path to a geoJSON file containing the extent of interest coordinates\n'\
                        'Example:  --geojson my/path/to/geofile.json\n \n')
    parser.add_argument('--buoys', type=str, metavar='', default=None, help='Path to the JSON file containing URLs to the buoys csv data')
    parser.add_argument('--mode', type=str, metavar='', default=None, help='One of "baffin", "lincoln", or "platypus" ')
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    manual_list = []

    # subdirs will only be yyyy-mm-dd (or yyyy_mm_dd in older sdown versions)
    # subdirs = sorted([os.path.join(args.fdir, f) for f in os.listdir(args.fdir) if os.path.isdir(os.path.join(args.fdir, f))])
    subdirs = []
    for f in os.listdir(args.fdir):
        if os.path.isdir(os.path.join(args.fdir, f)) and (len(os.listdir(os.path.join(args.fdir, f))) > 2): # metadata + two files minimum
            subdirs.append(os.path.join(args.fdir, f))

    subdirs = sorted(subdirs)

    if args.start_date is None or args.end_date is None:

            # start_dt_hhmm, end_dt_hhmm = get_start_end_dates_metadata(fdir)
            start_dt_hhmm, end_dt_hhmm = get_start_end_dates_metadata(subdirs[-1])
            # start_dt_hhmm = end_dt_hhmm - datetime.timedelta(hours=args.max_hours)
            if ((end_dt_hhmm - start_dt_hhmm) > datetime.timedelta(hours=3)):
                start_dt_hhmm = end_dt_hhmm - datetime.timedelta(hours=3)
                print("Message [visualize_satellites]: Start time and end times were too far apart...limiting to 3 hour gap.")

    elif args.start_date is not None and args.end_date is not None:
        if args.max_hours is None:
            # start at 0 UTC unless specified by user
            start_hr = 0
            start_min = 0
            if len(args.start_date) == 12:
                start_hr, start_min = int(args.start_date[8:10]), int(args.start_date[10:12])

            start_dt_hhmm  = datetime.datetime(int(args.start_date[:4]), int(args.start_date[4:6]), int(args.start_date[6:8]), start_hr, start_min)

            # end at 2359 UTC unless specified by user
            end_hr = 23
            end_min = 59
            if len(args.end_date) == 12:
                end_hr, end_min = int(args.end_date[8:10]), int(args.end_date[10:12])

            end_dt_hhmm  = datetime.datetime(int(args.end_date[:4]), int(args.end_date[4:6]), int(args.end_date[6:8]), end_hr, end_min)

        else:
            _, end_dt_hhmm = get_start_end_dates_metadata(subdirs[-1])
            start_dt_hhmm = end_dt_hhmm - datetime.timedelta(hours=args.max_hours)

    else:
        print("Something went wrong with start and end dates.\n")
        sys.exit()


    start_dt_hhmm_str = start_dt_hhmm.strftime('%Y-%m-%d-%H%M')
    end_dt_hhmm_str   = end_dt_hhmm.strftime('%Y-%m-%d-%H%M')

    outdir_dt =  start_dt_hhmm_str + '_' + end_dt_hhmm_str

    print("=====================================================================")
    print("Message [visualize_satellites]: Start datetime: {}, End datetime: {}".format(start_dt_hhmm_str, end_dt_hhmm_str))
    print("=====================================================================")

    if (args.ndir_recent is not None) and (len(subdirs) >= args.ndir_recent):
        subdirs = subdirs[-args.ndir_recent:]

    print("Message [visualize_satellites]: {} sub-directories will be analyzed".format(len(subdirs)))

    for fdir in tqdm(subdirs):
        dt = os.path.basename(fdir)
        print("Currently analyzing:", dt) # date
        year, month, date, hour, minute, sec, extent = get_metadata(fdir)

        outdir = args.outdir

        metadata = np.array([year, month, date, hour, minute, sec, extent[0], extent[1], extent[2], extent[3]])
        # lon_1d, lat_1d = calc_lonlat_arr(west=extent[0], south=extent[2],
        #                                 width=args.width, height=args.height,
        #                                 resolution=args.resolution)

        ret = 0
        if args.nrt:
            ret += save_to_file_modis_viirs_ref_geo(fdir, outdir, extent, metadata, geojson_fpath=args.geojson, buoys=args.buoys, start_dt=start_dt_hhmm, end_dt=end_dt_hhmm, mode=args.mode)
            ret += save_to_file_modis_only_geo_cld_opt(fdir, outdir, extent, metadata, geojson_fpath=args.geojson, buoys=args.buoys, start_dt=start_dt_hhmm, end_dt=end_dt_hhmm, mode=args.mode)

        if ret == 0:
            manual_list.append(fdir)
            continue


    print("Missed: ", manual_list)
    print("Finished!")

    END_TIME = datetime.datetime.now()
    print('Time taken to execute:', END_TIME - START_TIME)
