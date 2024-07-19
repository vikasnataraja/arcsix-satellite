import os
import sys
import argparse
import numpy as np
from util.arctic_gridding_utils import modis_l1b, modis_03, modis_l2, viirs_l1b, viirs_03, viirs_cldprop_l2
from util.arctic_gridding_utils import within_range, get_satellite_group_name
import util.constants
import datetime
from tqdm import tqdm
from imagery import Imagery

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# moved to util.constants; will be removed at a later version
# EARTH_RADIUS = 6371.009
# SZA_LIMIT = 81.36


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

    # remove duplicates if any based on acq_dts
    seen_ref = set()
    fref = [x for x in fref if not (os.path.basename(x).split('.')[1] + '.' + os.path.basename(x).split('.')[2] in seen_ref or seen_ref.add(os.path.basename(x).split('.')[1] + '.' + os.path.basename(x).split('.')[2]))]

    seen_geo = set()
    f03 = [x for x in f03 if not (os.path.basename(x).split('.')[1] + '.' + os.path.basename(x).split('.')[2] in seen_geo or seen_geo.add(os.path.basename(x).split('.')[1] + '.' + os.path.basename(x).split('.')[2]))]


    fref = sorted(fref, key=lambda x: x.split('.')[2])
    f03  = sorted(f03, key=lambda x: x.split('.')[2])
    return fref, f03



################################################################################################################

# Deprecated; Will be removed in a future version
"""
def get_modis_geo_cld_opt(fdir):

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


def save_to_file_modis_only_geo_cld_opt(fdir, outdir, extent, geojson_fpath, buoys, norway_ship, start_dt, end_dt, quicklook_fdir, mode):

    f03, fcld_l2 = get_modis_geo_cld_opt(fdir)
    if (len(f03) == 0) or (len(fcld_l2) == 0):
        print("\nMessage [modis_cld_geo]: Could not find any common ref + cld products for MODIS\n")
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
    print("Message [modis_cld_geo]: Start datetime:{}, End datetime: {}".format(start_dt_str, end_dt_str))
    print("Warning [modis_cld_geo]: Start datetime was changed to catch more data and account for latency")

    for i in tqdm(range(n_obs)):
        try:

            # geometries/geolocation file
            geo_file    = f03[i]

            # check if the file has already been processed
            bgeo_file = os.path.basename(geo_file)
            yyyydoy_hhmm = bgeo_file.split('.')[1][1:] + '.' + bgeo_file.split('.')[2]

            print("Message [modis_cld_geo]: Processing: ", bgeo_file)
            if yyyydoy_hhmm in exist_acq_dts: # if already processed, then skip
                print("Message [modis_cld_geo]: Skipping {} as it has likely already been processed previously".format(bgeo_file))
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
            print("Error [modis_cld_geo]: some error with {}...skipping...\nError {}\n".format(geo_file, err))
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
            if 'uwssec' in geo_file.lower():
                data_source = 'UWisc SSEC'
            else:
                data_source = 'NASA LANCE DAAC'

            arcsix_l2 = Imagery(data_source=data_source,
                                satellite=satellite,
                                acq_dt=group_name,
                                outdir=outdir,
                                geojson_fpath=geojson_fpath,
                                buoys=buoys,
                                norway_ship=norway_ship,
                                quicklook_fdir=quicklook_fdir,
                                mode=mode) # initialize class object

            _ = arcsix_l2.plot_liquid_water_paths(lon_2d=lon2d_1km, lat_2d=lat2d_1km, ctp=ctp, cwp=cwp_2d, cwp_1621=cwp_1621)
            _ = arcsix_l2.plot_ice_water_paths(lon_2d=lon2d_1km, lat_2d=lat2d_1km, ctp=ctp, cwp=cwp_2d, cwp_1621=cwp_1621)
            _ = arcsix_l2.plot_optical_depths(lon_2d=lon2d_1km, lat_2d=lat2d_1km, cot=cot_2d, cot_1621=cot_1621)
            _ = arcsix_l2.plot_cloud_phase(lon_2d=lon2d_1km, lat_2d=lat2d_1km, ctp_swir=ctp, ctp_ir=ctp_ir)
            _ = arcsix_l2.plot_cloud_top(lon_2d=lon2d_5km, lat_2d=lat2d_5km, cth=cth, ctt=ctt)

            valid_count += 1

        except Exception as err:
            print(err)
            continue

    if valid_count > 0:
        return 1
    else:
        return 0

"""
################################################################################################################

def get_modis_viirs_geo_cld_opt(fdir):

    # MODIS files
    modis_fnames = get_all_modis_files(fdir)
    if len(modis_fnames) == 0:
        modis_acq_dts_common = []
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

    # VIIRS files
    viirs_fnames = get_all_viirs_files(fdir)
    if len(viirs_fnames) == 0:
        viirs_acq_dts_common = []
    else:
        _, acq_dts_viirs_fcld = get_product_info(viirs_fnames, product_id="CLDPROP")
        _, acq_dts_viirs_f03  = get_product_info(viirs_fnames, product_id="03MOD")

        viirs_acq_dts_common = list(set.intersection(*map(set,
                                                    [acq_dts_viirs_fcld,
                                                    acq_dts_viirs_f03])))

    if (len(viirs_acq_dts_common) == 0) and (len(modis_acq_dts_common) == 0):
        print("IndexError: Could not find any common date/times among products")
        return [], []

    fnames = modis_fnames + viirs_fnames

    fcld, f03, = [], []

    for f in fnames:
        filename = os.path.basename(f).upper()
        acq_dt = filename.split(".")[1] + '.' + filename.split(".")[2]
        pname  = filename.split(".")[0]

        # use separate if statements to avoid discrepancy in file numbers and names

        if acq_dt in modis_acq_dts_common:
            if "06_L2" in pname:
                fcld.append(os.path.join(fdir, f))

            elif ("MOD03" in pname) or ("MYD03" in pname):
                f03.append(os.path.join(fdir, f))

            else:
                pass

        elif acq_dt in viirs_acq_dts_common:

            if "CLDPROP" in pname:
                fcld.append(os.path.join(fdir, f))

            elif "03MOD" in pname:
                f03.append(os.path.join(fdir, f))

            else:
                pass

    # remove duplicates if any based on acq_dts
    seen_cld = set()
    fcld = [x for x in fcld if not (os.path.basename(x).split('.')[1] + '.' + os.path.basename(x).split('.')[2] in seen_cld or seen_cld.add(os.path.basename(x).split('.')[1] + '.' + os.path.basename(x).split('.')[2]))]

    seen_geo = set()
    f03 = [x for x in f03 if not (os.path.basename(x).split('.')[1] + '.' + os.path.basename(x).split('.')[2] in seen_geo or seen_geo.add(os.path.basename(x).split('.')[1] + '.' + os.path.basename(x).split('.')[2]))]

    fcld = sorted(fcld, key=lambda x: x.split('.')[2])
    f03  = sorted(f03,  key=lambda x: x.split('.')[2])
    return f03, fcld



def save_to_file_modis_viirs_geo_cld_opt(fdir, outdir, extent, geojson_fpath, buoys, norway_ship, start_dt, end_dt, quicklook_fdir, mode):

    f03, fcld_l2 = get_modis_viirs_geo_cld_opt(fdir)
    if (len(f03) == 0) or (len(fcld_l2) == 0):
        print("\nMessage [modis_viirs_cld_geo]: Could not find any common ref + cld products for MODIS\n")
        return 0

    n_obs = len(f03)

    print("Found {} overpasses".format(n_obs))

    valid_count = 0 # just for reporting

    start_dt_str = start_dt.strftime('%Y-%m-%d-%H%M')
    end_dt_str   = end_dt.strftime('%Y-%m-%d-%H%M')

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
    print("Message [modis_viirs_cld_geo]: Start datetime:{}, End datetime: {}".format(start_dt_str, end_dt_str))
    print("Warning [modis_viirs_cld_geo]: Start datetime was changed to catch more data and account for latency")

    for i in tqdm(range(n_obs)):
        try:

            # geometries/geolocation file
            geo_file    = f03[i]

            # check if the file has already been processed
            bgeo_file = os.path.basename(geo_file)
            yyyydoy_hhmm = bgeo_file.split('.')[1][1:] + '.' + bgeo_file.split('.')[2]

            print("Message [modis_viirs_cld_geo]: Processing: ", bgeo_file)
            if yyyydoy_hhmm in exist_acq_dts: # if already processed, then skip
                print("Message [modis_viirs_cld_geo]: Skipping {} as it has likely already been processed previously".format(bgeo_file))
                continue


            if not within_range(geo_file, start_dt, end_dt):
                print("Message [modis_viirs_cld_geo]: Skipping {} as it is outside the provided date range: {} to {}".format(geo_file, start_dt.strftime("%Y-%m-%d:%H%M"), end_dt.strftime("%Y-%m-%d:%H%M")))
                continue

            acq_dt = os.path.basename(geo_file).split('.')[1] + '.' + os.path.basename(geo_file).split('.')[2]

            if os.path.basename(geo_file).upper().startswith('M'): # modis
                f_geo = modis_03(fnames=[geo_file], extent=extent, keep_dims=True)
            elif os.path.basename(geo_file).upper().startswith('V'): # viirs
                f_geo = viirs_03(fnames=[geo_file], extent=extent, keep_dims=True)

        except Exception as err:
            print("Error [modis_viirs_cld_geo]: some error with {}...skipping...\nError {}\n".format(geo_file, err))
            continue

        lon2d_1km = f_geo.data['lon']['data'].T
        lat2d_1km = f_geo.data['lat']['data'].T
        if len(lon2d_1km) == 0 or len(lat2d_1km) == 0:
            # print('Lat/lon not valid')
            continue

        try:
            # cld file
            cld_file    = fcld_l2[i]
            # cloud product
            if os.path.basename(cld_file).upper().startswith(('MOD06_L2', 'MYD06_L2')): # modis
                f_cld = modis_l2(fnames=[cld_file], f03=f_geo, extent=extent, keep_dims=True)
                lon2d_5km = f_cld.data['lon_5km']['data'].T
                lat2d_5km = f_cld.data['lat_5km']['data'].T

            # viirs doesn't have a 5km geolocation since it is an external algorithm rather than a traditional retrieval
            elif os.path.basename(cld_file).upper().startswith('CLDPROP'): # viirs
                f_cld = viirs_cldprop_l2(fnames=[cld_file], f03=f_geo, maskvars=False, quality_assurance=0, keep_dims=True)
                lon2d_5km = f_geo.data['lon']['data'].T
                lat2d_5km = f_geo.data['lat']['data'].T


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
            if 'uwssec' in geo_file.lower():
                data_source = 'UWisc SSEC'
            else:
                data_source = 'NASA LANCE DAAC'

            arcsix_l2 = Imagery(data_source=data_source,
                                satellite=satellite,
                                acq_dt=group_name,
                                outdir=outdir,
                                geojson_fpath=geojson_fpath,
                                buoys=buoys,
                                norway_ship=norway_ship,
                                quicklook_fdir=quicklook_fdir,
                                mode=mode) # initialize class object

            _ = arcsix_l2.plot_liquid_water_paths(lon_2d=lon2d_1km, lat_2d=lat2d_1km, ctp=ctp, cwp=cwp_2d, cwp_1621=cwp_1621)
            _ = arcsix_l2.plot_ice_water_paths(lon_2d=lon2d_1km, lat_2d=lat2d_1km, ctp=ctp, cwp=cwp_2d, cwp_1621=cwp_1621)
            _ = arcsix_l2.plot_optical_depths(lon_2d=lon2d_1km, lat_2d=lat2d_1km, cot=cot_2d, cot_1621=cot_1621)
            _ = arcsix_l2.plot_cloud_phase(lon_2d=lon2d_1km, lat_2d=lat2d_1km, ctp_swir=ctp, ctp_ir=ctp_ir)
            _ = arcsix_l2.plot_cloud_top(lon_2d=lon2d_5km, lat_2d=lat2d_5km, cth=cth, ctt=ctt)

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

    # remove duplicates if any based on acq_dts
    seen_ref = set()
    fref = [x for x in fref if not (os.path.basename(x).split('.')[1] + '.' + os.path.basename(x).split('.')[2] in seen_ref or seen_ref.add(os.path.basename(x).split('.')[1] + '.' + os.path.basename(x).split('.')[2]))]

    seen_geo = set()
    f03 = [x for x in f03 if not (os.path.basename(x).split('.')[1] + '.' + os.path.basename(x).split('.')[2] in seen_geo or seen_geo.add(os.path.basename(x).split('.')[1] + '.' + os.path.basename(x).split('.')[2]))]

    fref = sorted(fref, key=lambda x: x.split('.')[2])
    f03  = sorted(f03,  key=lambda x: x.split('.')[2])
    return fref, f03


def save_to_file_modis_viirs_ref_geo(fdir, outdir, extent, geojson_fpath, buoys, norway_ship, start_dt, end_dt, quicklook_fdir, mode):

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

            print("Message [modis_viirs_ref_geo]: Processing: ", bgeo_file)
            if yyyydoy_hhmm in exist_acq_dts: # if already processed, then skip
                print("Message [modis_viirs_ref_geo]: Skipping {} as it has likely already been processed previously".format(bgeo_file))
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
                    print("Message [modis_viirs_ref_geo]: Skipping {} as it is outside the provided date range: {} to {}".format(bgeo_file, viirs_start_dt.strftime("%Y-%m-%d:%H%M"), end_dt.strftime("%Y-%m-%d:%H%M")))
                    continue

            else: # for MODIS keep latency as is
                if not within_range(geo_file, start_dt, end_dt):
                    print("Message [modis_viirs_ref_geo]: Skipping {} as it is outside the provided date range: {} to {}".format(bgeo_file, start_dt.strftime("%Y-%m-%d:%H%M"), end_dt.strftime("%Y-%m-%d:%H%M")))
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
            print("Message [modis_viirs_ref_geo]: Some error: with {}...skipping...\nError: {}\n".format(geo_file, err))
            continue

        lon2d_1km = f_geo.data['lon']['data'].T
        lat2d_1km = f_geo.data['lat']['data'].T
        if len(lon2d_1km) == 0 or len(lat2d_1km) == 0:
            # print('Lat/lon not valid')
            continue

        sza_2d = f_geo.data['sza']['data'].T
        sza_2d[sza_2d > util.constants.SZA_LIMIT] = util.constants.SZA_LIMIT

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
                                    bands=["M05", "M07", "M03", "M04", "M10", "M11", "M09", "M15"])
            else:
                print("\nMessage [modis_viirs_ref_geo]: Only VIIRS and MODIS products are supported. Instead, file was: {}\n".format(ref_file))
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
            if 'uwssec' in geo_file.lower():
                data_source = 'UWisc SSEC'
            else:
                data_source = 'NASA LAADS DAAC'

            arcsix_imagery = Imagery(data_source=data_source,
                                     satellite=satellite,
                                     acq_dt=group_name,
                                     outdir=outdir,
                                     geojson_fpath=geojson_fpath,
                                     buoys=buoys,
                                     norway_ship=norway_ship,
                                     quicklook_fdir=quicklook_fdir,
                                     mode=mode) # initialize class object

            _ = arcsix_imagery.create_true_color_imagery(lon_2d=lon2d_1km, lat_2d=lat2d_1km, red=ref_650, green=ref_555, blue=ref_470, sza=sza_2d)

            _ = arcsix_imagery.create_false_color_721_imagery(lon_2d=lon2d_1km, lat_2d=lat2d_1km, red=ref_2130, green=ref_860, blue=ref_650, sza=sza_2d)

            if satellite != 'Aqua': # band 6 at 1640 is broken for Aqua

                _ = arcsix_imagery.create_false_color_367_imagery(lon_2d=lon2d_1km, lat_2d=lat2d_1km, red=ref_470, green=ref_1640, blue=ref_2130, sza=sza_2d)

                # 1.38-1.6-2.1 reflectance
                _ = arcsix_imagery.create_false_color_cirrus_imagery(lon_2d=lon2d_1km, lat_2d=lat2d_1km, red=ref_1380, green=ref_1640, blue=ref_2130, sza=sza_2d)

                # 11micron-1.6-2.1 radiance
                _ = arcsix_imagery.create_false_color_ir_imagery(lon_2d=lon2d_1km, lat_2d=lat2d_1km, red=rad_11000, green=rad_1640, blue=rad_2130)

            valid_count += 1
            print("Message [modis_viirs_ref_geo]: Successfully processed: ", acq_dt)

        except Exception as err:
            print("Error [modis_viirs_ref_geo]:", err)
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
    START_TIME = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir",  default=None,  type=str, help="Path to directory containing raw MODIS/VIIRS HDF files")
    parser.add_argument("--outdir", type=str, default=None, help="Path to directory where images will be saved")
    parser.add_argument('--ndir_recent', type=int, metavar='', default=None, help='Time range in terms of sub-directories. For example, if --ndir_recent 5, then the 5 most recent (based on filenames) datetime ranges will be used to create data')
    parser.add_argument('--max_hours', type=int, metavar='', default=3, help='Maximum time range in hours. For example, if --max_hours 5, then the 5 most recent hours from the last available satellite time are used')
    parser.add_argument("--nrt", action='store_true', help="Enable --nrt to process VIIRS L1b and 03 products, and MODIS L1b, 03, and clds")
    parser.add_argument('--geojson', type=str, metavar='',
                        help='Path to a geoJSON file containing the extent of interest coordinates\n'\
                        'Example:  --geojson my/path/to/geofile.json\n \n')
    parser.add_argument('--buoys', type=str, metavar='', default=None, help='Path to the JSON file containing URLs to the buoys csv data')
    parser.add_argument('--norway_ship', type=str, metavar='', default=None, help='Path to the JSON file where icebreaker data is/will be stored')
    parser.add_argument('--mode', type=str, metavar='', default='lincoln', help='One of "baffin", "lincoln", or "platypus" ')
    parser.add_argument("--quicklook_fdir", type=str, default=None, help="Path to directory where quicklook images will be saved")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if (args.quicklook_fdir is not None) and (not os.path.exists(args.quicklook_fdir)):
        os.makedirs(args.quicklook_fdir)

    manual_list = []

    # subdirs will only be yyyy-mm-dd (or yyyy_mm_dd in older sdown versions)
    # subdirs = sorted([os.path.join(args.fdir, f) for f in os.listdir(args.fdir) if os.path.isdir(os.path.join(args.fdir, f))])
    subdirs = []
    for f in os.listdir(args.fdir):
        if os.path.isdir(os.path.join(args.fdir, f)) and (len(os.listdir(os.path.join(args.fdir, f))) > 2): # metadata + two files minimum
            subdirs.append(os.path.join(args.fdir, f))

    subdirs = sorted(subdirs)

    start_dt_hhmm, end_dt_hhmm = get_start_end_dates_metadata(subdirs[-1])

    start_dt_hhmm = end_dt_hhmm - datetime.timedelta(hours=args.max_hours)
    print("Message [visualize_satellites]: Start time and end times were too far apart or not far enough...limiting to {} hour gap.".format(args.max_hours))


    start_dt_hhmm_str = start_dt_hhmm.strftime('%Y-%m-%d-%H%M')
    end_dt_hhmm_str   = end_dt_hhmm.strftime('%Y-%m-%d-%H%M')

    outdir_dt =  start_dt_hhmm_str + '_' + end_dt_hhmm_str

    print("==============================================================================================")
    print("Message [visualize_satellites]: Start: {}, End: {}".format(start_dt_hhmm_str, end_dt_hhmm_str))
    print("==============================================================================================")

    if (args.ndir_recent is not None) and (len(subdirs) >= args.ndir_recent):
        subdirs = subdirs[-args.ndir_recent:]

    print("Message [visualize_satellites]: {} sub-directories will be analyzed".format(len(subdirs)))

    for fdir in tqdm(subdirs):
        dt = os.path.basename(fdir)
        print("Currently analyzing:", dt) # date
        year, month, date, hour, minute, sec, extent = get_metadata(fdir)

        outdir = args.outdir

        ret = 0
        if args.nrt:
            ret += save_to_file_modis_viirs_ref_geo(fdir, outdir, extent, geojson_fpath=args.geojson, buoys=args.buoys, start_dt=start_dt_hhmm, end_dt=end_dt_hhmm, quicklook_fdir=args.quicklook_fdir, norway_ship=args.norway_ship, mode=args.mode)
            ret += save_to_file_modis_viirs_geo_cld_opt(fdir, outdir, extent, geojson_fpath=args.geojson, buoys=args.buoys, start_dt=start_dt_hhmm, end_dt=end_dt_hhmm, quicklook_fdir=args.quicklook_fdir, norway_ship=args.norway_ship, mode=args.mode)

        if ret == 0:
            manual_list.append(fdir)
            continue


    print("Missed: ", manual_list)
    print("Finished!")

    END_TIME = datetime.datetime.now()
    print('Time taken to execute {}: {}'.format(os.path.basename(__file__), END_TIME - START_TIME))
