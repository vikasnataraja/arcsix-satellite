import os
import sys
import math
import argparse
import h5py
import numpy as np
from arctic_gridding_utils import modis_l1b, modis_03, modis_l2, modis_35_l2, grid_by_lonlat
from arctic_gridding_utils import viirs_l1b, viirs_03, viirs_cldprop_l2
import datetime
from tqdm import tqdm
from pyhdf.error import HDF4Error

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

START_TIME = datetime.datetime.now()

EARTH_RADIUS = 6371.009

SZA_LIMIT = 81.36

def get_swath_area(lon, lat):
    swath_area = np.abs(np.abs(lon.max()) - np.abs(lon.min())) * np.abs(np.abs(lat.max()) - np.abs(lat.min()))
    return swath_area

# def calc_lwp(cot, cer, rho_water=1000.):
#     return 2. * cot * cer * 1e-6 * rho_water * 1e3 / 3.


def get_lon2lat2(lon1, lat1, bearing, distance):
    """
    Get lat/lon of the destination point given the initial point,
    bearing, and the great-circle distance between the points.
    Args:
        - lon1: float, longitude of the first point in degrees
        - lat1: float, latitude of the first point in degrees
        - bearing: float, bearing angle of trajectory.
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




utf8_type = h5py.string_dtype('utf-8', 30)




def get_modis_1km_geo_cld_fnames(fdir):

    fnames = get_all_fnames(fdir)
    f1km, acq_dts_f1km = get_product_info(fnames, product_id="1KM")
    ft03, acq_dts_ft03 = get_product_info(fnames, product_id="MOD03")
    fa03, acq_dts_fa03 = get_product_info(fnames, product_id="MYD03")
    f03  = ft03 + fa03
    acq_dts_f03 = acq_dts_ft03 + acq_dts_fa03
    f06_l2, acq_dts_f06_l2 = get_product_info(fnames, product_id="06_L2")

    acq_dts_common = list(set.intersection(*map(set,
                                                 [acq_dts_f1km,
                                                  acq_dts_f03,
                                                  acq_dts_f06_l2])))

    if len(acq_dts_common) == 0:
        raise IndexError("Could not find any common date/times among products")

    f1km, f03, f06_l2, = [], [], []
    for f in fnames:
        filename = os.path.basename(f).upper()
        acq_dt = filename.split(".")[1] + '.' + filename.split(".")[2]
        if acq_dt in acq_dts_common:
            if "1KM" in filename:
                f1km.append(os.path.join(fdir, f))
            elif ("MOD03" in filename) or ("MYD03" in filename):
                f03.append(os.path.join(fdir, f))
            elif "06_L2" in filename:
                f06_l2.append(os.path.join(fdir, f))

    return f1km, f03, f06_l2


def save_to_file_modis(fdir, outdir, lon_1d, lat_1d, o_thresh=85, zero_thresh=25):

    f1km, f03, f06_l2 = get_modis_1km_geo_cld_fnames(fdir)
    if (len(f1km) == 0) or (len(f03) == 0) or (len(f06_l2) == 0):
        raise FileNotFoundError('Could not find the valid files')

    # f1km = [i for i in f1km if "NRT" not in i]
    # f03  = [i for i in f03 if "NRT" not in i]
    # f06_l2 = [i for i in f06_l2 if "NRT" not in i]

    acq_dts = sorted([os.path.basename(i)[9:22] for i in f1km])
    print("Found {} overpasses".format(len(acq_dts)))
    extent = [lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()]

    ext_area = get_swath_area(lon_1d, lat_1d)

    valid_count = 0
    with h5py.File(os.path.join(outdir, 'ref_geo_cld-data.h5'), 'w') as f0:
        for acq_dt in acq_dts:
            sdata = {}

            # geometries file
            geo_file = list(filter(lambda x: acq_dt in x, f03))[0] # match acquisition to get geometry file
            modis_geo = modis_03(fnames=[geo_file], extent=extent)
            lon0 = modis_geo.data['lon']['data']
            lat0 = modis_geo.data['lat']['data']

            if len(lon0) == 0 or len(lat0) == 0:
                print('Lat/lon not valid\n')
                continue

            mod_area = get_swath_area(lon0, lat0)
            overlap  = mod_area * 100. / ext_area

            if overlap < o_thresh:
                print('Skipping {}, overlap {:0.1f}% is less than {}%'.format(acq_dt, overlap, o_thresh))
                continue

            # lon = modis_1km.data['lon']['data']
            # lat = modis_1km.data['lat']['data']
            # if len(lon) == 0 or len(lat) == 0:
            #     print('Lat/lon not valid')
            #     continue

            vza0 = modis_geo.data['vza']['data']
            sza0 = modis_geo.data['sza']['data']
            _, _, sdata['sza_2d'] = grid_by_lonlat(lon0, lat0, sza0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
            _, _, sdata['vza_2d'] = grid_by_lonlat(lon0, lat0, vza0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')

            # 1km reflectance file
            ref_file = list(filter(lambda x: acq_dt in x, f1km))[0]

            modis_1km = modis_l1b(fnames=[ref_file], extent=extent, f03=modis_geo, bands=[1, 2, 3, 4, 6, 7])
            dat = modis_1km.data['ref']['data']

            # Band 7 (2130nm)
            _, _, sdata['data_2130'] = grid_by_lonlat(lon0, lat0, dat[5], lon_1d=lon_1d, lat_1d=lat_1d, method='nearest', Ngrid_limit=1)
            # zero_pct = np.count_nonzero(sdata['data_2130'] == 0) * 100/sdata['data_2130'].size

            # if zero_pct > zero_thresh:
            #     print('Skipping {}, zero_pct {:0.1f}% is more than {}%'.format(acq_dt, zero_pct, zero_thresh))
            #     continue

            # [650.0, 860.0, 470.0, 555.0, 1240.0, 1640.0, 2130.0]
            # Bands 6 (1640nm) and 1 (650nm) and 2 (860nm) and 3 (470nm)
            _, _, sdata['data_1640'] = grid_by_lonlat(lon0, lat0, dat[4], lon_1d=lon_1d, lat_1d=lat_1d, method='nearest', Ngrid_limit=1)
            _, _, sdata['data_650'] = grid_by_lonlat(lon0, lat0, dat[0], lon_1d=lon_1d, lat_1d=lat_1d, method='nearest', Ngrid_limit=1)
            _, _, sdata['data_860'] = grid_by_lonlat(lon0, lat0, dat[1], lon_1d=lon_1d, lat_1d=lat_1d, method='nearest', Ngrid_limit=1)
            _, _, sdata['data_470'] = grid_by_lonlat(lon0, lat0, dat[2], lon_1d=lon_1d, lat_1d=lat_1d, method='nearest', Ngrid_limit=1)
            _, _, sdata['data_555'] = grid_by_lonlat(lon0, lat0, dat[3], lon_1d=lon_1d, lat_1d=lat_1d, method='nearest', Ngrid_limit=1)

            # cloud product
            cld_file = list(filter(lambda x: acq_dt in x, f06_l2))[0] # match aquisition to get cloud_l2 file
            modis_cld = modis_l2(fnames=[cld_file], f03=modis_geo, extent=extent)
            ctp0 = modis_cld.data['ctp']['data']
            ctp0 = np.float64(ctp0)
            cot0 = modis_cld.data['cot']['data']
            cer0 = modis_cld.data['cer']['data']
            cwp0 = modis_cld.data['cwp']['data']
            cot_1621 = modis_cld.data['cot_1621']['data']
            cer_1621 = modis_cld.data['cer_1621']['data']
            cwp_1621 = modis_cld.data['cwp_1621']['data']
            # lon0 = modis_cld.data['lon']['data']
            # lat0 = modis_cld.data['lat']['data']
            _, _, ctp_temp    = grid_by_lonlat(lon0, lat0, ctp0, lon_1d=lon_1d, lat_1d=lat_1d, method='nearest', Ngrid_limit=1)
            # sdata['ctp'] = ctp_temp.astype('int')
            _, _, sdata['cot_2d'] = grid_by_lonlat(lon0, lat0, cot0, lon_1d=lon_1d, lat_1d=lat_1d, method='nearest', Ngrid_limit=1)
            _, _, sdata['cer_2d'] = grid_by_lonlat(lon0, lat0, cer0, lon_1d=lon_1d, lat_1d=lat_1d, method='nearest', Ngrid_limit=1)

            _, _, sdata['cwp_2d'] = grid_by_lonlat(lon0, lat0, cwp0, lon_1d=lon_1d, lat_1d=lat_1d, method='nearest', Ngrid_limit=1)
            _, _, sdata['cot_1621'] = grid_by_lonlat(lon0, lat0, cot_1621, lon_1d=lon_1d, lat_1d=lat_1d, method='nearest', Ngrid_limit=1)
            _, _, sdata['cer_1621'] = grid_by_lonlat(lon0, lat0, cer_1621, lon_1d=lon_1d, lat_1d=lat_1d, method='nearest', Ngrid_limit=1)

            _, _, sdata['cwp_1621'] = grid_by_lonlat(lon0, lat0, cwp_1621, lon_1d=lon_1d, lat_1d=lat_1d, method='nearest', Ngrid_limit=1)

            # satellite name
            if os.path.basename(ref_file)[:3] == "MOD":
                satellite  = "Terra"
                group_name = "T" + acq_dt[1:]
            elif os.path.basename(ref_file)[:3] == "MYD":
                satellite  = "Aqua"
                group_name = "A" + acq_dt[1:]
            elif os.path.basename(ref_file)[:3] == "VNP":
                satellite  = "Suomi-NPP"
                group_name = "S" + acq_dt[1:]
            elif os.path.basename(ref_file)[:3] == "VJ1":
                satellite  = "NOAA-20 (JPSS-1)"
                group_name = "J" + acq_dt[1:]
            else:
                satellite  = "Unknown"
                group_name = "Z" + acq_dt[1:]

            satellite = np.array(satellite.encode("utf-8"), dtype=utf8_type)

            # aggregate to coarser resolution
            # print("Running aggregation\n")
            # for key in sdata.keys():
            #     sdata[key] = aggregate(sdata[key], dx=10, dy=10)

            # save data
            g0 = f0.create_group(acq_dt)
            _ = g0.create_dataset('ref_2130', data=sdata['data_2130'].T)
            _ = g0.create_dataset('ref_1640',  data=sdata['data_1640'].T)
            _ = g0.create_dataset('ref_860',  data=sdata['data_860'].T)
            _ = g0.create_dataset('ref_650',  data=sdata['data_650'].T)
            _ = g0.create_dataset('ref_555',  data=sdata['data_555'].T)
            _ = g0.create_dataset('ref_470',  data=sdata['data_470'].T)
            _ = g0.create_dataset('sza_2d',   data=sdata['sza_2d'].T)
            _ = g0.create_dataset('vza_2d',   data=sdata['vza_2d'].T)
            _ = g0.create_dataset('ctp',      data=sdata['ctp'].T)
            _ = g0.create_dataset('cot_2d',   data=sdata['cot_2d'].T)
            _ = g0.create_dataset('cer_2d',   data=sdata['cer_2d'].T)
            _ = g0.create_dataset('cwp_2d',   data=sdata['cwp_2d'].T)
            _ = g0.create_dataset('cot_1621',   data=sdata['cot_1621'].T)
            _ = g0.create_dataset('cer_1621',   data=sdata['cer_1621'].T)
            _ = g0.create_dataset('cwp_1621',   data=sdata['cwp_1621'].T)
            _ = g0.create_dataset('lon_1d',   data=lon_1d)
            _ = g0.create_dataset('lat_1d',   data=lat_1d)
            _ = g0.create_dataset('satellite', data=satellite)

            valid_count += 1

    print("Successfully extracted data from {} of {} overpasses".format(valid_count, len(acq_dts)))
    return 1





def get_modis_viirs_fnames(fdir):

    # MODIS files
    modis_fnames = get_all_modis_files(fdir)
    _, acq_dts_modis_fref = get_product_info(modis_fnames, product_id="1KM")
    _, acq_dts_ft03 = get_product_info(modis_fnames, product_id="MOD03")
    _, acq_dts_fa03 = get_product_info(modis_fnames, product_id="MYD03")
    # modis_f03  = ft03 + fa03
    acq_dts_modis_f03 = acq_dts_ft03 + acq_dts_fa03
    _, acq_dts_modis_f06_l2 = get_product_info(modis_fnames, product_id="06_L2")

    modis_acq_dts_common = list(set.intersection(*map(set,
                                                 [acq_dts_modis_fref,
                                                  acq_dts_modis_f03,
                                                  acq_dts_modis_f06_l2])))

    if len(modis_acq_dts_common) == 0:
        raise IndexError("Could not find any common date/times among products")

    # VIIRS files
    viirs_fnames = get_all_viirs_files(fdir)
    _, acq_dts_viirs_fref = get_product_info(viirs_fnames, product_id="02MOD")
    _, acq_dts_viirs_f03 = get_product_info(viirs_fnames, product_id="03MOD")
    _, acq_dts_viirs_fcld_l2 = get_product_info(viirs_fnames, product_id="CLDPROP_L2")

    viirs_acq_dts_common = list(set.intersection(*map(set,
                                                 [acq_dts_viirs_fref,
                                                  acq_dts_viirs_f03,
                                                  acq_dts_viirs_fcld_l2])))

    if len(viirs_acq_dts_common) == 0:
        raise IndexError("Could not find any common date/times among products")

    fnames = modis_fnames + viirs_fnames
    acq_dts_common = modis_acq_dts_common + viirs_acq_dts_common

    fref, f03, fcld_l2 = [], [], []
    for f in fnames:
        filename = os.path.basename(f).upper()
        acq_dt = filename.split(".")[1] + '.' + filename.split(".")[2]
        if acq_dt in acq_dts_common:
            if ("1KM" in filename.upper()) or \
                ("02MOD" in filename.upper()):
                fref.append(os.path.join(fdir, f))
            elif ("MOD03" in filename.upper()) or ("MYD03" in filename.upper()) or \
                  ("03MOD" in filename.upper()):
                f03.append(os.path.join(fdir, f))
            elif ("06_L2" in filename.upper()) or \
                  ("CLDPROP_L2" in filename.upper()):
                fcld_l2.append(os.path.join(fdir, f))

    fref = sorted(fref, key=lambda x: x.split('.')[2])
    f03  = sorted(f03, key=lambda x: x.split('.')[2])
    fcld_l2 = sorted(fcld_l2, key=lambda x: x.split('.')[2])
    return fref, f03, fcld_l2


def save_to_file_modis_viirs(fdir, outdir, lon_1d, lat_1d, o_thresh=75):

    fref, f03, fcld_l2 = get_modis_viirs_fnames(fdir)
    if (len(fref) == 0) or (len(f03) == 0) or (len(fcld_l2) == 0):
        raise FileNotFoundError("\nCould not find the valid files\n")

    n_obs = len(f03)
    # acquisition date times
    # acq_dts = sorted([os.path.basename(i)[9:22] for i in fref])[-15:]
    print("Found {} overpasses".format(n_obs))
    extent = [lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()]
    ext_area = get_swath_area(lon_1d, lat_1d)

    valid_count = 0 # just for reporting

    with h5py.File(os.path.join(outdir, 'ref_geo_cld-data.h5'), 'w') as f0:
        # for acq_dt in acq_dts:
        for i in range(n_obs):
            sdata = {}
            # geometries/geolocation file
            # geo_file = list(filter(lambda x: acq_dt in x, f03))[0] # match acquisition to get geometry file
            geo_file = f03[i]
            ref_file = fref[i]
            cld_file = fcld_l2[i]
            acq_dt = os.path.basename(geo_file).split('.')[1] + '.' + os.path.basename(geo_file).split('.')[2]

            try:
                if os.path.basename(geo_file).upper().startswith('M'): # modis
                    f_geo = modis_03(fnames=[geo_file], extent=extent)
                elif os.path.basename(geo_file).upper().startswith('V'): # viirs
                    f_geo = viirs_03(fnames=[geo_file], extent=extent)
                else:
                    raise NotImplementedError("\nOnly VIIRS and MODIS products are supported\n")
            except HDF4Error:
                print("PyHDF error with {}...skipping...\n".format(geo_file))
                continue
            except FileNotFoundError:
                print("netcdf error with {}...skipping...\n".format(geo_file))
                continue
            lon0 = f_geo.data['lon']['data']
            lat0 = f_geo.data['lat']['data']
            if len(lon0) == 0 or len(lat0) == 0:
                # print('Lat/lon not valid')
                continue

            mod_area = get_swath_area(lon0, lat0)
            overlap  = mod_area * 100. / ext_area
            if overlap < o_thresh:
                # print('Skipping {}, overlap {:0.1f} is less than {}%'.format(acq_dt, overlap, o_thresh))
                continue

            vza0 = f_geo.data['vza']['data']
            sza0 = f_geo.data['sza']['data']
            _, _, sdata['sza_2d'] = grid_by_lonlat(lon0, lat0, sza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear')
            _, _, sdata['vza_2d'] = grid_by_lonlat(lon0, lat0, vza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear')

            # reflectance file
            # ref_file = list(filter(lambda x: acq_dt in x, fref))[0]
            if os.path.basename(ref_file).upper().startswith('M'): # modis
                f_vis = modis_l1b(fnames=[ref_file], extent=extent, f03=f_geo, bands=[1, 2, 3, 4, 6, 7])
            elif os.path.basename(ref_file).upper().startswith('V'): # viirs
                f_vis = viirs_l1b(fnames=[ref_file], extent=extent, f03=f_geo, bands=["M05", "M07", "M02", "M04", "M10", "M11"])
            else:
                raise NotImplementedError("\nOnly VIIRS and MODIS products are supported\n")
            dat = f_vis.data['ref']['data']

            # Band 7 (2130nm)
            _, _, sdata['data_2130'] = grid_by_lonlat(lon0, lat0, dat[5], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            # [650.0, 860.0, 470.0, 555.0, 1240.0, 1640.0, 2130.0]
            # Bands 6 (1640nm) and 1 (650nm) and 2 (860nm) and 3 (470nm)
            _, _, sdata['data_1640'] = grid_by_lonlat(lon0, lat0, dat[4], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            _, _, sdata['data_650'] = grid_by_lonlat(lon0, lat0, dat[0], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            _, _, sdata['data_860'] = grid_by_lonlat(lon0, lat0, dat[1], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            _, _, sdata['data_470'] = grid_by_lonlat(lon0, lat0, dat[2], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            _, _, sdata['data_555'] = grid_by_lonlat(lon0, lat0, dat[3], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)


            # cloud product
            # cld_file = list(filter(lambda x: acq_dt in x, fcld_l2))[0] # match aquisition to get cloud_l2 file
            if os.path.basename(cld_file).upper().startswith(('MOD06_L2', 'MYD06_L2')): # modis
                f_cld = modis_l2(fnames=[cld_file], f03=f_geo, extent=extent)
            elif os.path.basename(cld_file).upper().startswith('CLDPROP'): # viirs
                f_cld = viirs_cldprop_l2(fnames=[cld_file], f03=f_geo, extent=extent)
            else:
                raise NotImplementedError("\nOnly VIIRS and MODIS products are supported\n")

            cot0 = f_cld.data['cot']['data']
            cer0 = f_cld.data['cer']['data']
            cwp0 = f_cld.data['cwp']['data']
            cot_1621 = f_cld.data['cot_1621']['data']
            cer_1621 = f_cld.data['cer_1621']['data']
            cwp_1621 = f_cld.data['cwp_1621']['data']
            ctp0 = np.float64(f_cld.data['ctp']['data'])
            lon0 = f_cld.data['lon']['data']
            lat0 = f_cld.data['lat']['data']
            _, _, ctp_temp = grid_by_lonlat(lon0, lat0, ctp0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            sdata['ctp'] = ctp_temp.astype('int')
            _, _, sdata['cot_2d'] = grid_by_lonlat(lon0, lat0, cot0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            _, _, sdata['cer_2d'] = grid_by_lonlat(lon0, lat0, cer0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            _, _, sdata['cwp_2d'] = grid_by_lonlat(lon0, lat0, cwp0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            _, _, sdata['cot_1621'] = grid_by_lonlat(lon0, lat0, cot_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            _, _, sdata['cer_1621'] = grid_by_lonlat(lon0, lat0, cer_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            _, _, sdata['cwp_1621'] = grid_by_lonlat(lon0, lat0, cwp_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)

            # satellite name
            if os.path.basename(ref_file)[:3] == "MOD":
                satellite  = "Terra"
                group_name = "T" + acq_dt[1:]
            elif os.path.basename(ref_file)[:3] == "MYD":
                satellite  = "Aqua"
                group_name = "A" + acq_dt[1:]
            elif os.path.basename(ref_file)[:3] == "VNP":
                satellite  = "Suomi-NPP"
                group_name = "S" + acq_dt[1:]
            elif os.path.basename(ref_file)[:3] == "VJ1":
                satellite  = "NOAA-20 (JPSS-1)"
                group_name = "J" + acq_dt[1:]
            else:
                satellite  = "Unknown"
                group_name = "Z" + acq_dt[1:]

            satellite = np.array(satellite.encode("utf-8"), dtype=utf8_type)

            print(acq_dt, group_name, satellite)
            print("-------------------")
            # save data
            g0 = f0.create_group(group_name)
            _ = g0.create_dataset('ref_2130', data=sdata['data_2130'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('ref_1640', data=sdata['data_1640'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('ref_860',  data=sdata['data_860'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('ref_650',  data=sdata['data_650'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('ref_555',  data=sdata['data_555'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('ref_470',  data=sdata['data_470'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('sza_2d',   data=sdata['sza_2d'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('vza_2d',   data=sdata['vza_2d'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('ctp',      data=sdata['ctp'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('cot_2d',   data=sdata['cot_2d'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('cer_2d',   data=sdata['cer_2d'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('cwp_2d',   data=sdata['cwp_2d'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('cot_1621', data=sdata['cot_1621'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('cer_1621', data=sdata['cer_1621'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('cwp_1621', data=sdata['cwp_1621'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('lon_1d',   data=lon_1d, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('lat_1d',   data=lat_1d, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('satellite',data=satellite)

            valid_count += 1

    print("Successfully extracted data from {} of {} overpasses".format(valid_count, n_obs))
    return 1



################################################################################################################


def get_modis_viirs_viz_fnames(fdir):

    # MODIS files
    modis_fnames = get_all_modis_files(fdir)
    _, acq_dts_modis_fref = get_product_info(modis_fnames, product_id="1KM")
    _, acq_dts_ft03 = get_product_info(modis_fnames, product_id="MOD03")
    _, acq_dts_fa03 = get_product_info(modis_fnames, product_id="MYD03")
    # modis_f03  = ft03 + fa03
    acq_dts_modis_f03 = acq_dts_ft03 + acq_dts_fa03

    modis_acq_dts_common = list(set.intersection(*map(set,
                                                 [acq_dts_modis_fref,
                                                  acq_dts_modis_f03])))

    if len(modis_acq_dts_common) == 0:
        raise IndexError("Could not find any common date/times among products")

    # VIIRS files
    viirs_fnames = get_all_viirs_files(fdir)
    _, acq_dts_viirs_fref = get_product_info(viirs_fnames, product_id="02MOD")
    _, acq_dts_viirs_f03 = get_product_info(viirs_fnames, product_id="03MOD")

    viirs_acq_dts_common = list(set.intersection(*map(set,
                                                 [acq_dts_viirs_fref,
                                                  acq_dts_viirs_f03])))

    if len(viirs_acq_dts_common) == 0:
        raise IndexError("Could not find any common date/times among products")

    fnames = modis_fnames + viirs_fnames
    acq_dts_common = modis_acq_dts_common + viirs_acq_dts_common

    fref, f03 = [], []
    for f in fnames:
        filename = os.path.basename(f).upper()
        acq_dt = filename.split(".")[1] + '.' + filename.split(".")[2]
        if acq_dt in acq_dts_common:
            if ("1KM" in filename.upper()) or \
                ("02MOD" in filename.upper()):
                fref.append(os.path.join(fdir, f))
            elif ("MOD03" in filename.upper()) or ("MYD03" in filename.upper()) or \
                  ("03MOD" in filename.upper()):
                f03.append(os.path.join(fdir, f))

    fref = sorted(fref, key=lambda x: x.split('.')[2])
    f03  = sorted(f03, key=lambda x: x.split('.')[2])
    return fref, f03


def save_to_file_modis_viirs_viz(fdir, outdir, lon_1d, lat_1d, o_thresh=75):

    fref, f03 = get_modis_viirs_viz_fnames(fdir)
    if (len(fref) == 0) or (len(f03) == 0):
        raise FileNotFoundError("\nCould not find the valid files\n")

    n_obs = len(f03)
    # acquisition date times
    # acq_dts = sorted([os.path.basename(i)[9:22] for i in fref])[-15:]
    print("Found {} overpasses".format(n_obs))
    extent = [lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()]
    ext_area = get_swath_area(lon_1d, lat_1d)

    valid_count = 0 # just for reporting

    with h5py.File(os.path.join(outdir, 'ref_geo_cld-data.h5'), 'w') as f0:
        # for acq_dt in acq_dts:
        for i in range(n_obs):
            sdata = {}
            # geometries/geolocation file
            # geo_file = list(filter(lambda x: acq_dt in x, f03))[0] # match acquisition to get geometry file
            geo_file = f03[i]
            ref_file = fref[i]
            acq_dt = os.path.basename(geo_file).split('.')[1] + '.' + os.path.basename(geo_file).split('.')[2]

            try:
                if os.path.basename(geo_file).upper().startswith('M'): # modis
                    f_geo = modis_03(fnames=[geo_file], extent=extent)
                elif os.path.basename(geo_file).upper().startswith('V'): # viirs
                    f_geo = viirs_03(fnames=[geo_file], extent=extent)
                else:
                    raise NotImplementedError("\nOnly VIIRS and MODIS products are supported\n")
            except HDF4Error:
                print("PyHDF error with {}...skipping...\n".format(geo_file))
                continue
            except FileNotFoundError:
                print("netcdf error with {}...skipping...\n".format(geo_file))
                continue
            lon0 = f_geo.data['lon']['data']
            lat0 = f_geo.data['lat']['data']
            if len(lon0) == 0 or len(lat0) == 0:
                # print('Lat/lon not valid')
                continue

            mod_area = get_swath_area(lon0, lat0)
            overlap  = mod_area * 100. / ext_area
            if overlap < o_thresh:
                # print('Skipping {}, overlap {:0.1f} is less than {}%'.format(acq_dt, overlap, o_thresh))
                continue

            vza0 = f_geo.data['vza']['data']
            sza0 = f_geo.data['sza']['data']
            _, _, sdata['sza_2d'] = grid_by_lonlat(lon0, lat0, sza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear')
            _, _, sdata['vza_2d'] = grid_by_lonlat(lon0, lat0, vza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear')

            # reflectance file
            # ref_file = list(filter(lambda x: acq_dt in x, fref))[0]
            if os.path.basename(ref_file).upper().startswith('M'): # modis
                f_vis = modis_l1b(fnames=[ref_file], extent=extent, f03=f_geo, bands=[1, 2, 3, 4, 6, 7])
            elif os.path.basename(ref_file).upper().startswith('V'): # viirs
                f_vis = viirs_l1b(fnames=[ref_file], extent=extent, f03=f_geo, bands=["M05", "M07", "M02", "M04", "M10", "M11"])
            else:
                raise NotImplementedError("\nOnly VIIRS and MODIS products are supported\n")
            dat = f_vis.data['ref']['data']

            # Band 7 (2130nm)
            _, _, sdata['data_2130'] = grid_by_lonlat(lon0, lat0, dat[5], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            # [650.0, 860.0, 470.0, 555.0, 1240.0, 1640.0, 2130.0]
            # Bands 6 (1640nm) and 1 (650nm) and 2 (860nm) and 3 (470nm)
            _, _, sdata['data_1640'] = grid_by_lonlat(lon0, lat0, dat[4], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            _, _, sdata['data_650'] = grid_by_lonlat(lon0, lat0, dat[0], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            _, _, sdata['data_860'] = grid_by_lonlat(lon0, lat0, dat[1], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            _, _, sdata['data_470'] = grid_by_lonlat(lon0, lat0, dat[2], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)
            _, _, sdata['data_555'] = grid_by_lonlat(lon0, lat0, dat[3], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1)



            # satellite name
            if os.path.basename(ref_file)[:3] == "MOD":
                satellite  = "Terra"
                group_name = "T" + acq_dt[1:]
            elif os.path.basename(ref_file)[:3] == "MYD":
                satellite  = "Aqua"
                group_name = "A" + acq_dt[1:]
            elif os.path.basename(ref_file)[:3] == "VNP":
                satellite  = "Suomi-NPP"
                group_name = "S" + acq_dt[1:]
            elif os.path.basename(ref_file)[:3] == "VJ1":
                satellite  = "NOAA-20 (JPSS-1)"
                group_name = "J" + acq_dt[1:]
            else:
                satellite  = "Unknown"
                group_name = "Z" + acq_dt[1:]

            satellite = np.array(satellite.encode("utf-8"), dtype=utf8_type)

            print(acq_dt, group_name, satellite)
            print("-------------------")
            # save data
            g0 = f0.create_group(group_name)
            _ = g0.create_dataset('ref_2130', data=sdata['data_2130'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('ref_1640', data=sdata['data_1640'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('ref_860',  data=sdata['data_860'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('ref_650',  data=sdata['data_650'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('ref_555',  data=sdata['data_555'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('ref_470',  data=sdata['data_470'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('sza_2d',   data=sdata['sza_2d'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('vza_2d',   data=sdata['vza_2d'].T, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('lon_1d',   data=lon_1d, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('lat_1d',   data=lat_1d, compression='gzip', compression_opts=9)
            _ = g0.create_dataset('satellite',data=satellite)

            valid_count += 1

    print("Successfully extracted data from {} of {} overpasses".format(valid_count, n_obs))
    return 1

# =====================================================================================================

modis_ids = ["MOD02QKM", "MOD02HKM", "MOD021KM", "MYD02QKM", "MYD02HKM", "MYD021KM",
             "MOD03", "MYD03",
             "MOD06_L2", "MYD06_L2",
             "MOD35_L2", "MYD35_L2",
             "VNP02IMG", "VJ102IMG", "VNP02MOD", "VJ102MOD",
             "VNP03IMG", "VJ103IMG", "VNP03MOD", "VJ103MOD",
             "CLDPROP_L2"
             ]

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


def get_modis_viirs_ref_geo_cld_opt_mskfnames(fdir):

    # MODIS files
    modis_fnames = get_all_modis_files(fdir)
    _, acq_dts_modis_fref = get_product_info(modis_fnames, product_id="1KM")
    _, acq_dts_ft03 = get_product_info(modis_fnames, product_id="MOD03")
    _, acq_dts_fa03 = get_product_info(modis_fnames, product_id="MYD03")
    acq_dts_modis_f03 = acq_dts_ft03 + acq_dts_fa03
    # modis_f03 = temp1 + temp2
    _, acq_dts_modis_f06_l2 = get_product_info(modis_fnames, product_id="06_L2")
    _, acq_dts_modis_f35_l2 = get_product_info(modis_fnames, product_id="35_L2")

    modis_acq_dts_common = list(set.intersection(*map(set,
                                                 [acq_dts_modis_fref,
                                                  acq_dts_modis_f03,
                                                  acq_dts_modis_f06_l2,
                                                  acq_dts_modis_f35_l2])))

    if len(modis_acq_dts_common) == 0:
        raise IndexError("Could not find any common date/times among products")

    # VIIRS files
    viirs_fnames = get_all_viirs_files(fdir)
    _, acq_dts_viirs_fref = get_product_info(viirs_fnames, product_id="02MOD")
    _, acq_dts_viirs_f03 = get_product_info(viirs_fnames, product_id="03MOD")
    _, acq_dts_viirs_fcld_l2 = get_product_info(viirs_fnames, product_id="CLDPROP_L2")
    # _, acq_dts_viirs_fcldmsk_l2 = get_product_info(viirs_fnames, product_id="CLDPROP_L2")

    viirs_acq_dts_common = list(set.intersection(*map(set,
                                                 [acq_dts_viirs_fref,
                                                  acq_dts_viirs_f03,
                                                  acq_dts_viirs_fcld_l2])))

    if len(viirs_acq_dts_common) == 0:
        raise IndexError("Could not find any common date/times among products")

    fnames = modis_fnames + viirs_fnames
    # acq_dts_common = modis_acq_dts_common + viirs_acq_dts_common
    # print(len(fnames), len(acq_dts_common))
    fref, f03, fcld_l2, fcldmsk_l2 = [], [], [], []

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

            elif "06_L2" in pname:
                fcld_l2.append(os.path.join(fdir, f))

            elif "35_L2" in pname:
                fcldmsk_l2.append(os.path.join(fdir, f))

            else:
                pass

        elif acq_dt in viirs_acq_dts_common:

            if "02MOD" in pname:
                fref.append(os.path.join(fdir, f))

            elif "03MOD" in pname:
                f03.append(os.path.join(fdir, f))

            elif "CLDPROP_L2" in pname:
                fcld_l2.append(os.path.join(fdir, f))
                fcldmsk_l2.append(os.path.join(fdir, f))

            else:
                pass

    fref = sorted(fref, key=lambda x: x.split('.')[2])
    f03  = sorted(f03, key=lambda x: x.split('.')[2])
    fcld_l2 = sorted(fcld_l2, key=lambda x: x.split('.')[2])
    fcldmsk_l2 = sorted(fcldmsk_l2, key=lambda x: x.split('.')[2])
    return fref, f03, fcld_l2, fcldmsk_l2


def save_to_file_modis_viirs_ref_geo_cld_opt_msk(fdir, outdir, lon_1d, lat_1d, metadata, o_thresh=75):

    fref, f03, fcld_l2, fcldmsk_l2 = get_modis_viirs_ref_geo_cld_opt_mskfnames(fdir)
    if (len(fref) == 0) or (len(f03) == 0) or (len(fcld_l2) == 0) or (len(fcldmsk_l2) == 0):
        raise FileNotFoundError("\nCould not find the valid files\n")

    #print(len(fref), len(f03), len(fcld_l2), len(fcldmsk_l2))
    # if len(fref) == len(f03) == len(fcld_l2) == len(fcldmsk_l2):
    #     pass
    # else:
    #     print(len(fref), len(f03), len(fcld_l2), len(fcldmsk_l2))

    # return 1
    n_obs = len(f03)
    # acquisition date times
    # acq_dts = sorted([os.path.basename(i)[9:22] for i in fref])[-15:]
    print("Found {} overpasses".format(n_obs))
    extent = [lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()]
    ext_area = get_swath_area(lon_1d, lat_1d)

    valid_count = 0 # just for reporting

    with h5py.File(os.path.join(outdir, 'ref_geo_cld-data.h5'), 'w') as f0:
        # for acq_dt in acq_dts:
        for i in range(n_obs):
            sdata = {}
            # geometries/geolocation file
            # geo_file = list(filter(lambda x: acq_dt in x, f03))[0] # match acquisition to get geometry file
            geo_file    = f03[i]
            ref_file    = fref[i]
            cld_file    = fcld_l2[i]
            cldmsk_file = fcldmsk_l2[i]

            acq_dt = os.path.basename(geo_file).split('.')[1] + '.' + os.path.basename(geo_file).split('.')[2]

            try:
                if os.path.basename(geo_file).upper().startswith('M'): # modis
                    f_geo = modis_03(fnames=[geo_file], extent=extent)
                elif os.path.basename(geo_file).upper().startswith('V'): # viirs
                    f_geo = viirs_03(fnames=[geo_file], extent=extent)
                else:
                    raise NotImplementedError("\nOnly VIIRS and MODIS products are supported\n")
            # except HDF4Error:
            #     print("PyHDF error with {}...skipping...\n".format(geo_file))
            #     continue
            # except FileNotFoundError:
            #     print("netcdf error with {}...skipping...\n".format(geo_file))
            #     continue
            except:
                print("some error with {}...skipping...\n".format(geo_file))
                continue

            lon0 = f_geo.data['lon']['data']
            lat0 = f_geo.data['lat']['data']
            if len(lon0) == 0 or len(lat0) == 0:
                # print('Lat/lon not valid')
                continue

            mod_area = get_swath_area(lon0, lat0)
            overlap  = mod_area * 100. / ext_area
            if overlap < o_thresh:
                # print('Skipping {}, overlap {:0.1f} is less than {}%'.format(acq_dt, overlap, o_thresh))
                continue

            vza0 = f_geo.data['vza']['data']
            sza0 = f_geo.data['sza']['data']
            vaa0 = f_geo.data['vaa']['data']
            saa0 = f_geo.data['saa']['data']
            sdata['sza_2d'] = grid_by_lonlat(lon0, lat0, sza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['vza_2d'] = grid_by_lonlat(lon0, lat0, vza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['saa_2d'] = grid_by_lonlat(lon0, lat0, saa0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['vaa_2d'] = grid_by_lonlat(lon0, lat0, vaa0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)

            # reflectance file
            # ref_file = list(filter(lambda x: acq_dt in x, fref))[0]

            try:
                if os.path.basename(ref_file).upper().startswith('M'): # modis
                    f_vis = modis_l1b(fnames=[ref_file], extent=extent, f03=f_geo, bands=[1, 2, 3, 4, 6, 7])
                elif os.path.basename(ref_file).upper().startswith('V'): # viirs
                    f_vis = viirs_l1b(fnames=[ref_file], extent=extent, f03=f_geo, bands=["M05", "M07", "M02", "M04", "M10", "M11"])
                else:
                    print("\nOnly VIIRS and MODIS products are supported. Instead, file was: {}\n".format(ref_file))
                    continue

                dat = f_vis.data['ref']['data']

                # Band 7 (2130nm)
                sdata['data_2130'] = grid_by_lonlat(lon0, lat0, dat[5], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                # [650.0, 860.0, 470.0, 555.0, 1240.0, 1640.0, 2130.0]
                # Bands 6 (1640nm) and 1 (650nm) and 2 (860nm) and 3 (470nm)
                sdata['data_1640'] = grid_by_lonlat(lon0, lat0, dat[4], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['data_650'] = grid_by_lonlat(lon0, lat0, dat[0], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['data_860'] = grid_by_lonlat(lon0, lat0, dat[1], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['data_470'] = grid_by_lonlat(lon0, lat0, dat[2], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['data_555'] = grid_by_lonlat(lon0, lat0, dat[3], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)


                # cloud product
                # cld_file = list(filter(lambda x: acq_dt in x, fcld_l2))[0] # match aquisition to get cloud_l2 file
                if os.path.basename(cld_file).upper().startswith(('MOD06_L2', 'MYD06_L2')): # modis
                    f_cld = modis_l2(fnames=[cld_file], f03=f_geo, extent=extent)
                elif os.path.basename(cld_file).upper().startswith('CLDPROP'): # viirs
                    f_cld = viirs_cldprop_l2(fnames=[cld_file], f03=f_geo, extent=extent, maskvars=False)
                else:
                    print("\nOnly VIIRS and MODIS products are supported. Instead, file was: {}\n".format(cld_file))
                    continue

                cot0 = f_cld.data['cot']['data']
                cer0 = f_cld.data['cer']['data']
                cwp0 = f_cld.data['cwp']['data']
                cot_1621 = f_cld.data['cot_1621']['data']
                cer_1621 = f_cld.data['cer_1621']['data']
                cwp_1621 = f_cld.data['cwp_1621']['data']
                ctp0 = np.float64(f_cld.data['ctp']['data'])
                lon0 = f_cld.data['lon']['data']
                lat0 = f_cld.data['lat']['data']
                ctp_temp = grid_by_lonlat(lon0, lat0, ctp0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                # sdata['ctp'] = ctp_temp.astype('int')
                sdata['ctp'] = ctp_temp
                sdata['cot_2d'] = grid_by_lonlat(lon0, lat0, cot0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['cer_2d'] = grid_by_lonlat(lon0, lat0, cer0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['cwp_2d'] = grid_by_lonlat(lon0, lat0, cwp0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['cot_1621'] = grid_by_lonlat(lon0, lat0, cot_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['cer_1621'] = grid_by_lonlat(lon0, lat0, cer_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['cwp_1621'] = grid_by_lonlat(lon0, lat0, cwp_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)

                # cloud mask
                if os.path.basename(cldmsk_file).upper().startswith(('MOD35_L2', 'MYD35_L2')): # modis
                    f_cldmsk = modis_35_l2(fnames=[cldmsk_file], f03=f_geo, extent=extent)
                    conf_qa = np.float64(f_cldmsk.data['confidence_qa']['data'])
                # read in cldprop file again for viirs
                elif os.path.basename(cldmsk_file).upper().startswith('CLDPROP'): # viirs
                    f_cldmsk = viirs_cldprop_l2(fnames=[cldmsk_file], f03=f_geo, extent=extent, maskvars=True)
                    conf_qa = np.float64(f_cldmsk.data['ret_1621_conf_qa']['data'])
                else:
                    print("\nOnly VIIRS and MODIS products are supported. Instead, file was: {}\n".format(cldmsk_file))
                    continue


                cm_flag = np.float64(f_cldmsk.data['cloud_mask_flag']['data'])
                fov_qa = np.float64(f_cldmsk.data['fov_qa_cat']['data'])
                sunglint = np.float64(f_cldmsk.data['sunglint_flag']['data'])
                lon0 = f_cldmsk.data['lon']['data']
                lat0 = f_cldmsk.data['lat']['data']

                conf_qa_temp = grid_by_lonlat(lon0, lat0, conf_qa, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                cm_flag_temp = grid_by_lonlat(lon0, lat0, cm_flag, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                fov_qa_temp = grid_by_lonlat(lon0, lat0, fov_qa, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sunglint_temp = grid_by_lonlat(lon0, lat0, sunglint, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)

                sdata['conf_qa'] = conf_qa_temp
                sdata['cm_flag'] = cm_flag_temp
                sdata['fov_qa'] = fov_qa_temp
                sdata['sunglint'] = sunglint_temp

                # sdata['conf_qa'] = conf_qa_temp.astype('int')
                # sdata['cm_flag'] = cm_flag.astype('int')
                # sdata['fov_qa'] = fov_qa_temp.astype('int')
                # sdata['sunglint'] = sunglint_temp.astype('int')

                # f_cldmsk.data['snow_ice_flag']['data']
                # f_cldmsk.data['land_water_cat']['data']

                # satellite name
                if os.path.basename(ref_file)[:3] == "MOD":
                    satellite  = "Terra"
                    group_name = "T" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "MYD":
                    satellite  = "Aqua"
                    group_name = "A" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "VNP":
                    satellite  = "Suomi-NPP"
                    group_name = "S" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "VJ1":
                    satellite  = "NOAA-20/JPSS-1"
                    group_name = "J" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "VJ2":
                    satellite  = "NOAA-21/JPSS-2"
                    group_name = "N" + acq_dt[1:]
                else:
                    satellite  = "Unknown"
                    group_name = "Z" + acq_dt[1:]

                satellite = np.array(satellite.encode("utf-8"), dtype=utf8_type)

                # print(acq_dt, group_name, satellite)
                # print("-------------------")
                # save data
                g0 = f0.create_group(group_name)
                _ = g0.create_dataset('ref_2130', data=sdata['data_2130'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_1640', data=sdata['data_1640'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_860',  data=sdata['data_860'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_650',  data=sdata['data_650'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_555',  data=sdata['data_555'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_470',  data=sdata['data_470'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('sza_2d',   data=sdata['sza_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('vza_2d',   data=sdata['vza_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('saa_2d',   data=sdata['saa_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('vaa_2d',   data=sdata['vaa_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ctp',      data=sdata['ctp'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('cot_2d',   data=sdata['cot_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('cer_2d',   data=sdata['cer_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('cwp_2d',   data=sdata['cwp_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('cot_1621', data=sdata['cot_1621'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('cer_1621', data=sdata['cer_1621'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('cwp_1621', data=sdata['cwp_1621'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('conf_qa',  data=sdata['conf_qa'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('cm_flag',  data=sdata['cm_flag'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('fov_qa',   data=sdata['fov_qa'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('sunglint', data=sdata['sunglint'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('lon_1d',   data=lon_1d, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('lat_1d',   data=lat_1d, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('metadata', data=metadata)
                _ = g0.create_dataset('satellite',data=satellite)

                valid_count += 1
            except Exception as err:
                print(err)
                continue

    print("Successfully extracted data from {} of {} overpasses".format(valid_count, n_obs))
    return 1




################################################################################################################

def save_to_file_viirs_only_ref_geo(fdir, outdir, lon_1d, lat_1d, metadata, o_thresh=75):
    fref, f03 = get_viirs_ref_geo(fdir)
    if (len(fref) == 0) or (len(f03) == 0):
        raise FileNotFoundError("\nCould not find the valid files\n")

    #print(len(fref), len(f03), len(fcld_l2), len(fcldmsk_l2))
    # if len(fref) == len(f03) == len(fcld_l2) == len(fcldmsk_l2):
    #     pass
    # else:
    #     print(len(fref), len(f03), len(fcld_l2), len(fcldmsk_l2))

    # return 1
    n_obs = len(f03)
    # acquisition date times
    # acq_dts = sorted([os.path.basename(i)[9:22] for i in fref])[-15:]
    print("Found {} overpasses".format(n_obs))
    extent = [lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()]
    ext_area = get_swath_area(lon_1d, lat_1d)

    valid_count = 0 # just for reporting

    with h5py.File(os.path.join(outdir, 'ref_geo_no-cld-data.h5'), 'w') as f0:
        # for acq_dt in acq_dts:
        for i in range(n_obs):
            sdata = {}
            # geometries/geolocation file
            # geo_file = list(filter(lambda x: acq_dt in x, f03))[0] # match acquisition to get geometry file
            geo_file    = f03[i]
            ref_file    = fref[i]

            acq_dt = os.path.basename(geo_file).split('.')[1] + '.' + os.path.basename(geo_file).split('.')[2]

            try:
                if os.path.basename(geo_file).upper().startswith('M'): # modis
                    f_geo = modis_03(fnames=[geo_file], extent=extent)
                elif os.path.basename(geo_file).upper().startswith('V'): # viirs
                    f_geo = viirs_03(fnames=[geo_file], extent=extent)
                else:
                    raise NotImplementedError("\nOnly VIIRS and MODIS products are supported\n")
            # except HDF4Error:
            #     print("PyHDF error with {}...skipping...\n".format(geo_file))
            #     continue
            # except FileNotFoundError:
            #     print("netcdf error with {}...skipping...\n".format(geo_file))
            #     continue
            except:
                print("some error with {}...skipping...\n".format(geo_file))
                continue

            lon0 = f_geo.data['lon']['data']
            lat0 = f_geo.data['lat']['data']
            if len(lon0) == 0 or len(lat0) == 0:
                # print('Lat/lon not valid')
                continue

            mod_area = get_swath_area(lon0, lat0)
            overlap  = mod_area * 100. / ext_area
            if overlap < o_thresh:
                # print('Skipping {}, overlap {:0.1f} is less than {}%'.format(acq_dt, overlap, o_thresh))
                continue

            vza0 = f_geo.data['vza']['data']
            sza0 = f_geo.data['sza']['data']
            vaa0 = f_geo.data['vaa']['data']
            saa0 = f_geo.data['saa']['data']
            sdata['sza_2d'] = grid_by_lonlat(lon0, lat0, sza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['vza_2d'] = grid_by_lonlat(lon0, lat0, vza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['saa_2d'] = grid_by_lonlat(lon0, lat0, saa0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['vaa_2d'] = grid_by_lonlat(lon0, lat0, vaa0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)

            # reflectance file
            # ref_file = list(filter(lambda x: acq_dt in x, fref))[0]

            try:
                if os.path.basename(ref_file).upper().startswith('M'): # modis
                    f_vis = modis_l1b(fnames=[ref_file], extent=extent, f03=f_geo, bands=[1, 2, 3, 4, 6, 7])
                elif os.path.basename(ref_file).upper().startswith('V'): # viirs
                    f_vis = viirs_l1b(fnames=[ref_file], extent=extent, f03=f_geo, bands=["M05", "M07", "M02", "M04", "M10", "M11"])
                else:
                    print("\nOnly VIIRS and MODIS products are supported. Instead, file was: {}\n".format(ref_file))
                    continue

                dat = f_vis.data['ref']['data']

                # Band 7 (2130nm)
                sdata['data_2130'] = grid_by_lonlat(lon0, lat0, dat[5], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                # [650.0, 860.0, 470.0, 555.0, 1240.0, 1640.0, 2130.0]
                # Bands 6 (1640nm) and 1 (650nm) and 2 (860nm) and 3 (470nm)
                sdata['data_1640'] = grid_by_lonlat(lon0, lat0, dat[4], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['data_650'] = grid_by_lonlat(lon0, lat0, dat[0], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['data_860'] = grid_by_lonlat(lon0, lat0, dat[1], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['data_470'] = grid_by_lonlat(lon0, lat0, dat[2], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['data_555'] = grid_by_lonlat(lon0, lat0, dat[3], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)


                # satellite name
                if os.path.basename(ref_file)[:3] == "MOD":
                    satellite  = "Terra"
                    group_name = "T" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "MYD":
                    satellite  = "Aqua"
                    group_name = "A" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "VNP":
                    satellite  = "Suomi-NPP"
                    group_name = "S" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "VJ1":
                    satellite  = "NOAA-20/JPSS-1"
                    group_name = "J" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "VJ2":
                    satellite  = "NOAA-21/JPSS-2"
                    group_name = "N" + acq_dt[1:]
                else:
                    satellite  = "Unknown"
                    group_name = "Z" + acq_dt[1:]

                satellite = np.array(satellite.encode("utf-8"), dtype=utf8_type)

                # print(acq_dt, group_name, satellite)
                # print("-------------------")
                # save data
                g0 = f0.create_group(group_name)
                _ = g0.create_dataset('ref_2130', data=sdata['data_2130'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_1640', data=sdata['data_1640'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_860',  data=sdata['data_860'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_650',  data=sdata['data_650'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_555',  data=sdata['data_555'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_470',  data=sdata['data_470'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('sza_2d',   data=sdata['sza_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('vza_2d',   data=sdata['vza_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('saa_2d',   data=sdata['saa_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('vaa_2d',   data=sdata['vaa_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('lon_1d',   data=lon_1d, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('lat_1d',   data=lat_1d, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('metadata', data=metadata)
                _ = g0.create_dataset('satellite',data=satellite)

                valid_count += 1

            except Exception as err:
                print(err)
                continue

    print("Successfully extracted data from {} of {} overpasses".format(valid_count, n_obs))
    return 1


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


def save_to_file_modis_only_geo_cld_opt(fdir, outdir, lon_1d, lat_1d, metadata, start_dt, end_dt, o_thresh=75):

    f03, fcld_l2 = get_modis_noref_geo_cld_opt(fdir)
    if (len(f03) == 0) or (len(fcld_l2) == 0):
        print("\nMessage [create_training_data]: Could not find any common ref + cld products for MODIS\n")
        return 0

    n_obs = len(f03)

    # end_dt = get_last_opass(f03)
    # start_dt = end_dt - datetime.timedelta(hours=nhours)

    # acquisition date times
    # acq_dts = sorted([os.path.basename(i)[9:22] for i in fref])[-15:]
    print("Found {} overpasses".format(n_obs))
    extent = [lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()]
    ext_area = get_swath_area(lon_1d, lat_1d)

    valid_count = 0 # just for reporting

    start_dt_str = start_dt.strftime('%Y-%m-%d-%H%M')
    end_dt_str   = end_dt.strftime('%Y-%m-%d-%H%M')
    outdir_dt =  start_dt_str + '_' + end_dt_str

    # get all subdirs
    subdirs = sorted([ff for ff in os.listdir(outdir) if len(os.listdir(os.path.join(outdir, ff))) > 0])
    if len(subdirs) > 2:
        subdirs = subdirs[-2:]

    # get only those that have the reflectance geo file
    exist_acq_dts = []
    for i in range(len(subdirs)):
        h5dir = os.path.join(outdir, subdirs[i])
        if (os.path.isdir(h5dir)) and ('geo_cld-data.h5' in os.listdir(h5dir)):
            h5file = os.path.join(h5dir, 'geo_cld-data.h5')
            try:
                with h5py.File(h5file, 'r') as f:
                    keys = list(f.keys())
                    for key in keys:
                        exist_acq_dts.append(key[1:]) # disregard first letter since it is now satellite coded
            except Exception as err:
                print(err, "Message [modis_cld_geo]: No older files will be used")


    outdir = os.path.join(outdir, outdir_dt)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # purposely change start date to account for latency but not affect filenames
    start_dt = start_dt - datetime.timedelta(hours=2)
    start_dt_str = start_dt.strftime('%Y-%m-%d-%H%M')
    print("Message [create_training_data]: Start datetime:{}, End datetime: {}".format(start_dt_str, end_dt_str))
    print("Warning [create_training_data]: Start datetime was changed to catch more data and account for latency")

    with h5py.File(os.path.join(outdir, 'geo_cld-data.h5'), 'w') as f1:
        # for acq_dt in acq_dts:
        for i in range(n_obs):
            try:

                # geometries/geolocation file
                # geometries/geolocation file
                # geo_file = list(filter(lambda x: acq_dt in x, f03))[0] # match acquisition to get geometry file
                geo_file    = f03[i]

                # check if the file has already been processed
                bgeo_file = os.path.basename(geo_file)
                yyyydoy = bgeo_file.split('.')[1][1:] + '.' + bgeo_file.split('.')[2]

                print("Message [modis_cld_geo]: yyyydoy", yyyydoy)
                if yyyydoy in exist_acq_dts: # if already processed, then skip
                    print("Message [modis_cld_geo]: Skipping {} as it has likely already been processed previously".format(geo_file))
                    continue


                if not within_range(geo_file, start_dt, end_dt):
                    print("Message [modis_cld_geo]: Skipping {} as it is outside the provided date range: {} to {}".format(geo_file, start_dt.strftime("%Y-%m-%d:%H%M"), end_dt.strftime("%Y-%m-%d:%H%M")))
                    continue

                cld_file    = fcld_l2[i]

                acq_dt = os.path.basename(geo_file).split('.')[1] + '.' + os.path.basename(geo_file).split('.')[2]

                if os.path.basename(geo_file).upper().startswith('M'): # modis
                    f_geo = modis_03(fnames=[geo_file], extent=extent)

            # except HDF4Error:
            #     print("PyHDF error with {}...skipping...\n".format(geo_file))
            #     continue
            # except FileNotFoundError:
            #     print("netcdf error with {}...skipping...\n".format(geo_file))
            #     continue
            except Exception as err:
                print("some error with {}...skipping...\nError {}\n".format(geo_file, err))
                continue

            lon0 = f_geo.data['lon']['data']
            lat0 = f_geo.data['lat']['data']
            if len(lon0) == 0 or len(lat0) == 0:
                # print('Lat/lon not valid')
                continue

            # mod_area = get_swath_area(lon0, lat0)
            # overlap  = mod_area * 100. / ext_area
            # if overlap < o_thresh:
            #     # print('Skipping {}, overlap {:0.1f} is less than {}%'.format(acq_dt, overlap, o_thresh))
            #     continue

            sdata = {}

            vza0 = f_geo.data['vza']['data']
            sza0 = f_geo.data['sza']['data']
            vaa0 = f_geo.data['vaa']['data']
            saa0 = f_geo.data['saa']['data']
            sdata['sza_2d'] = grid_by_lonlat(lon0, lat0, sza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['vza_2d'] = grid_by_lonlat(lon0, lat0, vza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['saa_2d'] = grid_by_lonlat(lon0, lat0, saa0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['vaa_2d'] = grid_by_lonlat(lon0, lat0, vaa0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)

            # reflectance file
            # ref_file = list(filter(lambda x: acq_dt in x, fref))[0]

            try:
                # cloud product
                # cld_file = list(filter(lambda x: acq_dt in x, fcld_l2))[0] # match aquisition to get cloud_l2 file
                if os.path.basename(cld_file).upper().startswith(('MOD06_L2', 'MYD06_L2')): # modis
                    f_cld = modis_l2(fnames=[cld_file], f03=f_geo, extent=extent)


                cot0 = f_cld.data['cot']['data']
                cer0 = f_cld.data['cer']['data']
                cwp0 = f_cld.data['cwp']['data']
                cot_1621 = f_cld.data['cot_1621']['data']
                cer_1621 = f_cld.data['cer_1621']['data']
                cwp_1621 = f_cld.data['cwp_1621']['data']
                ctp0 = np.float64(f_cld.data['ctp']['data'])
                lon0 = f_cld.data['lon']['data']
                lat0 = f_cld.data['lat']['data']
                ctp_temp = grid_by_lonlat(lon0, lat0, ctp0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=10, return_geo=False)
                # sdata['ctp'] = ctp_temp.astype('int')
                sdata['ctp'] = ctp_temp
                sdata['cot_2d'] = grid_by_lonlat(lon0, lat0, cot0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=10, return_geo=False)
                sdata['cer_2d'] = grid_by_lonlat(lon0, lat0, cer0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=10, return_geo=False)
                sdata['cwp_2d'] = grid_by_lonlat(lon0, lat0, cwp0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=10, return_geo=False)
                sdata['cot_1621'] = grid_by_lonlat(lon0, lat0, cot_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=10, return_geo=False)
                sdata['cer_1621'] = grid_by_lonlat(lon0, lat0, cer_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=10, return_geo=False)
                sdata['cwp_1621'] = grid_by_lonlat(lon0, lat0, cwp_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=10, return_geo=False)

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

                satellite = np.array(satellite.encode("utf-8"), dtype=utf8_type)

                # print(acq_dt, group_name, satellite)
                # print("-------------------")
                # save data
                g1 = f1.create_group(group_name)
                _ = g1.create_dataset('sza_2d',   data=sdata['sza_2d'].T, compression='gzip', compression_opts=9)
                _ = g1.create_dataset('vza_2d',   data=sdata['vza_2d'].T, compression='gzip', compression_opts=9)
                _ = g1.create_dataset('saa_2d',   data=sdata['saa_2d'].T, compression='gzip', compression_opts=9)
                _ = g1.create_dataset('vaa_2d',   data=sdata['vaa_2d'].T, compression='gzip', compression_opts=9)
                _ = g1.create_dataset('ctp',      data=sdata['ctp'].T, compression='gzip', compression_opts=9)
                _ = g1.create_dataset('cot_2d',   data=sdata['cot_2d'].T, compression='gzip', compression_opts=9)
                _ = g1.create_dataset('cer_2d',   data=sdata['cer_2d'].T, compression='gzip', compression_opts=9)
                _ = g1.create_dataset('cwp_2d',   data=sdata['cwp_2d'].T, compression='gzip', compression_opts=9)
                _ = g1.create_dataset('cot_1621', data=sdata['cot_1621'].T, compression='gzip', compression_opts=9)
                _ = g1.create_dataset('cer_1621', data=sdata['cer_1621'].T, compression='gzip', compression_opts=9)
                _ = g1.create_dataset('cwp_1621', data=sdata['cwp_1621'].T, compression='gzip', compression_opts=9)
                _ = g1.create_dataset('lon_1d',   data=lon_1d, compression='gzip', compression_opts=9)
                _ = g1.create_dataset('lat_1d',   data=lat_1d, compression='gzip', compression_opts=9)
                _ = g1.create_dataset('metadata', data=metadata)
                _ = g1.create_dataset('satellite',data=satellite)

                valid_count += 1

            except Exception as err:
                print(err)
                continue

    if valid_count == 0:
        print("Could not extract data. Deleting the HDF5 file...\n")
        os.remove(os.path.join(outdir, 'geo_cld-data.h5'))
        return 0
    else:
        print("{}: Successfully extracted data from {} of {} overpasses".format(os.path.join(outdir, 'geo_cld-data.h5'), valid_count, n_obs))
        return 1





#########################################################################################################

def save_to_file_modis_only_ref_geo_cld_opt_msk(fdir, outdir, lon_1d, lat_1d, metadata, o_thresh=75):

    fref, f03, fcld_l2, fcldmsk_l2 = get_modis_ref_geo_cld_opt_mskfnames(fdir)
    if (len(fref) == 0) or (len(f03) == 0) or (len(fcld_l2) == 0) or (len(fcldmsk_l2) == 0):
        raise FileNotFoundError("\nCould not find the valid files\n")

    #print(len(fref), len(f03), len(fcld_l2), len(fcldmsk_l2))
    # if len(fref) == len(f03) == len(fcld_l2) == len(fcldmsk_l2):
    #     pass
    # else:
    #     print(len(fref), len(f03), len(fcld_l2), len(fcldmsk_l2))

    # return 1
    n_obs = len(f03)
    # acquisition date times
    # acq_dts = sorted([os.path.basename(i)[9:22] for i in fref])[-15:]
    print("Found {} overpasses".format(n_obs))
    extent = [lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()]
    ext_area = get_swath_area(lon_1d, lat_1d)

    valid_count = 0 # just for reporting

    with h5py.File(os.path.join(outdir, 'ref_geo_cld-data.h5'), 'w') as f0:
        # for acq_dt in acq_dts:
        for i in range(n_obs):
            sdata = {}
            # geometries/geolocation file
            # geo_file = list(filter(lambda x: acq_dt in x, f03))[0] # match acquisition to get geometry file
            geo_file    = f03[i]
            ref_file    = fref[i]
            cld_file    = fcld_l2[i]
            cldmsk_file = fcldmsk_l2[i]

            acq_dt = os.path.basename(geo_file).split('.')[1] + '.' + os.path.basename(geo_file).split('.')[2]

            try:
                if os.path.basename(geo_file).upper().startswith('M'): # modis
                    f_geo = modis_03(fnames=[geo_file], extent=extent)
                elif os.path.basename(geo_file).upper().startswith('V'): # viirs
                    f_geo = viirs_03(fnames=[geo_file], extent=extent)
                else:
                    raise NotImplementedError("\nOnly VIIRS and MODIS products are supported\n")
            # except HDF4Error:
            #     print("PyHDF error with {}...skipping...\n".format(geo_file))
            #     continue
            # except FileNotFoundError:
            #     print("netcdf error with {}...skipping...\n".format(geo_file))
            #     continue
            except:
                print("some error with {}...skipping...\n".format(geo_file))
                continue

            lon0 = f_geo.data['lon']['data']
            lat0 = f_geo.data['lat']['data']
            if len(lon0) == 0 or len(lat0) == 0:
                # print('Lat/lon not valid')
                continue

            mod_area = get_swath_area(lon0, lat0)
            overlap  = mod_area * 100. / ext_area
            if overlap < o_thresh:
                # print('Skipping {}, overlap {:0.1f} is less than {}%'.format(acq_dt, overlap, o_thresh))
                continue

            vza0 = f_geo.data['vza']['data']
            sza0 = f_geo.data['sza']['data']
            vaa0 = f_geo.data['vaa']['data']
            saa0 = f_geo.data['saa']['data']
            sdata['sza_2d'] = grid_by_lonlat(lon0, lat0, sza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['vza_2d'] = grid_by_lonlat(lon0, lat0, vza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['saa_2d'] = grid_by_lonlat(lon0, lat0, saa0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['vaa_2d'] = grid_by_lonlat(lon0, lat0, vaa0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)

            # reflectance file
            # ref_file = list(filter(lambda x: acq_dt in x, fref))[0]

            try:
                if os.path.basename(ref_file).upper().startswith('M'): # modis
                    f_vis = modis_l1b(fnames=[ref_file], extent=extent, f03=f_geo, bands=[1, 2, 3, 4, 6, 7])
                elif os.path.basename(ref_file).upper().startswith('V'): # viirs
                    f_vis = viirs_l1b(fnames=[ref_file], extent=extent, f03=f_geo, bands=["M05", "M07", "M02", "M04", "M10", "M11"])
                else:
                    print("\nOnly VIIRS and MODIS products are supported. Instead, file was: {}\n".format(ref_file))
                    continue

                dat = f_vis.data['ref']['data']

                # Band 7 (2130nm)
                sdata['data_2130'] = grid_by_lonlat(lon0, lat0, dat[5], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                # [650.0, 860.0, 470.0, 555.0, 1240.0, 1640.0, 2130.0]
                # Bands 6 (1640nm) and 1 (650nm) and 2 (860nm) and 3 (470nm)
                sdata['data_1640'] = grid_by_lonlat(lon0, lat0, dat[4], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['data_650'] = grid_by_lonlat(lon0, lat0, dat[0], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['data_860'] = grid_by_lonlat(lon0, lat0, dat[1], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['data_470'] = grid_by_lonlat(lon0, lat0, dat[2], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['data_555'] = grid_by_lonlat(lon0, lat0, dat[3], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)


                # cloud product
                # cld_file = list(filter(lambda x: acq_dt in x, fcld_l2))[0] # match aquisition to get cloud_l2 file
                if os.path.basename(cld_file).upper().startswith(('MOD06_L2', 'MYD06_L2')): # modis
                    f_cld = modis_l2(fnames=[cld_file], f03=f_geo, extent=extent)
                elif os.path.basename(cld_file).upper().startswith('CLDPROP'): # viirs
                    f_cld = viirs_cldprop_l2(fnames=[cld_file], f03=f_geo, extent=extent, maskvars=False)
                else:
                    print("\nOnly VIIRS and MODIS products are supported. Instead, file was: {}\n".format(cld_file))
                    continue

                cot0 = f_cld.data['cot']['data']
                cer0 = f_cld.data['cer']['data']
                cwp0 = f_cld.data['cwp']['data']
                cot_1621 = f_cld.data['cot_1621']['data']
                cer_1621 = f_cld.data['cer_1621']['data']
                cwp_1621 = f_cld.data['cwp_1621']['data']
                ctp0 = np.float64(f_cld.data['ctp']['data'])
                lon0 = f_cld.data['lon']['data']
                lat0 = f_cld.data['lat']['data']
                ctp_temp = grid_by_lonlat(lon0, lat0, ctp0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                # sdata['ctp'] = ctp_temp.astype('int')
                sdata['ctp'] = ctp_temp
                sdata['cot_2d'] = grid_by_lonlat(lon0, lat0, cot0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['cer_2d'] = grid_by_lonlat(lon0, lat0, cer0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['cwp_2d'] = grid_by_lonlat(lon0, lat0, cwp0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['cot_1621'] = grid_by_lonlat(lon0, lat0, cot_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['cer_1621'] = grid_by_lonlat(lon0, lat0, cer_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sdata['cwp_1621'] = grid_by_lonlat(lon0, lat0, cwp_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)

                # cloud mask
                if os.path.basename(cldmsk_file).upper().startswith(('MOD35_L2', 'MYD35_L2')): # modis
                    f_cldmsk = modis_35_l2(fnames=[cldmsk_file], f03=f_geo, extent=extent)
                    conf_qa = np.float64(f_cldmsk.data['confidence_qa']['data'])
                # read in cldprop file again for viirs
                elif os.path.basename(cldmsk_file).upper().startswith('CLDPROP'): # viirs
                    f_cldmsk = viirs_cldprop_l2(fnames=[cldmsk_file], f03=f_geo, extent=extent, maskvars=True)
                    conf_qa = np.float64(f_cldmsk.data['ret_1621_conf_qa']['data'])
                else:
                    print("\nOnly VIIRS and MODIS products are supported. Instead, file was: {}\n".format(cldmsk_file))
                    continue


                cm_flag = np.float64(f_cldmsk.data['cloud_mask_flag']['data'])
                fov_qa = np.float64(f_cldmsk.data['fov_qa_cat']['data'])
                sunglint = np.float64(f_cldmsk.data['sunglint_flag']['data'])
                lon0 = f_cldmsk.data['lon']['data']
                lat0 = f_cldmsk.data['lat']['data']

                conf_qa_temp = grid_by_lonlat(lon0, lat0, conf_qa, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                cm_flag_temp = grid_by_lonlat(lon0, lat0, cm_flag, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                fov_qa_temp = grid_by_lonlat(lon0, lat0, fov_qa, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
                sunglint_temp = grid_by_lonlat(lon0, lat0, sunglint, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)

                sdata['conf_qa'] = conf_qa_temp
                sdata['cm_flag'] = cm_flag_temp
                sdata['fov_qa'] = fov_qa_temp
                sdata['sunglint'] = sunglint_temp

                # sdata['conf_qa'] = conf_qa_temp.astype('int')
                # sdata['cm_flag'] = cm_flag.astype('int')
                # sdata['fov_qa'] = fov_qa_temp.astype('int')
                # sdata['sunglint'] = sunglint_temp.astype('int')

                # f_cldmsk.data['snow_ice_flag']['data']
                # f_cldmsk.data['land_water_cat']['data']

                # satellite name
                if os.path.basename(ref_file)[:3] == "MOD":
                    satellite  = "Terra"
                    group_name = "T" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "MYD":
                    satellite  = "Aqua"
                    group_name = "A" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "VNP":
                    satellite  = "Suomi-NPP"
                    group_name = "S" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "VJ1":
                    satellite  = "NOAA-20/JPSS-1"
                    group_name = "J" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "VJ2":
                    satellite  = "NOAA-21/JPSS-2"
                    group_name = "N" + acq_dt[1:]
                else:
                    satellite  = "Unknown"
                    group_name = "Z" + acq_dt[1:]

                satellite = np.array(satellite.encode("utf-8"), dtype=utf8_type)

                # print(acq_dt, group_name, satellite)
                # print("-------------------")
                # save data
                g0 = f0.create_group(group_name)
                _ = g0.create_dataset('ref_2130', data=sdata['data_2130'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_1640', data=sdata['data_1640'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_860',  data=sdata['data_860'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_650',  data=sdata['data_650'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_555',  data=sdata['data_555'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_470',  data=sdata['data_470'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('sza_2d',   data=sdata['sza_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('vza_2d',   data=sdata['vza_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('saa_2d',   data=sdata['saa_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('vaa_2d',   data=sdata['vaa_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ctp',      data=sdata['ctp'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('cot_2d',   data=sdata['cot_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('cer_2d',   data=sdata['cer_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('cwp_2d',   data=sdata['cwp_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('cot_1621', data=sdata['cot_1621'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('cer_1621', data=sdata['cer_1621'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('cwp_1621', data=sdata['cwp_1621'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('conf_qa',  data=sdata['conf_qa'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('cm_flag',  data=sdata['cm_flag'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('fov_qa',   data=sdata['fov_qa'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('sunglint', data=sdata['sunglint'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('lon_1d',   data=lon_1d, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('lat_1d',   data=lat_1d, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('metadata', data=metadata)
                _ = g0.create_dataset('satellite',data=satellite)

                valid_count += 1

            except Exception as err:
                print(err)
                # continue

    print("Successfully extracted data from {} of {} overpasses".format(valid_count, n_obs))
    return 1



def get_modis_ref_geo_cld_opt_mskfnames(fdir):

    # MODIS files
    modis_fnames = get_all_modis_files(fdir)
    _, acq_dts_modis_fref = get_product_info(modis_fnames, product_id="1KM")
    _, acq_dts_ft03 = get_product_info(modis_fnames, product_id="MOD03")
    _, acq_dts_fa03 = get_product_info(modis_fnames, product_id="MYD03")
    acq_dts_modis_f03 = acq_dts_ft03 + acq_dts_fa03
    # modis_f03 = temp1 + temp2
    _, acq_dts_modis_f06_l2 = get_product_info(modis_fnames, product_id="06_L2")
    _, acq_dts_modis_f35_l2 = get_product_info(modis_fnames, product_id="35_L2")

    modis_acq_dts_common = list(set.intersection(*map(set,
                                                 [acq_dts_modis_fref,
                                                  acq_dts_modis_f03,
                                                  acq_dts_modis_f06_l2,
                                                  acq_dts_modis_f35_l2])))

    if len(modis_acq_dts_common) == 0:
        raise IndexError("Could not find any common date/times among products")

    fnames = modis_fnames
    # acq_dts_common = modis_acq_dts_common + viirs_acq_dts_common
    # print(len(fnames), len(acq_dts_common))
    fref, f03, fcld_l2, fcldmsk_l2 = [], [], [], []

    for f in fnames:
        filename = os.path.basename(f).upper()
        # acq_dt = filename.split(".")[1] + '.' + filename.split(".")[2]
        pname  = filename.split(".")[0]

        if "1KM" in pname:
            fref.append(os.path.join(fdir, f))

        elif ("MOD03" in pname) or ("MYD03" in pname):
            f03.append(os.path.join(fdir, f))

        elif "06_L2" in pname:
            fcld_l2.append(os.path.join(fdir, f))

        elif "35_L2" in pname:
            fcldmsk_l2.append(os.path.join(fdir, f))

        else:
            pass

    fref = sorted(fref, key=lambda x: x.split('.')[2])
    f03  = sorted(f03, key=lambda x: x.split('.')[2])
    fcld_l2 = sorted(fcld_l2, key=lambda x: x.split('.')[2])
    fcldmsk_l2 = sorted(fcldmsk_l2, key=lambda x: x.split('.')[2])
    return fref, f03, fcld_l2, fcldmsk_l2

###################################################################################################################

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


def save_to_file_modis_viirs_ref_geo(fdir, outdir, lon_1d, lat_1d, metadata, start_dt, end_dt, o_thresh=75):

    fref, f03 = get_modis_viirs_ref_geo(fdir)
    if (len(fref) == 0) or (len(f03) == 0):
        print("Message [create_training_data]: Could not find any common products for both MODIS and VIIRS")
        return 0

    n_obs = len(f03)

    # end_dt = get_last_opass(f03)
    # start_dt = end_dt - datetime.timedelta(hours=nhours)

    # acquisition date times
    # acq_dts = sorted([os.path.basename(i)[9:22] for i in fref])[-15:]
    print("Found {} overpasses".format(n_obs))
    extent = [lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()]
    ext_area = get_swath_area(lon_1d, lat_1d)

    valid_count = 0 # just for reporting

    start_dt_str = start_dt.strftime('%Y-%m-%d-%H%M')
    end_dt_str   = end_dt.strftime('%Y-%m-%d-%H%M')
    outdir_dt =  start_dt_str + '_' + end_dt_str

    print("Message [modis_viirs_ref_geo]: Start datetime:{}, End datetime: {}".format(start_dt_str, end_dt_str))

    # get all subdirs
    subdirs = sorted([ff for ff in os.listdir(outdir) if len(os.listdir(os.path.join(outdir, ff))) > 0])
    if len(subdirs) > 2:
        subdirs = subdirs[-2:]

    # get only those that have the reflectance geo file
    exist_acq_dts = []
    for i in range(len(subdirs)):
        h5dir = os.path.join(outdir, subdirs[i])
        if (os.path.isdir(h5dir)) and ('ref_geo-data.h5' in os.listdir(h5dir)):
            h5file = os.path.join(h5dir, 'ref_geo-data.h5')
            try:
                with h5py.File(h5file, 'r') as f:
                    keys = list(f.keys())
                    for key in keys:
                        exist_acq_dts.append(key[1:]) # disregard first letter since it is now satellite coded
            except Exception as err:
                print(err, "Message [modis_viirs_ref_geo]: No older files will be used")


    print("Message [modis_viirs_ref_geo]: Already processed", exist_acq_dts)
    outdir = os.path.join(outdir, outdir_dt)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with h5py.File(os.path.join(outdir, 'ref_geo-data.h5'), 'w') as f0:
        # for acq_dt in acq_dts:
        for i in range(n_obs):
            try:
                # geometries/geolocation file
                # geo_file = list(filter(lambda x: acq_dt in x, f03))[0] # match acquisition to get geometry file
                geo_file    = f03[i]

                # check if the file has already been processed
                bgeo_file = os.path.basename(geo_file)
                yyyydoy = bgeo_file.split('.')[1][1:] + '.' + bgeo_file.split('.')[2]

                print("Message [modis_viirs_ref_geo]: yyyydoy", yyyydoy)
                if yyyydoy in exist_acq_dts: # if already processed, then skip
                    print("Message [modis_viirs_ref_geo]: Skipping {} as it has likely already been processed previously".format(geo_file))
                    continue

                ########################################################################################################################
                # # for VIIRS latency add an extra hour if necessary:
                if os.path.basename(geo_file).startswith('V'):

                    if (end_dt - start_dt) < datetime.timedelta(hours=2):
                        viirs_start_dt = start_dt - datetime.timedelta(hours=1)
                    elif ((end_dt - start_dt) > datetime.timedelta(hours=3)):
                        viirs_start_dt = end_dt - datetime.timedelta(hours=3)
                        print("Message [create_training_data]: Start time and end times were too far apart...limiting to 3 hour gap.")
                    else:
                        viirs_start_dt = start_dt

                    if not within_range(geo_file, viirs_start_dt, end_dt):
                        print("Message [modis_viirs_ref_geo]: VIIRS: Skipping {} as it is outside the provided date range: {} to {}".format(geo_file, viirs_start_dt.strftime("%Y-%m-%d:%H%M"), end_dt.strftime("%Y-%m-%d:%H%M")))
                        continue

                else: # for MODIS latency
                    if not within_range(geo_file, start_dt, end_dt):
                        print("Message [modis_viirs_ref_geo]: MODIS: Skipping {} as it is outside the provided date range: {} to {}".format(geo_file, start_dt.strftime("%Y-%m-%d:%H%M"), end_dt.strftime("%Y-%m-%d:%H%M")))
                        continue


                # print("{} within range".format(os.path.basename(geo_file)))

                ########################################################################################################################

                ref_file    = fref[i]

                acq_dt = os.path.basename(geo_file).split('.')[1] + '.' + os.path.basename(geo_file).split('.')[2]

                if os.path.basename(geo_file).upper().startswith('M'): # modis
                    f_geo = modis_03(fnames=[geo_file], extent=extent)
                elif os.path.basename(geo_file).upper().startswith('V'): # viirs
                    f_geo = viirs_03(fnames=[geo_file], extent=extent)
                else:
                    raise NotImplementedError("\nOnly VIIRS and MODIS products are supported\n")
            except Exception as err:
                print("some error: with {}...skipping...\nError: {}\n".format(geo_file, err))
                continue

            lon0 = f_geo.data['lon']['data']
            lat0 = f_geo.data['lat']['data']
            if len(lon0) == 0 or len(lat0) == 0:
                # print('Lat/lon not valid')
                continue

            # mod_area = get_swath_area(lon0, lat0)
            # overlap  = mod_area * 100. / ext_area
            # if overlap < o_thresh:
            #     # print('Skipping {}, overlap {:0.1f} is less than {}%'.format(acq_dt, overlap, o_thresh))
            #     continue

            sdata = {}

            vza0 = f_geo.data['vza']['data']
            sza0 = f_geo.data['sza']['data']
            vaa0 = f_geo.data['vaa']['data']
            saa0 = f_geo.data['saa']['data']
            sdata['sza_2d'] = grid_by_lonlat(lon0, lat0, sza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['vza_2d'] = grid_by_lonlat(lon0, lat0, vza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['saa_2d'] = grid_by_lonlat(lon0, lat0, saa0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
            sdata['vaa_2d'] = grid_by_lonlat(lon0, lat0, vaa0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)

            # reflectance file
            # ref_file = list(filter(lambda x: acq_dt in x, fref))[0]

            try:
                if os.path.basename(ref_file).upper().startswith('M'): # modis
                    f_vis = modis_l1b(fnames=[ref_file], extent=extent, f03=f_geo, bands=[1, 2, 3, 4, 6, 7])

                elif os.path.basename(ref_file).upper().startswith('V'): # viirs
                    f_vis = viirs_l1b(fnames=[ref_file], extent=extent, f03=f_geo, bands=["M05", "M07", "M02", "M04", "M10", "M11"])
                else:
                    print("\nOnly VIIRS and MODIS products are supported. Instead, file was: {}\n".format(ref_file))
                    continue

                dat = f_vis.data['ref']['data']

                # Band 7 (2130nm)
                sdata['data_2130'] = grid_by_lonlat(lon0, lat0, dat[5], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=10, return_geo=False)
                # [650.0, 860.0, 470.0, 555.0, 1240.0, 1640.0, 2130.0]
                # Bands 6 (1640nm) and 1 (650nm) and 2 (860nm) and 3 (470nm)
                sdata['data_1640'] = grid_by_lonlat(lon0, lat0, dat[4], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=10, return_geo=False)
                sdata['data_650'] = grid_by_lonlat(lon0, lat0, dat[0], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=10, return_geo=False)
                sdata['data_860'] = grid_by_lonlat(lon0, lat0, dat[1], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=10, return_geo=False)
                sdata['data_470'] = grid_by_lonlat(lon0, lat0, dat[2], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=10, return_geo=False)
                sdata['data_555'] = grid_by_lonlat(lon0, lat0, dat[3], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=10, return_geo=False)

                # satellite name
                if os.path.basename(ref_file)[:3] == "MOD":
                    satellite  = "Terra"
                    group_name = "T" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "MYD":
                    satellite  = "Aqua"
                    group_name = "A" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "VNP":
                    satellite  = "Suomi-NPP"
                    group_name = "S" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "VJ1":
                    satellite  = "NOAA-20/JPSS-1"
                    group_name = "J" + acq_dt[1:]
                elif os.path.basename(ref_file)[:3] == "VJ2":
                    satellite  = "NOAA-21/JPSS-2"
                    group_name = "N" + acq_dt[1:]
                else:
                    satellite  = "Unknown"
                    group_name = "Z" + acq_dt[1:]

                satellite = np.array(satellite.encode("utf-8"), dtype=utf8_type)

                # print(acq_dt, group_name, satellite)
                # print("-------------------")
                # save data
                g0 = f0.create_group(group_name)
                _ = g0.create_dataset('ref_2130', data=sdata['data_2130'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_1640', data=sdata['data_1640'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_860',  data=sdata['data_860'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_650',  data=sdata['data_650'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_555',  data=sdata['data_555'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('ref_470',  data=sdata['data_470'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('sza_2d',   data=sdata['sza_2d'].T, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('lon_1d',   data=lon_1d, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('lat_1d',   data=lat_1d, compression='gzip', compression_opts=9)
                _ = g0.create_dataset('metadata', data=metadata)
                _ = g0.create_dataset('satellite',data=satellite)

                valid_count += 1
            except Exception as err:
                print(err)
                continue

    if valid_count == 0:
        print("Could not extract data. Deleting the HDF5 file...\n")
        os.remove(os.path.join(outdir, 'ref_geo-data.h5'))
        if len(os.listdir(outdir)) == 0:
            print("Also deleting the output directory {}...".format(outdir))
            os.rmdir(outdir)
        return 0
    else:
        print("{}: Successfully extracted data from {} of {} overpasses".format(os.path.join(outdir, 'ref_geo-data.h5'), valid_count, n_obs))
        return 1

########################################################################################################################

# def save_to_file_modis_viirs_ref_geo_cld_opt_msk_ctt(fdir, outdir, lon_1d, lat_1d, metadata, o_thresh=75):

#     fref, f03, fcld_l2, fcldmsk_l2 = get_modis_viirs_red_geo_cld_opt_mskfnames(fdir)
#     if (len(fref) == 0) or (len(f03) == 0) or (len(fcld_l2) == 0) or (len(fcldmsk_l2) == 0):
#         raise FileNotFoundError("\nCould not find the valid files\n")

#     #print(len(fref), len(f03), len(fcld_l2), len(fcldmsk_l2))
#     # if len(fref) == len(f03) == len(fcld_l2) == len(fcldmsk_l2):
#     #     pass
#     # else:
#     #     print(len(fref), len(f03), len(fcld_l2), len(fcldmsk_l2))

#     # return 1
#     n_obs = len(f03)
#     # acquisition date times
#     # acq_dts = sorted([os.path.basename(i)[9:22] for i in fref])[-15:]
#     print("Found {} overpasses".format(n_obs))
#     extent = [lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()]
#     ext_area = get_swath_area(lon_1d, lat_1d)

#     valid_count = 0 # just for reporting

#     with h5py.File(os.path.join(outdir, 'ref_geo_cld-data.h5'), 'w') as f0:
#         # for acq_dt in acq_dts:
#         for i in range(n_obs):
#             sdata = {}
#             # geometries/geolocation file
#             # geo_file = list(filter(lambda x: acq_dt in x, f03))[0] # match acquisition to get geometry file
#             geo_file    = f03[i]
#             ref_file    = fref[i]
#             cld_file    = fcld_l2[i]
#             cldmsk_file = fcldmsk_l2[i]

#             acq_dt = os.path.basename(geo_file).split('.')[1] + '.' + os.path.basename(geo_file).split('.')[2]

#             try:
#                 if os.path.basename(geo_file).upper().startswith('M'): # modis
#                     f_geo = modis_03(fnames=[geo_file], extent=extent)
#                 elif os.path.basename(geo_file).upper().startswith('V'): # viirs
#                     f_geo = viirs_03(fnames=[geo_file], extent=extent)
#                 else:
#                     raise NotImplementedError("\nOnly VIIRS and MODIS products are supported\n")
#             except HDF4Error:
#                 print("PyHDF error with {}...skipping...\n".format(geo_file))
#                 continue
#             except FileNotFoundError:
#                 print("netcdf error with {}...skipping...\n".format(geo_file))
#                 continue

#             lon0 = f_geo.data['lon']['data']
#             lat0 = f_geo.data['lat']['data']
#             if len(lon0) == 0 or len(lat0) == 0:
#                 # print('Lat/lon not valid')
#                 continue

#             mod_area = get_swath_area(lon0, lat0)
#             overlap  = mod_area * 100. / ext_area
#             if overlap < o_thresh:
#                 # print('Skipping {}, overlap {:0.1f} is less than {}%'.format(acq_dt, overlap, o_thresh))
#                 continue

#             vza0 = f_geo.data['vza']['data']
#             sza0 = f_geo.data['sza']['data']
#             vaa0 = f_geo.data['vaa']['data']
#             saa0 = f_geo.data['saa']['data']
#             sdata['sza_2d'] = grid_by_lonlat(lon0, lat0, sza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
#             sdata['vza_2d'] = grid_by_lonlat(lon0, lat0, vza0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
#             sdata['saa_2d'] = grid_by_lonlat(lon0, lat0, saa0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)
#             sdata['vaa_2d'] = grid_by_lonlat(lon0, lat0, vaa0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='linear', return_geo=False)

#             # reflectance file
#             # ref_file = list(filter(lambda x: acq_dt in x, fref))[0]

#             try:
#                 if os.path.basename(ref_file).upper().startswith('M'): # modis
#                     f_vis = modis_l1b(fnames=[ref_file], extent=extent, f03=f_geo, bands=[1, 2, 3, 4, 6, 7])
#                 elif os.path.basename(ref_file).upper().startswith('V'): # viirs
#                     f_vis = viirs_l1b(fnames=[ref_file], extent=extent, f03=f_geo, bands=["M05", "M07", "M02", "M04", "M10", "M11"])
#                 else:
#                     print("\nOnly VIIRS and MODIS products are supported. Instead, file was: {}\n".format(ref_file))
#                     continue

#                 dat = f_vis.data['ref']['data']

#                 # Band 7 (2130nm)
#                 sdata['data_2130'] = grid_by_lonlat(lon0, lat0, dat[5], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
#                 # [650.0, 860.0, 470.0, 555.0, 1240.0, 1640.0, 2130.0]
#                 # Bands 6 (1640nm) and 1 (650nm) and 2 (860nm) and 3 (470nm)
#                 sdata['data_1640'] = grid_by_lonlat(lon0, lat0, dat[4], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
#                 sdata['data_650'] = grid_by_lonlat(lon0, lat0, dat[0], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
#                 sdata['data_860'] = grid_by_lonlat(lon0, lat0, dat[1], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
#                 sdata['data_470'] = grid_by_lonlat(lon0, lat0, dat[2], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
#                 sdata['data_555'] = grid_by_lonlat(lon0, lat0, dat[3], lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)


#                 # cloud product
#                 # cld_file = list(filter(lambda x: acq_dt in x, fcld_l2))[0] # match aquisition to get cloud_l2 file
#                 if os.path.basename(cld_file).upper().startswith(('MOD06_L2', 'MYD06_L2')): # modis
#                     f_cld = modis_l2(fnames=[cld_file], f03=f_geo, extent=extent)
#                 elif os.path.basename(cld_file).upper().startswith('CLDPROP'): # viirs
#                     f_cld = viirs_cldprop_l2(fnames=[cld_file], f03=f_geo, extent=extent, maskvars=False)
#                 else:
#                     print("\nOnly VIIRS and MODIS products are supported. Instead, file was: {}\n".format(cld_file))
#                     continue

#                 cot0 = f_cld.data['cot']['data']
#                 cer0 = f_cld.data['cer']['data']
#                 cwp0 = f_cld.data['cwp']['data']
#                 cot_1621 = f_cld.data['cot_1621']['data']
#                 cer_1621 = f_cld.data['cer_1621']['data']
#                 cwp_1621 = f_cld.data['cwp_1621']['data']
#                 ctp0 = np.float64(f_cld.data['ctp']['data'])
#                 lon0 = f_cld.data['lon']['data']
#                 lat0 = f_cld.data['lat']['data']
#                 ctp_temp = grid_by_lonlat(lon0, lat0, ctp0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
#                 # sdata['ctp'] = ctp_temp.astype('int')
#                 sdata['ctp'] = ctp_temp
#                 sdata['cot_2d'] = grid_by_lonlat(lon0, lat0, cot0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
#                 sdata['cer_2d'] = grid_by_lonlat(lon0, lat0, cer0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
#                 sdata['cwp_2d'] = grid_by_lonlat(lon0, lat0, cwp0, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
#                 sdata['cot_1621'] = grid_by_lonlat(lon0, lat0, cot_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
#                 sdata['cer_1621'] = grid_by_lonlat(lon0, lat0, cer_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
#                 sdata['cwp_1621'] = grid_by_lonlat(lon0, lat0, cwp_1621, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)

#                 # cloud mask
#                 if os.path.basename(cldmsk_file).upper().startswith(('MOD35_L2', 'MYD35_L2')): # modis
#                     f_cldmsk = modis_35_l2(fnames=[cldmsk_file], f03=f_geo, extent=extent)
#                     conf_qa = np.float64(f_cldmsk.data['confidence_qa']['data'])
#                 # read in cldprop file again for viirs
#                 elif os.path.basename(cldmsk_file).upper().startswith('CLDPROP'): # viirs
#                     f_cldmsk = viirs_cldprop_l2(fnames=[cldmsk_file], f03=f_geo, extent=extent, maskvars=True)
#                     conf_qa = np.float64(f_cldmsk.data['ret_1621_conf_qa']['data'])
#                 else:
#                     print("\nOnly VIIRS and MODIS products are supported. Instead, file was: {}\n".format(cldmsk_file))
#                     continue


#                 cm_flag = np.float64(f_cldmsk.data['cloud_mask_flag']['data'])
#                 fov_qa = np.float64(f_cldmsk.data['fov_qa_cat']['data'])
#                 sunglint = np.float64(f_cldmsk.data['sunglint_flag']['data'])
#                 lon0 = f_cldmsk.data['lon']['data']
#                 lat0 = f_cldmsk.data['lat']['data']

#                 conf_qa_temp = grid_by_lonlat(lon0, lat0, conf_qa, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
#                 cm_flag_temp = grid_by_lonlat(lon0, lat0, cm_flag, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
#                 fov_qa_temp = grid_by_lonlat(lon0, lat0, fov_qa, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)
#                 sunglint_temp = grid_by_lonlat(lon0, lat0, sunglint, lon_1d=lon_1d, lat_1d=lat_1d, fill_value=np.nan, method='nearest', Ngrid_limit=1, return_geo=False)

#                 sdata['conf_qa'] = conf_qa_temp
#                 sdata['cm_flag'] = cm_flag_temp
#                 sdata['fov_qa'] = fov_qa_temp
#                 sdata['sunglint'] = sunglint_temp

#                 # sdata['conf_qa'] = conf_qa_temp.astype('int')
#                 # sdata['cm_flag'] = cm_flag.astype('int')
#                 # sdata['fov_qa'] = fov_qa_temp.astype('int')
#                 # sdata['sunglint'] = sunglint_temp.astype('int')

#                 # f_cldmsk.data['snow_ice_flag']['data']
#                 # f_cldmsk.data['land_water_cat']['data']

#                 # satellite name
#                 if os.path.basename(ref_file)[:3] == "MOD":
#                     satellite  = "Terra"
#                     group_name = "T" + acq_dt[1:]
#                 elif os.path.basename(ref_file)[:3] == "MYD":
#                     satellite  = "Aqua"
#                     group_name = "A" + acq_dt[1:]
#                 elif os.path.basename(ref_file)[:3] == "VNP":
#                     satellite  = "Suomi-NPP"
#                     group_name = "S" + acq_dt[1:]
#                 elif os.path.basename(ref_file)[:3] == "VJ1":
#                     satellite  = "NOAA-20 (JPSS-1)"
#                     group_name = "J" + acq_dt[1:]
#                 else:
#                     satellite  = "Unknown"
#                     group_name = "Z" + acq_dt[1:]

#                 satellite = np.array(satellite.encode("utf-8"), dtype=utf8_type)

#                 # print(acq_dt, group_name, satellite)
#                 # print("-------------------")
#                 # save data
#                 g0 = f0.create_group(group_name)
#                 _ = g0.create_dataset('ref_2130', data=sdata['data_2130'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('ref_1640', data=sdata['data_1640'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('ref_860',  data=sdata['data_860'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('ref_650',  data=sdata['data_650'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('ref_555',  data=sdata['data_555'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('ref_470',  data=sdata['data_470'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('sza_2d',   data=sdata['sza_2d'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('vza_2d',   data=sdata['vza_2d'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('saa_2d',   data=sdata['saa_2d'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('vaa_2d',   data=sdata['vaa_2d'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('ctp',      data=sdata['ctp'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('cot_2d',   data=sdata['cot_2d'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('cer_2d',   data=sdata['cer_2d'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('cwp_2d',   data=sdata['cwp_2d'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('cot_1621', data=sdata['cot_1621'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('cer_1621', data=sdata['cer_1621'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('cwp_1621', data=sdata['cwp_1621'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('conf_qa',  data=sdata['conf_qa'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('cm_flag',  data=sdata['cm_flag'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('fov_qa',   data=sdata['fov_qa'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('sunglint', data=sdata['sunglint'].T, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('lon_1d',   data=lon_1d, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('lat_1d',   data=lat_1d, compression='gzip', compression_opts=9)
#                 _ = g0.create_dataset('metadata', data=metadata)
#                 _ = g0.create_dataset('satellite',data=satellite)

#                 valid_count += 1
#             except Exception as err:
#                 print(err)
#                 continue

#     print("Successfully extracted data from {} of {} overpasses".format(valid_count, n_obs))
#     return 1
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
    parser.add_argument("--fdir",  default=None,  type=str, help="Path to directory containing raw MODIS HDF files")
    parser.add_argument("--outdir", type=str, default=None, help="Path to directory where images will be saved")
    parser.add_argument("--resolution", type=int, default=1000, help="Resolution to be gridded onto in meters")
    parser.add_argument("--width", type=int, default=480, help="Width of the image")
    parser.add_argument("--height", type=int, default=480, help="Height of the image")
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
    # parser.add_argument("--all", action='store_true', help="Enable --all to create training data for all locations and projects")
    # parser.add_argument("--use_metadates", action='store_true', help="Enable --use_metadates to use start and end dates written by the satellite download program to a metadata file")
    parser.add_argument("--nrt", action='store_true', help="Enable --nrt to process VIIRS L1b and 03 products, and MODIS L1b, 03, and clds")
    parser.add_argument("--viz", action='store_true')
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
                print("Message [create_training_data]: Start time and end times were too far apart...limiting to 3 hour gap.")

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
    print("Message [create_training_data]: Start datetime: {}, End datetime: {}".format(start_dt_hhmm_str, end_dt_hhmm_str))
    print("=====================================================================")

    if (args.ndir_recent is not None) and (len(subdirs) >= args.ndir_recent):
        subdirs = subdirs[-args.ndir_recent:]

    print("Message [create_training_data]: {} sub-directories will be analyzed".format(len(subdirs)))

    for fdir in tqdm(subdirs):
        dt = os.path.basename(fdir)
        print("Currently analyzing:", dt) # date
        year, month, date, hour, minute, sec, extent = get_metadata(fdir)

        outdir = args.outdir

        metadata = np.array([year, month, date, hour, minute, sec, extent[0], extent[1], extent[2], extent[3]])
        lon_1d, lat_1d = calc_lonlat_arr(west=extent[0], south=extent[2],
                                        width=args.width, height=args.height,
                                        resolution=args.resolution)

        # ret = save_to_file_modis_viirs_ref_geo_cld_opt_msk(fdir, outdir, lon_1d, lat_1d, metadata, o_thresh=75)
        # ret = save_to_file_modis_viirs_ref_geo(fdir, outdir, lon_1d, lat_1d, metadata, o_thresh=40)
        ret = 0
        if args.nrt:
            ret += save_to_file_modis_viirs_ref_geo(fdir, outdir, lon_1d, lat_1d, metadata, start_dt=start_dt_hhmm, end_dt=end_dt_hhmm, o_thresh=40)
            ret += save_to_file_modis_only_geo_cld_opt(fdir, outdir, lon_1d, lat_1d, metadata, start_dt=start_dt_hhmm, end_dt=end_dt_hhmm, o_thresh=40)

        else:
            ret += save_to_file_modis_viirs_ref_geo_cld_opt_msk(fdir, outdir, lon_1d, lat_1d, metadata, o_thresh=40)
        # if args.viz:
        #     ret = save_to_file_modis_viirs_viz(fdir, outdir, project_name, lon_1d, lat_1d, o_thresh=75)
        # else:
        #     ret = save_to_file_modis_viirs(fdir, outdir, project_name, lon_1d, lat_1d, o_thresh=75)
        if ret == 0:
            print("{} failed".format(fdir))
            manual_list.append(fdir)
            continue

        # else:
        #     print("Provided date ranges {} to {} are outside the date ranges of the downloaded satellites.".format(start_dt_hhmm.strftime("%Y-%m-%d: %H%M"), end_dt_hhmm.strftime("%Y-%m-%d: %H%M")))
        #     continue


    print("Missed: ", manual_list)
    print("Finished!")

    END_TIME = datetime.datetime.now()
    print('Time taken to execute:', END_TIME - START_TIME)
