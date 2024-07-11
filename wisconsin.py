"""
This file is for grabbing files off the University of Wisconsin (SSEC)
near real time server.

Author
------
Vikas Nataraja


Data Credit
-----------
Steve Dutcher, Robert Holz, SSEC Team
"""

import os
import json
import datetime
import subprocess
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from argparse import ArgumentParser, RawTextHelpFormatter
import matplotlib

import util.constants
import util.wisc_util


class WisconsinDownload:

    def __init__(self,
                 start_time,
                 end_time,
                 rlon_2d,
                 rlat_2d,
                 verbose=True,
                 iou=0,
                 save_dir='tmp-data/',
                 file_check=True,
                 overwrite=False):

        self.iou        = iou      # overlap percentage
        self.roi_lon2d  = rlon_2d  # gridded lon of region of interest
        self.roi_lat2d  = rlat_2d  # gridded lon of region of interest
        self.save_dir   = save_dir # directory to which files will be downloaded
        self.verbose    = verbose  # verbose flag
        self.start_time = start_time # datetime object
        self.end_time   = end_time # datetime object
        self.file_check = file_check # set to True to run a basic file check after download
        self.overwrite  = overwrite  # if False AND file exists AND it works, download will be skipped;
        self.df         = []         # pandas dataframe


    def format_start_end_time(self):
        """Format time for asipcli call"""
        start_time_str = self.start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_time_str = self.end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        return start_time_str, end_time_str


    def generate_asipscli_granule_metadata_command(self, satellite, instrument):
        """
        Equivalent to grabbing the geoMeta file on a NASA DAAC

        Example: asipscli granule --satellite snpp --sensor viirs -s 2024-07-07T14:00:00Z -e 2024-07-07T17:00:00Z --json-pretty
        """
        satellite  = satellite.replace('-', '').replace('/', '').lower()
        instrument = instrument.lower()
        start_time_str, end_time_str = self.format_start_end_time()

        cmd = ["./asipscli", "granule", "--satellite", "{}".format(satellite), "--sensor", "{}".format(instrument), "--start", "{}".format(start_time_str), "--end", "{}".format(end_time_str), "--json"]
        return cmd


    def generate_asipscli_files_metadata_command(self, product):
        """
        Equivalent to grabbing the geoMeta file on a NASA DAAC

        Example: asipscli files -p VJ102MOD-nrt -S noaa20 -s 2024-07-07T14:00:00Z -e 2024-07-07T17:00:00Z  --version 3.1.0"
        """
        pname    = util.wisc_util.wisc_info[product]['cli_fname']
        pversion = util.wisc_util.wisc_info[product]['version']
        start_time_str, end_time_str = self.format_start_end_time()

        cmd = ["./asipscli", "files", "--products", "{}".format(pname), "--start", "{}".format(start_time_str), "--end", "{}".format(end_time_str), "--version", "{}".format(pversion), "--json"]
        return cmd


    def get_granule_metadata(self, command):
        """
        Run asipscli granule command to get granule metadata.
        Does not download an actual file.

        Args:
            command: list, a list of strings containing exec programs and args

        Returns:
            dict, either empty or containing the granule(s) info
        """
        try:
            ret = subprocess.run(command, capture_output=True, check=True, text=True)

        except Exception as err:
            print("Message [get_granule_metadata]: ", err)
            return {}

        out = ret.stdout
        jout = json.loads(out) # json output formatted as dict
        if (jout is None):
            return {}

        if (jout['status'] != 'success'):
            print("Message [get_granule_metadata]: Failed to run command: {}".format(command))
            return {}

        return jout


    def get_file_metadata(self, command):
        """
        Run asipscli file command to get file metadata.
        Does not download an actual file.

        Args:
            command: list, a list of strings containing exec programs and args

        Returns:
            dict, either empty or containing the granule(s) info
        """
        try:
            ret = subprocess.run(command, capture_output=True, check=True, text=True)

        except Exception as err:
            print("Message [get_file_metadata]: ", err)
            return {}

        out = ret.stdout
        jout = json.loads(out) # json output formatted as dict
        if (jout is None):
            return {}

        if (jout['status'] != 'success'):
            print("Message [get_file_metadata]: Failed to run command: {}".format(command))
            return {}

        return jout


    def gtime2acqdt_df(self, gstart_time_str):
        """
        Generate acquisition datetime
        Example: 'A2024184.2235'

        """
        gstart_dt = datetime.datetime.strptime(gstart_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        return gstart_dt.strftime("A%Y%j.%H%M")


    def format_asipscli_granule_dtypes_df(self, df):
        """
        Format and change dtype of dataframe columns
        """

        # these are read incorrectly by pandas so use int and then to str later
        int_cols  = ['id', 'orbit_id', 'source_file_id', 'orbit_number']
        int_dtype = ['int'] * len(int_cols)
        int_dtype_dict = dict(zip(int_cols, int_dtype))

        float_cols   = ['solar_zen_min', 'solar_zen_max', 'orbit_crossing_lon']
        float_dtype  = ['float16'] * len(float_cols)
        float_dtype_dict = dict(zip(float_cols, float_dtype))

        str_cols   = ['begin_time', 'end_time', 'satellite', 'sensor', 'daynight', 'direction', 'source_type', 'orbit_crossing_time']
        str_dtype  = ['str'] * len(str_cols)
        str_dtype_dict = dict(zip(str_cols, str_dtype))

        df = df.astype({**int_dtype_dict, **float_dtype_dict, **str_dtype_dict})

        # but actually the int columns need to be str
        int2str_dtype = ['str'] * len(int_cols)
        int2str_dtype_dict = dict(zip(int_cols, int2str_dtype))
        df = df.astype(int2str_dtype_dict)
        return df


    def format_asipscli_files_dtypes_df(self, df):
        """
        Format and change dtype of dataframe columns
        """
        int_cols  = ['size']
        int_dtype = ['int'] * len(int_cols)
        int_dtype_dict = dict(zip(int_cols, int_dtype))

        float_cols   = ['solar_zen_min', 'solar_zen_max']
        float_dtype  = ['float16'] * len(float_cols)
        float_dtype_dict = dict(zip(float_cols, float_dtype))

        str_cols   = ['name', 'product_name', 'version', 'begin_time', 'end_time', 'satellite', 'sensor', 'daynight', 'direction']
        str_dtype  = ['str'] * len(str_cols)
        str_dtype_dict = dict(zip(str_cols, str_dtype))

        df = df.astype({**int_dtype_dict, **float_dtype_dict, **str_dtype_dict})
        return df


    def calculate_overlap_df(self, bounds):
        """
        Calculate overlap between satellite `bounds` and
        region of interest defined by gridded `rlon_2d`, `rlat_2d`
        """
        coords = np.squeeze(bounds['coordinates'])

        slons = coords[:, 0] # satellite longitudes of bbox
        slats = coords[:, 1] # satellite latitudes of bbox
        clon = np.mean(slons[:-1]) # last one is a repeat of the first
        clat = np.mean(slats[:-1])

        proj_data = ccrs.PlateCarree()
        proj_plot = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)

        xy  = proj_plot.transform_points(proj_data, slons, slats)[:, [0, 1]]

        sat_granule  = matplotlib.path.Path(xy, closed=True)

        # check if the overpass/granule overlaps with region of interest
        xy_region  = proj_plot.transform_points(proj_data, self.roi_lon2d, self.roi_lat2d)
        rlons      = xy_region[:, :, 0] # region longitudes of bbox
        rlats      = xy_region[:, :, 1] # region latitudes of bbox
        rpoints    = np.vstack((rlons.flatten(), rlats.flatten())).T
        points_in  = sat_granule.contains_points(rpoints)

        Npoints_in  = points_in.sum()
        Npoints_tot = points_in.size

        percent_in  = float(Npoints_in) * 100.0 / float(Npoints_tot)
        return percent_in


    def run_curl_download_command(self, outdir):
        """
        Generate command and download ASIPSCLI files to `outdir`
        """
        success_count = 0

        for i in range(len(self.df)):

            fname_local = os.path.join(outdir, self.df.name.iloc[i]) # full filepath incl. directory

            if os.path.isfile(fname_local):
                if self.overwrite:
                    print('Message [run_curl_download_command]: Deleting existing file: {}'.format(fname_local))
                    os.remove(fname_local)

                else:
                    fs_ = self.check_file_status(fname_local)
                    if fs_ == 0:
                        print('Message [run_curl_download_command]: File {} exists but file check failed. File will be deleted.'.format(fname_local))
                        os.remove(fname_local)

                    else:
                        print('Message [run_curl_download_command]: File {} exists and file check was successful. File will not be re-downloaded.'.format(fname_local))
                        continue

            split_fname_local = os.path.basename(fname_local).split('.')
            acq_dt = split_fname_local[0] + '.' + split_fname_local[1] + '.' + split_fname_local[2]
            if acq_dt in util.wisc_util.get_pacq_dts(outdir):
                print("Message [run_curl_download_command]: File {} already exists based on acq_dt (maybe another source). Will not re-download this file.".format(fname_local))
                continue

            url = self.df['urls'].iloc[i]['public']
            cmd = ["curl", "{}".format(url), "-sSf", "-H", "X-API-Token: {}".format(os.environ['ASIPSCLI_TOKEN']), "-o", "{}".format(fname_local)]
            if self.verbose:
                print('Message [run_curl_download_command]: Running command:\n', ' '.join(cmd), '\n')
            try:
                ret = subprocess.run(cmd, capture_output=True, check=True, text=True)

                if self.file_check:
                    if ret.returncode == 0:
                        fs = self.check_file_status(fname_local)
                        if fs == 0:
                            print('Message [run_curl_download_command]: Download of {} was unsuccessful. File will be deleted.'.format(fname_local))
                            if os.path.isfile(fname_local):
                                os.remove(fname_local)
                        else:
                            print('Message [run_curl_download_command]: File check complete. Download of {} successful!'.format(fname_local))
                            success_count += 1

                    else:
                        print('Message [run_curl_download_command]: Something went wrong. `curl` returned {}'.format(ret.returncode))

                else:
                    if self.verbose:
                        print('Message [run_curl_download_command]: `curl` ran but did not verify file status.')
                    success_count += 1

            except Exception as err:
                print('Message [run_curl_download_command]: Following error occurred when running `curl`:', err)

        return success_count


    def read_granule_json(self, json_output):

        if self.verbose:
            print("Message [read_granule_json]: Found {} granules between provided start and end times".format(len(json_output['data'])))

        df = pd.DataFrame(json_output['data'])
        # df = df.replace(to_replace='None', value=np.nan).dropna().reset_index(drop=True) # clean it up
        df = df[(df.solar_zen_min < util.constants.SZA_LIMIT) & (df.daynight == 'day') & ((df.source_type == 'nrt') | (df.source_type == 'production'))]
        df = df.reset_index(drop=True)
        df = self.format_asipscli_granule_dtypes_df(df) # format the dtypes of easier use
        df['overlap'] = df['bounds'].apply(self.calculate_overlap_df)
        df = df[df['overlap'] > self.iou].reset_index(drop=True) # get granules only those that are > iou
        if self.verbose:
            print("Message [read_file_json]: Found {} granules between provided start and end times after filtering by overlap, ".format(len(df)))

        return df


    def read_file_json(self, json_output):

        if self.verbose:
            print("Message [read_file_json]: Found {} granules between provided start and end times".format(len(json_output['data'])))

        if len(json_output['data']) == 0: # no applicable overpasses found
            if self.verbose:
                print("Message [read_file_json]: Could not find any applicable overpasses after filtering by SZA ad DayNight mode")
            return []

        df = pd.DataFrame(json_output['data'])
        use_columns = ['name', 'size', 'product_name', 'version',
                       'begin_time', 'end_time', 'satellite', 'sensor', 'daynight',
                       'direction', 'solar_zen_min', 'solar_zen_max', 'bounds', 'urls']
        df = df[use_columns]

        df = df[(df.solar_zen_min < util.constants.SZA_LIMIT) & (df.daynight == 'day')].reset_index(drop=True)

        if len(df) == 0: # no applicable overpasses found
            if self.verbose:
                print("Message [read_file_json]: Could not find any applicable overpasses after filtering by SZA ad DayNight mode")
            return df

        df = self.format_asipscli_files_dtypes_df(df) # format the dtypes of easier use

        df['overlap'] = df['bounds'].apply(self.calculate_overlap_df)
        df = df[df['overlap'] > self.iou].reset_index(drop=True) # get granules only those that are > iou

        if len(df) == 0: # no applicable overpasses found
            if self.verbose:
                print("Message [read_file_json]: Could not find any applicable overpasses after filtering by overlap")
            return df

        df['acq_dt'] = df['begin_time'].apply(self.gtime2acqdt_df)

        if self.verbose:
            print("Message [read_file_json]: Found {} granules between provided start and end times after filtering by overlap, ".format(len(df)))

        return df


    def download_asipscli_file(self, product, outdir):
        """
        Downloads ASIPSCLI files for a given product.

        Args:
            product (str): The name of the product to download. Must be one of the keys in `util.wisc_util.wisc_info`.

        Returns:
            int: The number of files downloaded.
        """

        # format and check product name
        product = product.upper()
        if product not in list(util.wisc_util.wisc_info.keys()):
            print("Error [download_asipscli_file]: Product must be one of: {}".format(list(util.wisc_util.wisc_info.keys())))
            return 0

        # generate and run command to get metadata
        cmd = self.generate_asipscli_files_metadata_command(product)
        json_out = self.get_file_metadata(cmd)

        # parse the output json metadata
        df = self.read_file_json(json_out) # make df available to class object

        # concatenate or save the dataframe, otherwise return empty
        if len(df) == 0 and self.verbose:
            print("Message [download_asipscli_file]: No applicable overpasses found.")
            return 0

        else:
            current_count = len(self.df)
            if len(self.df) == 0:
                self.df = df

            else:
                self.df = pd.concat([self.df, df], axis=0)
                self.df = self.df.reset_index(drop=True)

            success_count = self.run_curl_download_command(outdir=outdir)
            if success_count == 0:
                print("Message [download_asipscli_file]: One or more of the downloads did not complete, either due to a file already existing or another issue. Enable verbose option to debug.\n")
                return 0

            print("Message [download_asipscli_file]: Download successful.")
            return len(self.df) - current_count


    def check_file_status(self, fname_local, data_format=None):
        """
        Check if a file has been successfully downloaded.

        This function currently supports checking for HDF, NetCDF, and HDF5 files.
        Params:
            fname_local (str): The local filename to check.
            data_format (str): The format of the file. If None, the format is inferred from the file extension.
        Returns:
            Returns 1 if the file has been successfully downloaded and 0 otherwise.

        Adapted from EaR3T: https://github.com/hong-chen/er3t
        """


        if data_format is None:
            data_format = os.path.basename(fname_local).split('.')[-1].lower()

        if data_format in ['hdf', 'hdf4', 'h4']:

            try:
                from pyhdf.SD import SD, SDC
                f = SD(fname_local, SDC.READ)
                f.end()
                return 1

            except Exception as error:
                print("Message [check_file_status]", error)
                return 0

        elif data_format in ['nc', 'nc4', 'netcdf', 'netcdf4']:
            try:
                import netCDF4 as nc
                f = nc.Dataset(fname_local, 'r')
                f.close()
                return 1

            except Exception as error:
                print("Message [check_file_status]", error)
                return 0


    def save_df_to_file(self, fpath):
        extension = fpath.split('.')[-1]
        if extension != 'csv':
            fpath = fpath + '.csv'
        self.df.to_csv(fpath, index=False)

# ==============================================================================================================================

def run(args):
    llons, llats = util.wisc_util.region_parser(args.extent)
    lon_2d, lat_2d = np.meshgrid(llons, llats, indexing='ij')

    start_dt, end_dt = util.wisc_util.time_handler(args.start_date, args.end_date, args.latest_nhours)

    date_list = [start_dt.date() + datetime.timedelta(days=x) for x in range((end_dt.date() - start_dt.date()).days + 1)]

    product_counter = {} # to count number of products downloaded

    for date_x in date_list: # download products date by date
        fdir_out_dt = os.path.join(args.fdir, date_x.strftime('%Y-%m-%d')) # save files in dirs with dates specified
        if not os.path.exists(fdir_out_dt):
            os.makedirs(fdir_out_dt)

        # ==================================================================================================================
        # Save metadata but if file exists, delete it
        if os.path.isfile(os.path.join(fdir_out_dt, 'metadata.txt')):
            print('Message [sdown]: metadata.txt file was removed from {}'.format(os.path.join(fdir_out_dt, 'metadata.txt')))
            os.remove(os.path.join(fdir_out_dt, 'metadata.txt'))


        with open(os.path.join(fdir_out_dt, 'metadata.txt'), "w") as f:
            f.write('Date: {}\n'.format(date_x.strftime('%Y-%m-%d')))
            f.write('Extent: {}\n'.format([np.nanmin(llons), np.nanmax(llons), np.nanmin(llats), np.nanmax(llats)]))
        # ==================================================================================================================

        wisc =    WisconsinDownload(start_time=start_dt,
                                    end_time=end_dt,
                                    verbose=True, # force verbose for now
                                    iou=args.iou,
                                    rlon_2d=lon_2d,
                                    rlat_2d=lat_2d,
                                    save_dir=args.fdir,
                                    file_check=True,
                                    overwrite=False)

        for product in args.products:
            product_counter[product] = wisc.download_asipscli_file(product=product, outdir=fdir_out_dt)

    return product_counter


if __name__ == '__main__':


    parser = ArgumentParser(prog='wisconsin', formatter_class=RawTextHelpFormatter,
                            description="Tool to Download University of Wisconsin SSEC data")

    parser.add_argument('--fdir', type=str, metavar='', default='sat-data/',
                        help='Directory where the files will be downloaded\n'\
                        'By default, files will be downloaded to \'sat-data/\'\n \n')
    parser.add_argument('--start_date', type=str, metavar='', default=None,
                        help='The start date of the range of dates for which you would like to download data. '\
                        'Use yyyymmdd or yyyymmddhhmm format.\n'\
                        'Example: --start_date 20210404\n \n')
    parser.add_argument('--end_date', type=str, metavar='', default=None,
                        help='The end date of the range of dates for which you would like to download data. '\
                        'Use yyyymmdd or yyyymmddhhmm format.\n'\
                        'Example: --end_date 20210414\n \n')
    parser.add_argument('--iou', type=int, metavar='', default=0,
                        help='Percentage of points within the region of interest that must overlap with the satellite granule. \n'\
                        'If the overlap <= iou, then the granule file will not be downloaded.\n'\
                        'Example:  --iou 60\n \n')
    parser.add_argument('--latest_nhours', type=int, metavar='', default=None,
                        help='Time range in hours to use to get near-real time data. Only used if --nrt is enabled and --start_date and --end_date not supplied. \n'\
                        'Example: --latest_nhours 6 will download the most recent 6 hours of NRT data\n \n')
    parser.add_argument('--extent', nargs='+', type=float, metavar='',
                        help='Extent of region of interest \nlon1 lon2 lat1 lat2 in West East South North format.\n'\
                        'Example:  --extent -10 -5 25 30\n \n')
    parser.add_argument('--products', type=str, nargs='+', metavar='',
                        help='Short prefix (case insensitive) for the product name, VIIRS only.\n'\
                        'Example:  --products VNP02MOD VNP03MOD\n'
                        )

    args = parser.parse_args()

    product_counter = run(args)

    if len(product_counter) == 0:
        print("Message [wisconsin]: One or more of the downloads did not complete, either due to a file already existing or another issue. Enable verbose option to debug.\n")
    else:
        print('Script finished running. Number of downloads:')
        for key, value in product_counter.items():
            print(key, ':', value)
