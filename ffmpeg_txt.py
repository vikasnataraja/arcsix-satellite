import os
import sys
import datetime
from argparse import ArgumentParser
import numpy as np


def fname2dt(fpng):
    dt_str, satellite = os.path.splitext(os.path.basename(fpng))[0].split('_')
    dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d-%H%MZ')
    return dt, satellite

def dt2fname(dt, satellite):
    dt_str = dt.strftime('%Y-%m-%d-%H%M') + 'Z_{}.png'.format(satellite)
    return dt_str


no_video_dirs = ['ice_path', 'water_path', 'optical_thickness', 'cloud_phase', 'cloud_top_height_temperature']
if __name__ == "__main__":

    START_TIME = datetime.datetime.now()
    parser = ArgumentParser(prog='ffmpeg_txt')
    parser.add_argument('--fdir', type=str, metavar='',
                        help='Top-level source directory\n')
    parser.add_argument('--frame_rate', type=float, metavar='', default=0.5,
                        help='Reciprocal of frame rate i.e., --frame rate=0.5 is 2 frames per second.\n')
    args = parser.parse_args()

    # fdir = sys.argv[1]
    # frame_rate = 0.5

    subs = sorted([f for f in os.listdir(args.fdir) if os.path.isdir(os.path.join(args.fdir, f))])

    for sub in subs:
        if sub in no_video_dirs:
            print("Message [ffmpeg_txt]: Skipping {}...".format(sub))
            continue
        outpath = os.path.join(args.fdir, sub, 'create_video_metadata.txt')
        fpngs = [png for png in os.listdir(os.path.join(args.fdir, sub)) if png.endswith('.png')]

        fpngs = sorted(fpngs, key=lambda x: x.split('_')[0])

        dt_pngs, sat_names = [], []
        for png in fpngs:
            dt, sat = fname2dt(png)
            dt_pngs.append(dt)
            sat_names.append(sat)

        if len(dt_pngs) == 0:
            print("No png files found for {}".format(os.path.join(args.fdir, sub)))
            continue

        dt_pngs  = np.array(dt_pngs)
        sat_names = np.array(sat_names)


        upper_limit = dt_pngs[-1]
        lower_limit = upper_limit - datetime.timedelta(hours=24)

        date_logic = (dt_pngs >= lower_limit) & (dt_pngs <= upper_limit)

        dt_pngs = dt_pngs[date_logic]
        dt_pngs = list(dt_pngs)

        sat_names = sat_names[date_logic]
        sat_names = list(sat_names)

        # append last frame since it tends to get skipped sometimes
        dt_pngs.append(dt_pngs[-1])
        sat_names.append(sat_names[-1])

        # append first frame since it tends to get skipped sometimes
        dt_pngs.insert(0, dt_pngs[0])
        sat_names.insert(0, sat_names[0])

        if os.path.isfile(outpath): # if it already exists then delete it
            print("File {} already exists...deleting before creating new file".format(outpath))
            os.remove(outpath)

        with open(outpath, "w") as f:
            for i in range(len(dt_pngs)):
                f.write("file '{}'\n".format(dt2fname(dt_pngs[i], sat_names[i])))
                f.write("duration {}\n".format(args.frame_rate))

    print("Finished creating video metadata file.\n")
    END_TIME = datetime.datetime.now()
    print('Time taken to execute {}: {}'.format(os.path.basename(__file__), END_TIME - START_TIME))
