import os
import sys
import subprocess
from argparse import ArgumentParser

no_video_dirs = ['ice_path', 'water_path', 'optical_thickness', 'cloud_phase', 'cloud_top_height_temperature']
if __name__ == "__main__":

    parser = ArgumentParser(prog='create_video')
    parser.add_argument('--fdir', type=str, metavar='',
                        help='Top-level source directory\n')
    args = parser.parse_args()

    subs = sorted([f for f in os.listdir(args.fdir) if os.path.isdir(os.path.join(args.fdir, f))])

    for sub in subs:
        if sub in no_video_dirs:
            print("Message [create_video]: Skipping {}...".format(sub))
            continue

        mp4name = os.path.join(args.fdir, sub) + '.mp4'
        print(mp4name)

        if os.path.isfile(mp4name): # if it already exists then delete it
            print("File {} already exists...deleting before creating new file".format(mp4name))
            os.remove(mp4name)

        meta_file = os.path.join(args.fdir, sub, 'create_video_metadata.txt')
        # command = "ffmpeg -f concat -i {} -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p {}".format(os.path.join(args.fdir, sub, 'create_video_metadata.txt'), mp4name)

        command = ["ffmpeg", "-f", "concat", "-i", "{}".format(meta_file), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "{}".format(mp4name)]

        # os.system(command)
        try:
            ret = subprocess.run(command, capture_output=True, check=True)
            print(' '.join(ret.args))
        except Exception as err:
            print(err, ret.returncode)

    print("Finished creating video files in {}.\n".format(args.fdir))
