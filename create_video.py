import os
import sys
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser(prog='create_video')
    parser.add_argument('--fdir', type=str, metavar='',
                        help='Top-level source directory\n')
    args = parser.parse_args()

    # fdir = sys.argv[1]
    # frame_rate = sys.argv[2]
    subs = sorted([f for f in os.listdir(args.fdir) if os.path.isdir(os.path.join(args.fdir, f))])

    for sub in subs:

        mp4name = os.path.join(args.fdir, sub) + '.mp4'

        if os.path.isfile(mp4name): # if it already exists then delete it
            print("File {} already exists...deleting before creating new file".format(mp4name))
            os.remove(mp4name)
        # command = "ffmpeg -r {} -start_number 0 -i {}/%d.png -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p {}_fr{}.mp4".format(frame_rate, os.path.join(fdir, sub), os.path.join(fdir, sub), frame_rate)
        command = "ffmpeg -f concat -i {} -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p {}".format(os.path.join(args.fdir, sub, 'create_video_metadata.txt'), mp4name)
        # command = "ffmpeg -f concat -i {} -framerate 4/1 -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p {}.avi".format(os.path.join(fdir, sub, 'video.txt'), os.path.join(fdir, sub))
        os.system(command)
        # break
        # command = "ffmpeg -r 2 -start_number 0 -i {}/%d.png -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p {}.mp4".format(os.path.join(fdir, sub), os.path.join(fdir, sub))
        # os.system(command)

    print("Finished creating video files in {}.\n".format(args.fdir))
