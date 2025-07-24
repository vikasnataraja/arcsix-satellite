import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import cartopy.crs as ccrs

# Add every font at the specified location
# parent directory
current_dir = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

# set matplotlib style
MPL_STYLE_PATH = os.path.join(current_dir, 'util/plotting_utils/sifsat_whitestyle.mplstyle')


def set_plot_fonts(plt, serif_style='sans-serif', font='Helvetica Neue'):
    """
    Set the fonts for matplotlib plots.
    This function configures the font settings for matplotlib plots by adding
    custom fonts from a specified directory and setting the global font family.

    Args:
    ----
        plt : module
            The matplotlib.pyplot module.
        serif_style : str, optional
            The style of the font family to use, one of 'serif' or 'sans-serif'.
        font : str, optional
            The name of the font to use (default is 'Helvetica Neue').
    """

    # look for fonts in the fonts directory
    all_fonts_dir = os.path.join(current_dir, 'util/plotting_utils/fonts/')
    font_dir = [os.path.join(all_fonts_dir, f) for f in os.listdir(all_fonts_dir) if os.path.isdir(os.path.join(all_fonts_dir, f))]
    # add font if available from user's environment
    if 'MPL_FONT_DIR' in os.environ.keys():
        font_dir.append(os.environ['MPL_FONT_DIR'])

    # add all the fonts to matplotlib's library
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)

    # set font family globally (in-place for plt)
    plt.rc('font', **{'family':'{}'.format(serif_style),
                      '{}'.format(serif_style):['{}'.format(font)]})



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
                    'baffin':    {'view_extent': [-100, -40, 67, 84],
                                'vlon':         -60,
                                'vlat':          84},
                    'villum': {'view_extent': [-40, 0, 80, 90],
                                            'vlon': -40,
                                            'vlat': 80},
                    'villum_to_north_pole': {'view_extent': [-50, 10, 80, 90],
                                            'vlon': -35,
                                            'vlat': 82},
                }

# quicklook settings
ql_settings = {'proj_data': ccrs.PlateCarree(),
            #    'proj_plot': ccrs.PlateCarree(),
               'dpi': 300,
               'pad_inches': 0.0,
               'bbox_inches': 'tight',
               'extent': [-80, -30, 71, 88]}
