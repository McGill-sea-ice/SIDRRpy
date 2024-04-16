"""
--------------------------------------------------------------------------------
Visualisation tool to show dataset format and characterstics
--------------------------------------------------------------------------------
Includes:

- plot_start_end_points :: Creates a figure mapping the start and end
                           postion of the tracked features in a pair of
                           SAR images.

- show_tripcolor_field  :: Creates a figure showing the stacked tripcolor of a specific field,
                           from the SIDRR dataset (i.e., triangle Area, dudx, rotation,
                           as specified by the user).

- show_deformations     :: Creates a figure showing the normal deformation rates (a), the
                           shear deformation rates (b) and the vorticity (c), deemded valid
                           for a specific date (data from multiple SAR image pairs are stacked).

- show_stacked_pairs    :: Creates a figure showing the area contour of each of the SAR image
                           pairs included in a given SIDRR netcdf file.

- plot_triangles        :: Creates 2 figures zooming on the data calculated from a specific
                           SAR image pair, with the normal deformation rate in the background.
                           The second figure is optional, and offer a closer zoom on the
                           triangulated data and motion vectors.

- plot_area_dist        :: Creates a figure showing data distributions from the SIDRR object,
                           as specified by the user.

- show_spatial_coverage :: Creates a figure showing the percent daily coverage of the SIDRR
                           dataset, in the analysis period and in 10x10km bins.

- show_coverage_tseries :: Creates a figure showing timeseries of the SIDRR data coverage
                           based on 10x10km bins.


Authors: Mathieu Plante, Amelie Bouchat, Damien Ringeisen, Lekima Yakuden, Beatrice Duval
"""

# Loading from default packages
import os
import sys
parent = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,parent)
from datetime import datetime
from netCDF4 import Dataset
import numpy as np
import pyproj as pyproj
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from tqdm import tqdm
import haversine as hs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull, convex_hull_plot_2d


class visualisation:

    def __init__(self, config= None):
        """
        Initialising the object with the namelist,
        Input: - config :: config parser object containing namelist options
                           see options.ini.
        """

        IO            = config['IO']
        output_folder = IO['output_folder']
        exp           = IO['exp']
        self.figsPath =  output_folder + '/' + exp + '/'
        os.makedirs(self.figsPath, exist_ok=True)


    def plot_start_end_points(self, data = None, datestring=None):
        """
        Plots the start and end points of the triangle vertices on a map.
        Start points are blue, end points are red.

        INPUTS:
        data          :: data object from the SIDRR dataset (LoadDataset.py)
        datestring    :: string indicating date of 1st image acquisition time

        OUTPUTS:---

        """
        print('--- Plotting start and end points ---')

        # Set the matplotlib projection and transform
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        trans = ccrs.Geodetic()

        # Initialize figure
        fig = plt.figure(figsize=(5.5, 5.5))
        ax = fig.add_subplot(projection = proj, frameon=False)

        lxextent = -4400000
        uxextent = 2500000
        uyextent = 3500000
        lyextent = -2500000
        ax.set_extent((lxextent, uxextent, uyextent, lyextent), ccrs.NorthPolarStereo())

        #---------------------------------
        # Get tracked features position and add scatter to figure
        #---------------------------------

        # Fetch a specific image pair
        j = np.unique(data.day_flag)[0]
        no_day = data.idpair[np.where(data.day_flag==j)]
        i = np.unique(no_day)[0]

        # Get the first and last row of data corresponding to the specific pair of SAR images
        condi = (data.idpair[:] == i) & (data.day_flag[:] == j)
        min_index = np.where(condi)[0][0]
        max_index = np.where(condi)[0][-1]+1

        #Reconstruct the position vectors used for triangulation
        LatVector, LonVector = data.reconstruct_position_lists(min_index = min_index, max_index = max_index)
        LatVectorEnd, LonVectorEnd = data.reconstruct_position_lists(min_index = min_index, max_index = max_index, EndPoint = True)

        # Converting start/end lat/lons to x/y (North pole stereographic)
        new_coords     = proj.transform_points(trans, np.array(LonVector), np.array(LatVector))
        new_coords_end = proj.transform_points(trans, np.array(LonVector), np.array(LatVector))

        # Plotting start points (Blue) and end points (Red)
        ax.scatter(new_coords[:,0], new_coords[:,1], color = 'blue', s = 0.1, marker='x')
        ax.scatter(new_coords_end[:,0], new_coords_end[:,1], color = 'red', s = 0.1, marker='+')

        #--------------------------------------------
        # Figure labeling and saving
        #--------------------------------------------

        # Show lat/lon grid and coastlines
        ax.gridlines(draw_labels=True)
        ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')

        if datestring is None:
            prefix = "undefined_date"
        else:
            prefix = datestring

        FigPath   = self.figsPath + prefix + '_start_end_points.png'

        print('Saving start and end points figure at %s' % (FigPath))
        fig.savefig(FigPath, bbox_inches='tight', dpi=600)
        plt.close(fig)


    def show_tripcolor_field(self, data=None, Field = None, title = None, label = None, datestring=None):
        """
        This function plots deformations from a netCDF file using matplotlib's ax.tripcolor.
        The function assumes that the netCDF was generated from src/SeaIceDeformation's M01_d03_compute_deformations.py.

        INPUTS: - data        :: SIDRR data object (From LoadDataset.py).
                - Field       :: String labelling the mapped field (chosen by user).
                - Title       :: String to be add in the figure as title (above plot).
                - label       :: String added in the figure name to identify the
                                 mapped field.
                - datestring  :: string indicating the start and end dates of the
                                 analysis.

        OUTPUTS: ---
        """

        # Set the matplotlib projection and transform
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        trans = ccrs.Geodetic()

        # Initialize figures for total deformation (tot), divergence (I) and shear (II)
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=proj)
        ax.set_extent((-3800000, 2300000, 3000000, -2500000), ccrs.NorthPolarStereo())

        #---------------------------------
        # Get data for specfic SAR image pair and prepare for tripcolor
        #---------------------------------

        # Selecting a specific SAR image pair (Unique triangulations)
        for j in tqdm(np.unique(data.day_flag)):
          no_day = data.idpair[np.where(data.day_flag==j)]
          for i in tqdm(np.unique(no_day)):

            # Get the first and last row of data corresponding to the specific pair of SAR images
            condi = (data.idpair[:] == i) & (data.day_flag[:] == j)
            min_index = np.where(condi)[0][0]
            max_index = np.where(condi)[0][-1]+1

            # Get vertex ids from specific pair, and stack into triangle array, for tripcolor
            triangles = np.stack((data.ids1[min_index:max_index],
                                  data.ids2[min_index:max_index],
                                  data.ids3[min_index:max_index]), axis=-1)

            #Reconstruct the position vectors used for triangulation
            LatVector, LonVector = data.reconstruct_position_lists(min_index = min_index, max_index = max_index)

            #Keep only values from specific SAR pair
            data_colours = Field[min_index:max_index]

            # tranform the coordinates already to improve the plot efficiency
            new_coords = proj.transform_points(trans, np.array(LonVector), np.array(LatVector))
            tria = tri.Triangulation(new_coords[:,0], new_coords[:,1], triangles=triangles)

            #--------------------------------------------
            # Add tripcolor to figure
            #--------------------------------------------

            if len(triangles) != 0:
                if np.nanmin(data_colours) >= 0.0:
                    cb = ax.tripcolor(tria, facecolors=data_colours, cmap='plasma', vmin=0.0, vmax=np.nanmax(data_colours))
                elif np.nanmax(data_colours) <= 0.0:
                    cb = ax.tripcolor(tria, facecolors=data_colours, cmap='plasma_r', vmin=np.nanmin(data_colours), vmax=0.0)
                else:
                    c_lim = (np.nanmax(data_colours[:]**2.0))**0.5
                    cb = ax.tripcolor(tria, facecolors=data_colours, cmap='coolwarm', vmin=-c_lim, vmax=c_lim)

        #--------------------------------------------
        # Figure labeling and saving
        #--------------------------------------------

        #Add land and labels
        plt.text(0.5,1.02,title,ha='center', va='center', transform=ax.transAxes,fontsize=8)
        ax.gridlines()
        ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')

        # Add a colorbar
        clb = plt.colorbar(cb, ax = ax, shrink = 0.9,pad=.04)
        clb.ax.tick_params(labelsize=8)

        if datestring is None:
            prefix = "undefined_date"
        else:
            prefix = datestring

        FigPath   = self.figsPath + prefix + '_' + label + '.png'
        print('Saving %s figure at %s' % (title, FigPath))
        fig.savefig(FigPath, bbox_inches='tight', dpi=600)
        plt.close(fig)




    def plot_deformations(self, data=None, datestring=None):
        """
        This function plots deformations from a netCDF file using matplotlib's ax.tripcolor.
        The function assumes that the netCDF was generated from src/SeaIceDeformation's M01_d03_compute_deformations.py.

        INPUTS: - data        :: SIDRR data object (From LoadDataset.py)
                - datestring  :: string indicating the start and end dates of the
                                 analysis.

        OUTPUTS:
        None -- Saves plot to the output directory
        """

        # Set the matplotlib projection and transform
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        trans = ccrs.Geodetic()

        # Initialize figures for total deformation (tot), divergence (I) and shear (II)
        fig_defs = plt.figure(figsize=(5, 9))

        # Initialize subplots
        ax_div = fig_defs.add_axes([0.1, 0.62, 0.8, 0.25], projection=proj)
        ax_shr = fig_defs.add_axes([0.1, 0.34, 0.8, 0.25], projection=proj)
        ax_vrt = fig_defs.add_axes([0.1, 0.06, 0.8, 0.25], projection=proj)

        # Create a list of axes to be iterated over
        ax_list = [ax_div, ax_shr, ax_vrt]
        for ax in ax_list:
            ax.set_extent((-3800000, 2300000, 3000000, -2500000), ccrs.NorthPolarStereo())


        #---------------------------------
        # Get data for specfic SAR image pair and prepare for tripcolor
        #---------------------------------

        print('--- Creating sea-ice deformation figures ---')

        # Looping over SAR image pairs (each image pair IDs from each daily netcdf)
        for j in tqdm(np.unique(data.day_flag)):
          no_day = data.idpair[np.where(data.day_flag==j)]
          for i in tqdm(np.unique(no_day)):

            # Get the first and last row of data corresponding to the specific pair of SAR images
            condi = (data.idpair[:] == i) & (data.day_flag[:] == j)
            min_index = np.where(condi)[0][0]
            max_index = np.where(condi)[0][-1]+1

            # Get vertex ids from specific pair, and stack into triangle array, for tripcolor
            triangles = np.stack((data.ids1[min_index:max_index],
                                  data.ids2[min_index:max_index],
                                  data.ids3[min_index:max_index]), axis=-1)

            #Reconstruct the position vectors used for triangulation
            LatVector, LonVector = data.reconstruct_position_lists(min_index = min_index, max_index = max_index)

            div_colours = data.div[min_index:max_index]
            shr_colours = data.shr[min_index:max_index]
            vrt_colours = data.vrt[min_index:max_index]

            # tranform the coordinates already to improve the plot efficiency
            new_coords = proj.transform_points(trans, np.array(LonVector), np.array(LatVector))
            tria = tri.Triangulation(new_coords[:,0], new_coords[:,1], triangles=triangles)

            #--------------------------------------------
            # Add tripcolor to figure
            #--------------------------------------------
            if len(triangles) != 0:
                cb_div = ax_div.tripcolor(tria, facecolors=div_colours, cmap='coolwarm', vmin=-0.04, vmax=0.04)
                cb_shr = ax_shr.tripcolor(tria, facecolors=shr_colours, cmap='plasma', vmin=0, vmax=0.1)
                cb_vrt = ax_vrt.tripcolor(tria, facecolors=vrt_colours, cmap='coolwarm', vmin=-0.1, vmax=0.1)

        #--------------------------------------------
        # Labeling and saving
        #--------------------------------------------

        # Create a list of colorbars and titles to be iterated over
        cb_list = [cb_div, cb_shr, cb_vrt]

        #List of titles and labels
        eI_title ="$\dot{\epsilon}_{I}$ (day$^{-1}$)"
        eII_title = "$\dot{\epsilon}_{II}$ (day$^{-1}$)"
        zeta_title = "$\zeta$ (day$^{-1}$)"
        title_list = [eI_title, eII_title,zeta_title]
        lbl_list = ["a)","b)","c)"]

        for ax, title, cb, lbl in zip(ax_list, title_list, cb_list, lbl_list):

            # Add a colorbar
            clb = plt.colorbar(cb, ax = ax, shrink = 0.9,pad=.04)
            clb.ax.tick_params(labelsize=8)

            # Add colorbar label
            plt.text(1.12,1.02,title,ha='center', va='center', transform=ax.transAxes,fontsize=8)

            #Panel label
            plt.text(-0.1,0.95,lbl,ha='center', va='center', transform=ax.transAxes,fontsize=12)

            #Grid and landmask
            ax.gridlines()
            ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')

        # Create the figure filenames
        if datestring is None:
            prefix = "undefined_date"
        else:
            prefix = datestring

        defs_path  = self.figsPath + prefix + '_defs.png'
        print("Printing deformation figure at : %s" % defs_path)
        fig_defs.savefig(defs_path, bbox_inches='tight', dpi=600)
        plt.close(fig_defs)

        return


    def show_stacked_pairs(self, data=None,  datestring=None):
        """
        This function prints the contours (convex hull) of the tracked
        points in a spectic netCDF.

        INPUTS:
        data       :: dataset object including data from SIDRR netcdf file
                      (LoadDataset.py)
        datestring :: String indicating the SIDRR netcdf date

        OUTPUTS: --
        """

        # Set the matplotlib projection and transform
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        trans = ccrs.Geodetic()

        # Initialize figure
        fig_pairs = plt.figure(figsize=(5, 5))
        ax_pairs = fig_pairs.add_axes([0.1, 0.1, 0.8, 0.8], projection=proj)
        ax_pairs.set_extent((-4400000, 2500000, 3500000, -2500000), ccrs.NorthPolarStereo())

        print('--- Creating figure showing SAR pair areas ---')


        #---------------------------------
        # Get data for each pair and add its points and area to figure
        #---------------------------------

        # Looping over SAR image pairs (each image pair IDs in the netcdf)
        j  = np.unique(data.day_flag)[0]
        no_day = data.idpair[np.where(data.day_flag==j)]
        for i in tqdm(np.unique(no_day)):
            # Get the first and last row of data corresponding to the specific pair of SAR images
            condi = (data.idpair[:] == i) & (data.day_flag[:] == j)
            min_index = np.where(condi)[0][0]
            max_index = np.where(condi)[0][-1]+1

            # Get vertex ids from specific pair, and stack into triangle array, for tripcolor
            triangles = np.stack((data.ids1[min_index:max_index],
                                  data.ids2[min_index:max_index],
                                  data.ids3[min_index:max_index]), axis=-1)

            # Get the list of tracked position and add scatter in figure
            LatVector, LonVector = data.reconstruct_position_lists(min_index = min_index, max_index = max_index)
            points = proj.transform_points(trans, np.array(LonVector), np.array(LatVector))
            ax_pairs.scatter(points[:,0], points[:,1], color = 'red', s = 0.1, marker='+')
            points = points[:,:-1]
            points = points[~np.isnan(points[:,0]),:]

            # Make the hull around the points and add area contour line to figure
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

            # Get center location for ID labeling, and write to figure.
            x_mean = np.nanmean(np.squeeze(points[hull.simplices,0]))
            y_mean = np.nanmean(np.squeeze(points[hull.simplices,1]))
            ax_pairs.text(x_mean, y_mean,str(i),fontsize = 3,
                          verticalalignment ='center',
                                  horizontalalignment = 'center')

        #--------------------------------------------
        # Labeling and saving
        #--------------------------------------------

        # Add gridlines
        ax_pairs.gridlines()
        ax_pairs.add_feature(cfeature.LAND, zorder=100, edgecolor='k')

        # Create the figure filenames
        if datestring is None:
            prefix = "undefined_date"
        else:
            prefix = datestring
        pair_path  = self.figsPath + prefix + '_pairs.png'

        print('Saving pairs area figure at ',pair_path)
        fig_pairs.savefig(pair_path, bbox_inches='tight', dpi=600)
        plt.close(fig_pairs)

        return



    def plot_triangles(self, data=None,  datestring=None, idpair = None, triangle_zoom = None):
        """
        This function makes figures zooming on the triangulared data
        from a specific SAR image pair.

        INPUTS:
        data          :: data object from the SIDRR dataset (LoadDataset.py)
        datestring    :: string indicating date of 1st image acquisition time
        no            :: ID of the specific SAR image pair investigated
        triangle_zoom :: if true, a figure zooming on the triangles is also produced.

        OUTPUTS:   :: ---
        """


        # Set the matplotlib projection and transform
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        trans = ccrs.Geodetic()

        # Initialize figure
        fig_tris = plt.figure(figsize=(5, 5))
        ax_tris = fig_tris.add_axes([0.1, 0.1, 0.8, 0.8], projection=proj)
        ax_tris.set_extent((-4400000, 2500000, 3500000, -2500000), ccrs.NorthPolarStereo())

        print('--- Creating figures zooming on SAR image pair ID = %s ---' % no)

        #---------------------------------
        # Get data from required SAR image pair
        #---------------------------------

        # Fetch the image pair data
        j = np.unique(data.day_flag)[0]
        no_day = data.idpair[np.where(data.day_flag==j)]
        i = int(idpair)

        # Get the first and last row of data corresponding to the specific pair of SAR images
        condi = (data.idpair[:] == i) & (data.day_flag[:] == j)
        min_index = np.where(condi)[0][0]
        max_index = np.where(condi)[0][-1]+1

        # Get vertex ids from specific pair, and stack into triangle array, for tripcolor
        triangles = np.stack((data.ids1[min_index:max_index],
                              data.ids2[min_index:max_index],
                              data.ids3[min_index:max_index]), axis=-1)

        #Reconstruct the position vectors used for triangulation
        LatVector, LonVector = data.reconstruct_position_lists(min_index = min_index, max_index = max_index)

        #Keep only values from specific SAR pair
        data_colours = data.div[min_index:max_index]

        # tranform the coordinates already to improve the plot efficiency
        new_coords = proj.transform_points(trans, np.array(LonVector), np.array(LatVector))
        tria = tri.Triangulation(new_coords[:,0], new_coords[:,1], triangles=triangles)

        #---------------------------------
        # Calculate new boundaries to zoom on figure
        #---------------------------------

        # Get 4 reference point for map extent, based on the tracked point positions
        x,y = new_coords[:,0],new_coords[:,1]
        a1 = np.nanmin(new_coords[:,0])
        a2 = np.nanmax(new_coords[:,0])
        a3 = np.nanmin(new_coords[:,1])
        a4 = np.nanmax(new_coords[:,1])
        a = len(x)

        points = new_coords.copy()
        points = points[:,:-1]
        points = points[~np.isnan(points[:,0]),:]
        try:
            ax_tris.set_extent((a1, a2, a3, a4),proj)
        except:
            sys.exit("error in extent : %s, %s, %s, %s" % (a1, a2, a3, a4))


        #---------------------------------
        # Add pcolor and hull contour to figure
        #---------------------------------

        if len(triangles) != 0:

            #Add tripcolor of the divergence rate calculated from the SAR image pair
            cb_div = ax_tris.tripcolor(tria, facecolors=data_colours,cmap='coolwarm',
                                       vmin=-0.04, vmax=0.04, edgecolors='k')

            # Make the hull around the points and add area contour line to figure
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')


        #--------------------------------------------
        # Labeling and saving
        #--------------------------------------------

        plt.title("Triangulated data for pair id = %s" % idpair )
        ax_tris.gridlines()
        ax_tris.add_feature(cfeature.LAND, zorder=100, edgecolor='k')

        # Create the figure filenames
        if datestring is None:
            prefix = "undefined_date"
        else:
            prefix = datestring

        tri_path  = '%s%s_pair_id_%s.png' % (self.figsPath,prefix,idpair)
        print('Saving triangulated data figure at ',tri_path)
        fig_tris.savefig(tri_path, dpi=600)
        plt.close(fig_tris)

        #=============================================================
        #--------------------------------------------
        # Making an other figure, if zoom_on_triangle is true:
        #--------------------------------------------
        #=============================================================

        if (len(triangles) != 0) and (triangle_zoom is True):

            # Initialize figure
            fig_zoom = plt.figure(figsize=(5, 5))
            ax_zoom = fig_tris.add_axes([0.1, 0.1, 0.8, 0.8], projection=proj)

            #---------------------------------
            # Get additional motion vector data from SAR image pair
            #---------------------------------

            #Make ID and position vectors by concatenating the 3 vertex IDs and positions
            IDs = np.concatenate((data.ids1[min_index:max_index], data.ids2[min_index:max_index], data.ids3[min_index:max_index]), axis=0)

            #Also get the end point coordinates
            LatVectorEnd, LonVectorEnd = data.reconstruct_position_lists(min_index = min_index, max_index = max_index, EndPoint = True)
            new_coords_end = proj.transform_points(trans, np.array(LonVectorEnd), np.array(LatVectorEnd))
            tria_end = tri.Triangulation(new_coords_end[:,0], new_coords_end[:,1], triangles=triangles)

            #Get the drift of each point between the 2 images, in X-Y coords.
            deltaXY = new_coords_end - new_coords

            #---------------------------------
            # Add pcolor and start/end positions to figure
            #---------------------------------

            #Show triangles with no fill color
            cb_div_zoom = ax_zoom.tripcolor(tria, facecolors=data_colours*np.nan,cmap='coolwarm', vmin=-0.04, vmax=0.04, edgecolors='k')

            #Scatter start and end positions
            ax_zoom.scatter(new_coords[:,0], new_coords[:,1], color = 'k', s = 5, marker='*',label = 'start points')
            ax_zoom.scatter(new_coords_end[:,0], new_coords_end[:,1], color = 'b', s = 5, marker='*',label = 'end points')

            #Add vertex ID labels
            for idk in tqdm(np.unique(IDs)):
                IDxk = new_coords[idk,0]
                IDyk = new_coords[idk,1]
                IDstr = str(idk)
                t = ax_zoom.text(IDxk, IDyk,IDstr,fontsize = 4,clip_on=True)
                t.clipbox = ax_zoom.bbox

                #Add motion vector
                dX = deltaXY[idk,0]
                dY = deltaXY[idk,1]
                ax_zoom.quiver(IDxk, IDyk, dX, dY, angles='xy',
                               scale_units='xy', scale=1, width = 0.005,color = 'b')

            #--------------------------------------------
            # Set the zoomed figure mapping extent
            #--------------------------------------------

            x_mean = np.nanmean(np.squeeze(points[hull.simplices,0]))
            y_mean = np.nanmean(np.squeeze(points[hull.simplices,1]))
            ax_zoom.set_extent([x_mean - (a2-a1)/20,
                                x_mean + (a2-a1)/20,
                                y_mean - (a4-a3)/30,
                                y_mean + (a4-a3)/30], proj)

            #--------------------------------------------
            # Labeling and saving
            #--------------------------------------------

            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                          ncol=2, mode="expand", borderaxespad=0.,fontsize=10.0 )

            zoom_path   = '%s%s_pair_id_%s_zoomed.png' % (self.figsPath,prefix,idpair)
            print('Saving zoomed data figure at ',zoom_path)

            fig_zoom.savefig(zoom_path, dpi=600)
            plt.close(fig_zoom)

        return


    def plot_area_dist(self, dist1 = None, dist2 = None, dist3 = None,
                             distDt1 = None, distDt2 = None, distDt3 = None,
                             datestring = None):

        """
        This function produces a figure showing distributions of
        SIDRR data, as defined by the input 1Ddistribution objects.

        Input: - dist1,2,3      :: Analysis objects data include the histograms to be printed.
                                   The object class is defined in Statistics_objects.py
               - datestring     :: string indicating the start and end dates of the
                                   analysis.
        """

        # figure initialization
        fig = plt.figure(figsize=(5, 8))

        #---------------------------------------
        #Pannel with the A distribution
        #---------------------------------------

        ax = fig.add_axes([0.1, 0.57, 0.8, 0.38])
        if dist2 is None and dist3 is None:
            plt1 = plt.bar(dist1.bins,dist1.distribution/np.sum(dist1.distribution),width = dist1.BinsWidth, color = 'b')
        else:
            plt1 = plt.bar(dist1.bins,dist1.distribution/np.sum(dist1.distribution),width = dist1.BinsWidth, alpha = 0.5, color = 'darkblue')
        if dist2 is not None:
            plt2 = plt.bar(dist2.bins,dist2.distribution/np.sum(dist2.distribution),width = dist2.BinsWidth, alpha = 0.5, color = 'darkorange')
        if dist3 is not None:
            plt3 = plt.bar(dist3.bins,dist3.distribution/np.sum(dist3.distribution),width = dist3.BinsWidth, alpha = 0.7, color = 'y')

        plt.xticks(dist1.bins)
        if dist2 is not None:
            if dist3 is not None:
                plt.legend((dist1.label,dist2.label,dist3.label))
            else:
                plt.legend((dist1.label,dist2.label))
        dens = True
        if dens:
            plt.ylabel('PDF')
        else :
            plt.ylabel('Number of triangles')
        plt.xlabel('L (km)')
        plt.xticks(range(0,25,5))
        plt.xlim(0,20)

        # Make figure path name and save
        if datestring is None:
            prefix = "undefined_date"
        else:
            prefix = datestring
        fig_path = self.figsPath + prefix + '_AT_hist.png'
        fig.savefig(fig_path, bbox_inches='tight', dpi=600)

        #---------------------------------------
        #Pannel with the Delta-t distribution
        #---------------------------------------
        if distDt1 is not None:
            ax2 = fig.add_axes([0.1, 0.07, 0.8, 0.38])

            if distDt2 is None and distDt3 is None:
                plt1 = plt.bar(distDt1.bins,distDt1.distribution/np.sum(distDt1.distribution),width = distDt1.BinsWidth, color = 'darkblue')
            else:
                plt1 = plt.bar(distDt1.bins,distDt1.distribution/np.sum(distDt1.distribution),width = distDt1.BinsWidth, alpha = 0.5, color = 'darkblue')
            if distDt2 is not None:
                plt2 = plt.bar(distDt2.bins,distDt2.distribution/np.sum(distDt2.distribution),width = distDt2.BinsWidth, alpha = 0.5, color = 'darkorange')
            if distDt3 is not None:
                plt3 = plt.bar(distDt3.bins,distDt3.distribution/np.sum(distDt3.distribution),width = distDt3.BinsWidth, alpha = 0.7, color = 'y')

            plt.xticks(distDt1.bins)
            if distDt2 is not None:
                if distDt3 is not None:
                    plt.legend((distDt1.label,distDt2.label,distDt3.label))
                else:
                    plt.legend((distDt1.label,distDt2.label))
            dens = True
            if dens:
                plt.ylabel('PDF')
            else :
                plt.ylabel('Number of triangles')
            plt.xlabel('$\Delta$t (h)')
            #plt.xticks(range(0,144,6))
            plt.xlim(0,150)

            # Make figure path name and save
            if datestring is None:
                prefix = "undefined_date"
            else:
                prefix = datestring
            fig_path = self.figsPath + prefix + '_AT_hist.png'
            fig.savefig(fig_path, bbox_inches='tight', dpi=600)


        plt.close(fig)

        return



    def show_spatial_coverage(self, distribution_2D = None,
                                    distribution_2D_2 = None,
                                    distribution_2D_3 = None,
                                    datestring = None):
        """
        This function produces a figure showing the daily coverage frequency
        in the SIDRR data, in 10x10 km bins, in the analysis time period
        as indicated in the datestring.

        Input: - 2Ddistribution :: Analysis object including a pan Arctic histogram counting
                                   the number of days with data in 10x10km bins. This class is
                                   defined in Statistics_objects.py
               - datestring     :: string indicating the start and end dates of the
                                   analysis.
        Output: ---
        """
        #Create figure and prepare projection
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        trans = ccrs.Geodetic()

        if distribution_2D_2 is None:
            fig = plt.figure(figsize=(6.5, 5.5))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8],projection = proj)
        else:
            fig = plt.figure(figsize=(10, 4.0))
            ax = fig.add_axes([0.07, 0.1, 0.3, 0.8],projection = proj)
        #Prepare data and projection for mapping
        xx, yy = np.meshgrid(distribution_2D.xbins, distribution_2D.ybins)

        distribution_2D.H[distribution_2D.H==0.0] = np.nan
        H = distribution_2D.H*100.0 /distribution_2D.ntime
        H = H.T

        #print the frequency map
        cmap1 = mpl.colormaps['plasma']
        cmap1.set_bad('w')
        im = ax.pcolormesh(xx, yy, H, vmin = 0.0, vmax = 100.0,cmap=cmap1)
        ax.gridlines()
        ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
        plt.text(0.5,1.1,'S1 + RCM',ha='center', va='center', transform=ax.transAxes,fontsize=12)
        plt.text(0.08,0.9,'a)',ha='center', va='center', transform=ax.transAxes,fontsize=12)



        if distribution_2D_2 is not None:
            ax2 = fig.add_axes([0.4, 0.1, 0.3, 0.8],projection = proj)
            #Prepare data and projection for mapping
            xx, yy = np.meshgrid(distribution_2D_2.xbins, distribution_2D_2.ybins)

            distribution_2D_2.H[distribution_2D_2.H==0.0] = np.nan
            H = distribution_2D_2.H*100.0 /distribution_2D_2.ntime
            H = H.T

            #print the frequency map
            cmap1 = mpl.colormaps['plasma']
            cmap1.set_bad('w')
            im2 = ax2.pcolormesh(xx, yy, H, vmin = 0.0, vmax = 100.0,cmap=cmap1)
            plt.text(1.12,1.02,'Freq.(%)',ha='center', va='center', transform=ax2.transAxes,fontsize=8)
            ax2.gridlines()
            ax2.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
            plt.text(0.5,1.1,'S1 only',ha='center', va='center', transform=ax2.transAxes,fontsize=12)
            plt.text(0.08,0.9,'b)',ha='center', va='center', transform=ax2.transAxes,fontsize=12)

            cb_ax = fig.add_axes([0.3,0.05,0.4,0.04])
            clb = plt.colorbar(im2, orientation='vertical',cax=cb_ax)
            clb.set_ticks(np.arange(0, 110, 10))
            clb.set_ticklabels(np.arange(0, 110, 10))
            clb.ax.tick_params(labelsize=8)


        if distribution_2D_3 is not None:
            ax3 = fig.add_axes([0.73, 0.1, 0.3, 0.8],projection = proj)
            #Prepare data and projection for mapping
            xx, yy = np.meshgrid(distribution_2D_3.xbins, distribution_2D_3.ybins)

            distribution_2D_3.H[distribution_2D_3.H==0.0] = np.nan
            H = distribution_2D_3.H*100.0 /distribution_2D_3.ntime
            H = H.T

            #print the frequency map
            cmap1 = mpl.colormaps['plasma']
            cmap1.set_bad('w')
            im3 = ax3.pcolormesh(xx, yy, H, vmin = 0.0, vmax = 100.0,cmap=cmap1)
#            clb = plt.colorbar(im3, shrink = 0.9,pad=.04)
#            clb.set_ticks(np.arange(0, 110, 10))
#            clb.set_ticklabels(np.arange(0, 110, 10))
#            clb.ax.tick_params(labelsize=8)
#            plt.text(1.12,1.02,'Freq.(%)',ha='center', va='center', transform=ax3.transAxes,fontsize=8)
            ax3.gridlines()
            ax3.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
            plt.text(0.5,1.1,'RCM only',ha='center', va='center', transform=ax3.transAxes,fontsize=12)
            plt.text(0.08,0.9,'c)',ha='center', va='center', transform=ax3.transAxes,fontsize=12)


        #Save figure in output folder
        if datestring is None:
            prefix = "undefined_date"
        else:
            prefix = datestring

        fig_path = self.figsPath + prefix + '_Coverage_area_map.png'
        print('Saving coverage 2D histogram figure at ' + fig_path)
        fig.savefig(fig_path, dpi=600)
        plt.close(fig)

        return


    def show_coverage_tseries(self, Data1 = None,
                                    Data2 = None,
                                    Data3 = None,
                                    datestring = None):

        labels_list = ['2017',' ', '2018',
                              ' ', '2019',
                              ' ', '2020',
                              ' ', '2021',
                              ' ', '2022',
                              ' ', '2023',]

        date_list = [datetime(2017,1,1,hour=0),
                     datetime(2017,7,1,hour=0),
                     datetime(2018,1,1,hour=0),
                     datetime(2018,7,1,hour=0),
                     datetime(2019,1,1,hour=0),
                     datetime(2019,7,1,hour=0),
                     datetime(2020,1,1,hour=0),
                     datetime(2020,7,1,hour=0),
                     datetime(2021,1,1,hour=0),
                     datetime(2021,7,1,hour=0),
                     datetime(2022,1,1,hour=0),
                     datetime(2022,7,1,hour=0),
                     datetime(2023,1,1,hour=0)]

        #2022-01-01 is day number 44562 since 1900-01-01
        reference_date = datetime(2022,1,1,hour=0)

        DateTicks = []
        for date in date_list:
            new_tick = 44562.0 +(date - reference_date).days + ((date - reference_date).seconds /(24*60*60))
            DateTicks.append(new_tick)


        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        plt.plot(Data1.timeaxis,Data1.covered_area_tseries,'k-')
        plt.plot(Data2.timeaxis,Data2.covered_area_tseries,'b-')
        plt.plot(Data3.timeaxis,Data3.covered_area_tseries,'orange')

        plt.xticks(DateTicks,labels_list)
        plt.xlim(Data1.timeaxis[0],Data1.timeaxis[len(Data1.timeaxis)-1])


        #Save figure in output folder
        if datestring is None:
            prefix = "undefined_date"
        else:
            prefix = datestring

        fig_path = self.figsPath + prefix + '_Coverage_tseries.png'
        print('Saving coverage time series at ' + fig_path)
        fig.savefig(fig_path, dpi=600)
        plt.close(fig)

        return



    def plot_error_dist(self, dist1 = None, dist2 = None, dist3 = None, datestring = None):

        """
        This function produces a figure showing distributions of
        SIDRR data, as defined by the input 1Ddistribution objects.

        Input: - dist1,2,3      :: Analysis objects data include the histograms to be printed.
                                   The object class is defined in Statistics_objects.py
               - datestring     :: string indicating the start and end dates of the
                                   analysis.
        """
        centroids = dist1.bins #(dist1.bins[1:] + dist1.bins[:-1]) / 2

        # figure initialization
        fig = plt.figure(figsize=(6.5, 5.5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        if dist2 is None and dist3 is None:
            plt1 = plt.hist(centroids,bins = len(dist1.distribution),weights = dist1.distribution/np.nansum(dist1.distribution),
                            range = ( min(dist1.bins), max(dist1.bins)), color = 'b')
        else:
            plt1 = plt.hist(centroids,bins = len(dist1.distribution),weights = dist1.distribution/np.nansum(dist1.distribution),
                            range = ( min(dist1.bins), max(dist1.bins)), alpha = 0.5, color = 'b')
        if dist2 is not None:
            plt2 = plt.hist(centroids,bins = len(dist2.distribution),weights = dist2.distribution/np.nansum(dist2.distribution),
                            range = ( min(dist2.bins), max(dist2.bins)), alpha = 0.5, color = 'b')
        if dist3 is not None:
            plt3 = plt.hist(centroids,bins = len(dist3.distribution),weights = dist3.distribution/np.nansum(dist3.distribution),
                            range = ( min(dist3.bins), max(dist3.bins)), alpha = 0.5, color = 'b')

        if dist3 is not None:
            plt.legend((dist1.label,dist2.label,dist3.label))
        elif dist2 is not None:
            plt.legend((dist1.label,dist2.label))

        dens = True
        if dens:
            plt.ylabel('PDF')
        else :
            plt.ylabel('Number of triangles')
        plt.xlabel('propagation error')
        plt.xlim(0,0.6)
        #plt.xticks(range(0,10)/100)

        # Make figure path name and save
        if datestring is None:
            prefix = "undefined_date"
        else:
            prefix = datestring
        fig_path = self.figsPath + prefix + '_errors_hist.png'
        print("saving propagation histogram at %s" % fig_path)
        fig.savefig(fig_path, bbox_inches='tight', dpi=600)
        plt.close(fig)

        return


    def plot_s2n_dist(self, dist1 = None, dist2 = None, dist3 = None, datestring = None):

        """
        This function produces a figure showing distributions of
        SIDRR data, as defined by the input 1Ddistribution objects.

        Input: - dist1,2,3      :: Analysis objects data include the histograms to be printed.
                                   The object class is defined in Statistics_objects.py
               - datestring     :: string indicating the start and end dates of the
                                   analysis.
        """

        centroids = dist1.bins #(dist1.bins[1:] + dist1.bins[:-1]) / 2

        # figure initialization
        fig = plt.figure(figsize=(6.5, 5.5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        if dist2 is None and dist3 is None:
            plt1 = plt.hist(centroids,bins = len(dist1.distribution),weights = dist1.distribution/np.nansum(dist1.distribution),
                            range = ( min(dist1.bins), max(dist1.bins)), color = 'b')
        else:
            plt1 = plt.hist(centroids,bins = len(dist1.distribution),weights = dist1.distribution/np.nansum(dist1.distribution),
                            range = ( min(dist1.bins), max(dist1.bins)), alpha = 0.5, color = 'b')
        if dist2 is not None:
            plt2 = plt.hist(centroids,bins = len(dist2.distribution),weights = dist2.distribution/np.nansum(dist2.distribution),
                            range = ( min(dist2.bins), max(dist2.bins)), alpha = 0.5, color = 'b')
        if dist3 is not None:
            plt3 = plt.hist(centroids,bins = len(dist3.distribution),weights = dist3.distribution/np.nansum(dist3.distribution),
                            range = ( min(dist3.bins), max(dist3.bins)), alpha = 0.5, color = 'b')

        if dist3 is not None:
            plt.legend((dist1.label,dist2.label,dist3.label))

        dens = True
        if dens:
            plt.ylabel('PDF')
        else :
            plt.ylabel('Number of triangles')
        plt.xlabel('signal-to-noise ratio')
        plt.xlim(0,3)

        # Make figure path name and save
        if datestring is None:
            prefix = "undefined_date"
        else:
            prefix = datestring
        fig_path = self.figsPath + prefix + '_s2n_hist.png'
        fig.savefig(fig_path, bbox_inches='tight', dpi=600)
        plt.close(fig)

        return



    def plot_errors(self, data=None, datestring=None):
        """
        This function plots the deformation errors from a netCDF file using matplotlib's ax.tripcolor.
        The function assumes that the netCDF was generated from src/SeaIceDeformation's M01_d03_compute_deformations.py.

        INPUTS: - data        :: SIDRR data object (From LoadDataset.py)
                - datestring  :: string indicating the start and end dates of the
                                 analysis.

        OUTPUTS:
        None -- Saves plot to the output directory
        """

        # Set the matplotlib projection and transform
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        trans = ccrs.Geodetic()

        # Initialize figures for total deformation (tot), divergence (I) and shear (II)
        fig_errs = plt.figure(figsize=(5, 9))

        # Initialize subplots
        ax_div = fig_errs.add_axes([0.1, 0.62, 0.8, 0.25], projection=proj)
        ax_shr = fig_errs.add_axes([0.1, 0.34, 0.8, 0.25], projection=proj)
        ax_vrt = fig_errs.add_axes([0.1, 0.06, 0.8, 0.25], projection=proj)

        # Create a list of axes to be iterated over
        ax_list = [ax_div, ax_shr, ax_vrt]
        for ax in ax_list:
            ax.set_extent((-3800000, 2300000, 3000000, -2500000), ccrs.NorthPolarStereo())


        #---------------------------------
        # Get data for specfic SAR image pair and prepare for tripcolor
        #---------------------------------

        print('--- Creating sea-ice error figures ---')
        data.errII[data.errII>1000.0] = np.nan
        # Looping over SAR image pairs (each image pair IDs from each daily netcdf)
        for j in tqdm(np.unique(data.day_flag)):
          no_day = data.idpair[np.where(data.day_flag==j)]
          for i in tqdm(np.unique(no_day)):

            # Get the first and last row of data corresponding to the specific pair of SAR images
            condi = (data.idpair[:] == i) & (data.day_flag[:] == j)
            min_index = np.where(condi)[0][0]
            max_index = np.where(condi)[0][-1]+1

            # Get vertex ids from specific pair, and stack into triangle array, for tripcolor
            triangles = np.stack((data.ids1[min_index:max_index],
                                  data.ids2[min_index:max_index],
                                  data.ids3[min_index:max_index]), axis=-1)

            #Reconstruct the position vectors used for triangulation
            LatVector, LonVector = data.reconstruct_position_lists(min_index = min_index, max_index = max_index)

            div_colours = data.errI[min_index:max_index]
            shr_colours = data.errII[min_index:max_index]
            vrt_colours = data.s2n[min_index:max_index]
            # tranform the coordinates already to improve the plot efficiency
            new_coords = proj.transform_points(trans, np.array(LonVector), np.array(LatVector))
            tria = tri.Triangulation(new_coords[:,0], new_coords[:,1], triangles=triangles)

            #--------------------------------------------
            # Add tripcolor to figure
            #--------------------------------------------
            if len(triangles) != 0:
                cb_div = ax_div.tripcolor(tria, facecolors=div_colours, cmap='plasma',vmin=0.0, vmax=0.4)
                cb_shr = ax_shr.tripcolor(tria, facecolors=shr_colours, cmap='plasma',vmin=0.0, vmax=0.4)
                cb_vrt = ax_vrt.tripcolor(tria, facecolors=vrt_colours, cmap='plasma',vmin=0.0, vmax=5.0)

        #--------------------------------------------
        # Labeling and saving
        #--------------------------------------------
        # Create a list of colorbars and titles to be iterated over
        cb_list = [cb_div, cb_shr, cb_vrt]

        #List of titles and labels
        eI_title ="$e_{I}$ (day$^{-1}$)"
        eII_title = "$e_{II}$ (day$^{-1}$)"
        zeta_title = "signal to noise"
        title_list = [eI_title, eII_title,zeta_title]
        lbl_list = ["a)","b)","c)"]

        for ax, title, cb, lbl in zip(ax_list, title_list, cb_list, lbl_list):

            # Add a colorbar
            clb = plt.colorbar(cb, ax = ax, shrink = 0.9,pad=.04)
            clb.ax.tick_params(labelsize=8)

            # Add colorbar label
            plt.text(1.12,1.02,title,ha='center', va='center', transform=ax.transAxes,fontsize=8)

            #Panel label
            plt.text(-0.1,0.95,lbl,ha='center', va='center', transform=ax.transAxes,fontsize=12)

            #Grid and landmask
            ax.gridlines()
            ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')

        # Create the figure filenames
        if datestring is None:
            prefix = "undefined_date"
        else:
            prefix = datestring

        errs_path  = self.figsPath + prefix + '_errs.png'
        print("Printing error figure at : %s" % errs_path)
        fig_errs.savefig(errs_path, bbox_inches='tight', dpi=600)
        plt.close(fig_errs)


        fig_scatter = plt.figure(figsize=(6,5))

        # Initialize subplots
        ax = fig_scatter.add_axes([0.1, 0.6, 0.36, 0.36])
        ax2 = fig_scatter.add_axes([0.1, 0.1, 0.36, 0.36])

        print('--- Creating sea-ice deformation figures ---')

        # Looping over SAR image pairs (each image pair IDs from each daily netcdf)
        for j in tqdm(np.unique(data.day_flag)):
            no_day = data.idpair[np.where(data.day_flag==j)]
            for i in tqdm(np.unique(no_day)):

                # Get the first and last row of data corresponding to the specific pair of SAR images
                condi = (data.idpair[:] == i) & (data.day_flag[:] == j)
                min_index = np.where(condi)[0][0]
                max_index = np.where(condi)[0][-1]+1
                data.errII[data.errII>1000.0] = np.nan
                ax.scatter(data.div[min_index:max_index],
                           data.errI[min_index:max_index],
                           c=data.s2n[min_index:max_index],
                           cmap='plasma', vmin=0.0, vmax=5.0)
                ax.set_xlim(0.0,2.0)
                ax.set_ylim(0.0,2.0)
                ax2.scatter(data.shr[min_index:max_index],
                           data.errII[min_index:max_index],
                           c=data.s2n[min_index:max_index],
                           cmap='plasma', vmin=0.0, vmax=5.0)
                ax2.set_xlim(0.0,2.0)
                ax2.set_ylim(0.0,2.0)


        scatter_path  = self.figsPath + prefix + '_err_scatter.png'
        print("Printing error figure at : %s" % scatter_path)
        fig_scatter.savefig(scatter_path, bbox_inches='tight', dpi=600)
        plt.close(fig_scatter)


        return
