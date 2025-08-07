"""
Authors: Lekima Yakuden, Mathieu Plante, Amelie Bouchat, Damien Ringeisen
GitHub: LekiYak

--------------------------------------------------------------------------------
Tools for analysing and processing netCDF files
--------------------------------------------------------------------------------

This file contains functions for analysing and processing netCDF files.

"""

# Loading from default packages
import os
import sys
parent = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,parent)
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import copy
from multiprocessing import Pool, shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from math import floor, atan2
import cartopy.crs as ccrs
import pyproj as pyproj

# Loads netCDF data
class Coarsened_Data:

    def __init__(self, Data = None, tri_array= None, CDFout = None,config=None, idpair = None):
        """
        This function takes the triangulated arrays and compute the
        SIDRR for each triangle.

        INPUTS:

        config   = namelist object, from the python config parser object.

        """

        self.Data = copy.deepcopy(Data)

        self.sigx = Data.trackingerror
        sigx = self.sigx

#        start_lat = xr.DataArray(Data.start_lat.copy(), dims=['x'], name='start_lat',coords={"x": np.arange(a)})
#        start_lon = xr.DataArray(Data.start_lon.copy(), dims=['x'], name='start_lon',coords={"x": np.arange(a)})
#        end_lat = xr.DataArray(Data.end_lat.copy(), dims=['x'], name='end_lat',coords={"x": np.arange(a)})
#        end_lon = xr.DataArray(Data.end_lon.copy(), dims=['x'], name='end_lon',coords={"x": np.arange(a)})


    def compute_kernel_deformations(self, xr_in = None, xr_ref = None):

        if xr_ref is not None:
            Data_ref = np.array(xr_ref['div']).copy()
        else:
            Data_ref = self.Data.div.copy()*np.nan
        Nblocks = len(np.unique(self.Data.idpair))
        nb = 0
        for nblock in [38,39]:#np.unique(self.Data.idpair):
            nb = nb+1
            print("Doing pairs %s/%s" % (nb, Nblocks))

            condi = (self.Data.idpair== nblock)
            condi_pts = (self.Data.pts_idpair==nblock)

            dt = self.Data.end_time[condi]- self.Data.start_time[condi] 

            Data_pts = np.array([self.Data.start_lat[condi_pts].copy(), #0
                           self.Data.start_lon[condi_pts].copy(), #1
                           self.Data.end_lat[condi_pts].copy(), #2
                           self.Data.end_lon[condi_pts].copy(), #3
                           self.Data.pts_dayflag[condi_pts].copy(), #4
                           self.Data.pts_idpair[condi_pts].copy()]) #5
                           

            Data_shared = np.array([self.Data.div[condi].copy(), #0
                           self.Data.shr[condi].copy(), #1
                           self.Data.vrt[condi].copy(), #2
                           self.Data.A[condi].copy(), #3
                           self.Data.dudx[condi].copy(), #4
                           self.Data.dudy[condi].copy(), #5
                           self.Data.dvdx[condi].copy(), #6
                           self.Data.dvdy[condi].copy(), #7
                           self.Data.errA[condi].copy(), #8
                           self.Data.errI[condi].copy(), #9
                           self.Data.errII[condi].copy(), #10
                           self.Data.errvrt[condi].copy(), #11
                           self.Data.ids1[condi].copy(), #12
                           self.Data.ids2[condi].copy(), #13
                           self.Data.ids3[condi].copy(), #14
                           self.Data.idpair[condi].copy(), #15
                           self.Data.day_flag[condi].copy(), #16
                           self.Data.s2n[condi].copy(), #17
                           Data_ref[condi].copy()]) #18
            Data_in = Data_shared.copy()
            Data_shared = Data_shared*np.nan
            shm = shared_memory.SharedMemory(create=True, size=Data_shared.nbytes)
            shared_arr = np.ndarray(Data_shared.shape, dtype=Data_shared.dtype, buffer=shm.buf)
            np.copyto(shared_arr, Data_shared)

            args = [(j, Data_pts, shm.name, shared_arr.shape, dt[j], Data_in, self.sigx) for j in range(0, len(shared_arr[0,:]))]
            with ProcessPoolExecutor(max_workers=1) as executor:
                futures = [executor.submit(Compute_deformation_from_kernel, *arg) for arg in args]
                for future in as_completed(futures):
                    pass #future.result()  # handle exceptions if needed

            self.Data.div[condi] = shared_arr[0,:].copy()
            self.Data.shr[condi] = shared_arr[1,:].copy()
            self.Data.vrt[condi] = shared_arr[2,:].copy()
            self.Data.A[condi]   = shared_arr[3,:].copy()
            self.Data.dudx[condi] = shared_arr[4,:].copy()
            self.Data.dudy[condi] = shared_arr[5,:].copy()
            self.Data.dvdx[condi] = shared_arr[6,:].copy()
            self.Data.dvdy[condi] = shared_arr[7,:].copy()
            self.Data.errA[condi] = shared_arr[8,:].copy()
            self.Data.errI[condi] = shared_arr[9,:].copy()
            self.Data.errII[condi] = shared_arr[10,:].copy()
            self.Data.errvrt[condi] = shared_arr[11,:].copy()

            shm.close()
            shm.unlink()
            del(Data_shared)
            del(shared_arr)


        return self.Data


def Compute_deformation_from_kernel(j, Data_pts, shm_name, shape, dt, Data_in, sigx):

    shm = shared_memory.SharedMemory(name=shm_name)
    shared_results = np.ndarray(shape,  buffer=shm.buf)

    # Set the matplotlib projection and transform
    proj = ccrs.NorthPolarStereo(central_longitude=0)
    trans = ccrs.PlateCarree()
    if np.isnan(Data_in[0,j]):
        print("skiped row %s" % j, Data_in[0,j],np.isnan(Data_in[0,j]))
    else:
 
        #Extract data only from the pair ID and with right date
        #triangle vectors
        day_flag = Data_in[16,j].copy()
        idpair = Data_in[15,j].copy()
        condi = (Data_in[15,:] == idpair) & (Data_in[16,:] == day_flag)
        min_index = np.squeeze(np.where(condi))[0]
        max_index = np.squeeze(np.where(condi))[-1]+1
        i = j- min_index

        #vertices vectors
        condi_pts = (Data_pts[5,:] == idpair) & (Data_pts[4,:] == day_flag)
        start_lats = Data_pts[0,condi_pts]
        start_lons = Data_pts[1,condi_pts]
        end_lats = Data_pts[2,condi_pts]
        end_lons = Data_pts[3,condi_pts]

        # Get the data from the shared array in a xarray dataset
        a1 = len(Data_in[0,condi])
        div = xr.DataArray(Data_in[0,condi].copy(), dims=['x'], name='div',coords={"x": np.arange(a1)})
        shr = xr.DataArray(Data_in[1,condi].copy(), dims=['x'], name='shr',coords={"x": np.arange(a1)})
        vrt = xr.DataArray(Data_in[2,condi].copy(), dims=['x'], name='vrt',coords={"x": np.arange(a1)})
        A = xr.DataArray(Data_in[3,condi].copy(), dims=['x'], name='A',coords={"x": np.arange(a1)})
        dudx = xr.DataArray(Data_in[4,condi].copy(), dims=['x'], name='dudx',coords={"x": np.arange(a1)})
        dudy = xr.DataArray(Data_in[5,condi].copy().copy(), dims=['x'], name='dudy',coords={"x": np.arange(a1)})
        dvdx = xr.DataArray(Data_in[6,condi].copy().copy(), dims=['x'], name='dvdx',coords={"x": np.arange(a1)})
        dvdy = xr.DataArray(Data_in[7,condi].copy().copy(), dims=['x'], name='dvdy',coords={"x": np.arange(a1)})
        errA = xr.DataArray(Data_in[8,condi].copy(), dims=['x'], name='ids1',coords={"x": np.arange(a1)})
        erreI = xr.DataArray(Data_in[9,condi].copy(), dims=['x'], name='ids2',coords={"x": np.arange(a1)})
        erreII = xr.DataArray(Data_in[10,condi].copy(), dims=['x'], name='ids3',coords={"x": np.arange(a1)})
        errvrt = xr.DataArray(Data_in[11,condi].copy(), dims=['x'], name='ids3',coords={"x": np.arange(a1)})
        ids1 = xr.DataArray(Data_in[12,condi].copy(), dims=['x'], name='ids1',coords={"x": np.arange(a1)})
        ids2 = xr.DataArray(Data_in[13,condi].copy(), dims=['x'], name='ids2',coords={"x": np.arange(a1)})
        ids3 = xr.DataArray(Data_in[14,condi].copy(), dims=['x'], name='ids3',coords={"x": np.arange(a1)})
        s2n = xr.DataArray(Data_in[17,condi].copy(), dims=['x'], name='s2n',coords={"x": np.arange(a1)})
        ref = xr.DataArray(Data_in[18,condi].copy(), dims=['x'], name='ref',coords={"x": np.arange(a1)})

        xr_pair = xr.Dataset({'div': div,
                                  'shr':  shr,
                                  'vrt':  vrt,
                                  'A': A,
                                  'dudx': dudx,
                                  'dudy': dudy,
                                  'dvdx': dvdx,
                                  'dvdy': dvdy,
                                  'errA': errA,
                                  'erreI':erreI,
                                  'erreII':erreII,
                                  'errvrt':errvrt,
                                  'ids1':  ids1,
                                  'ids2':  ids2,
                                  'ids3':  ids3,
                                  's2n': s2n,
                                  'ref': ref})

        # Remove vertices with triangles that are Nan in the reference dataset
        xr_list = xr_pair.where(xr_pair['ref'].notnull(), drop = True)
        indices = np.concatenate([np.array(xr_list['ids1']),  np.array(xr_list['ids2']), np.array(xr_list['ids3'])])

        xr_pair = xr_pair.where(~xr_pair['ids1'].isin(indices), drop = True)
        xr_pair = xr_pair.where(~xr_pair['ids2'].isin(indices), drop = True)
        xr_pair = xr_pair.where(~xr_pair['ids3'].isin(indices), drop = True)

        # Get the lat-lon points of all vertices from the selected triangles
        i1,i2,i3 = int(Data_in[12,j]),int(Data_in[13,j]),int(Data_in[14,j])
        xr_tri = xr_pair.where(np.logical_and(np.logical_and(xr_pair['ids1'].isin([i1,i2,i3]), xr_pair['ids2'].isin([i1,i2,i3])),xr_pair['ids3'].isin([i1,i2,i3])), drop=True)

        #-----------------------------
        # Option B: Recompute deformations
        #-----------------------------

        indices = np.unique(np.concatenate([np.array(xr_tri['ids1']),  np.array(xr_tri['ids2']), np.array(xr_tri['ids3'])]))
        start_lats = start_lats[indices.astype(int)].copy()
        start_lons = start_lons[indices.astype(int)].copy()
        end_lats = end_lats[indices.astype(int)].copy()
        end_lons = end_lons[indices.astype(int)].copy()

        # Change to XY values, with origin on the cloud of pts centroid
        new_coords_start = proj.transform_points(trans, np.array(start_lons).copy(), np.array(start_lats).copy())
        new_coords_end = proj.transform_points(trans, np.array(end_lons).copy(), np.array(end_lats).copy())
        sx_list=new_coords_start[:,0].copy()-(np.sum(new_coords_start[:,0].copy())/len(new_coords_start[:,0]))
        ex_list=new_coords_end[:,0].copy()-(np.sum(new_coords_start[:,0].copy())/len(new_coords_start[:,0]))
        sy_list=new_coords_start[:,1].copy()-(np.sum(new_coords_start[:,1].copy())/len(new_coords_start[:,1]))
        ey_list=new_coords_end[:,1].copy()-(np.sum(new_coords_start[:,1].copy())/len(new_coords_start[:,1]))

        #Sort according to the angle with the centroid
        angles = np.arctan2(sy_list, sx_list)
        sorted_indices = np.argsort(angles)
        a = len(sx_list)
        u_list =  xr.DataArray(((ex_list[sorted_indices] - sx_list[sorted_indices]) / dt).copy(), dims=['x'], name='u_list',coords={"x": np.arange(a)})
        v_list =  xr.DataArray(((ey_list[sorted_indices] - sy_list[sorted_indices]) / dt).copy(), dims=['x'], name='v_list',coords={"x": np.arange(a)})
        sx_list =  xr.DataArray(sx_list[sorted_indices].copy(), dims=['x'], name='sx_list',coords={"x": np.arange(a)})
        sy_list =  xr.DataArray(sy_list[sorted_indices].copy(), dims=['x'], name='sy_list',coords={"x": np.arange(a)})
        ex_list =  xr.DataArray(ex_list[sorted_indices].copy(), dims=['x'], name='ex_list',coords={"x": np.arange(a)})
        ey_list =  xr.DataArray(ey_list[sorted_indices].copy(), dims=['x'], name='ey_list',coords={"x": np.arange(a)})

        xr_pts = xr.Dataset({'sx_list': sx_list,
                             'sy_list': sy_list,
                             'ex_list': ex_list,
                             'ey_list': ey_list,
                             'u_list':  u_list,
                             'v_list':  v_list})

        test = 11
        # Remove concave angles by computing the cross product of segments at each point
        while test > 10:
            test += 1
            ux = np.array(xr_pts['sx_list']).copy()
            vy = np.array(xr_pts['sy_list']).copy()
            uxm = np.array(xr_pts['sx_list']).copy()
            uxm[1:] = np.array(xr_pts['sx_list'])[:-1].copy()
            uxm[0] = np.array(xr_pts['sx_list'])[-1].copy()
            vym = np.array(xr_pts['sy_list']).copy()
            vym[1:] = np.array(xr_pts['sy_list'])[:-1].copy()
            vym[0] = np.array(xr_pts['sy_list'])[-1].copy()
            uxp = np.array(xr_pts['sx_list']).copy().copy()
            uxp[:-1] = np.array(xr_pts['sx_list'])[1:].copy()
            uxp[-1] = np.array(xr_pts['sx_list'])[0].copy()
            vyp = np.array(xr_pts['sy_list']).copy()
            vyp[:-1] = np.array(xr_pts['sy_list'])[1:].copy()
            vyp[-1] = np.array(xr_pts['sy_list'])[0].copy()
         
            u1 = ux - uxm
            u2 = uxp - ux
            v1 = vy - vym
            v2 = vyp-vy
            cross = (u1*v2)-(v1*u2)
            if len(cross[cross<0]) > 0:
                cross = xr.DataArray(cross.copy(), dims=['x'], name='cr',coords={'x': xr_pts.coords['x']})
                xr_pts = xr_pts.where(cross>0, drop = True)
            else:
                test = 0

        sx = np.array(xr_pts['sx_list']).copy()
        sy = np.array(xr_pts['sy_list']).copy()

        sxp = np.array(xr_pts['sx_list']).copy()
        sxp[:-1] = np.array(xr_pts['sx_list'])[1:].copy()
        sxp[-1] = np.array(xr_pts['sx_list'])[0].copy()

        syp = np.array(xr_pts['sy_list']).copy()
        syp[:-1] = np.array(xr_pts['sy_list'])[1:].copy()
        syp[-1] = np.array(xr_pts['sy_list'])[0].copy()

        u = np.array(xr_pts['u_list']).copy()
        v = np.array(xr_pts['v_list']).copy()
        um = np.array(xr_pts['u_list']).copy()
        um[1:] = np.array(xr_pts['u_list'])[:-1].copy()
        um[0] = np.array(xr_pts['u_list'])[-1].copy()
        vm = np.array(xr_pts['v_list']).copy()
        vm[1:] = np.array(xr_pts['v_list'])[:-1].copy()
        vm[0] = np.array(xr_pts['v_list'])[-1].copy()
        up = np.array(xr_pts['u_list']).copy().copy()
        up[:-1] = np.array(xr_pts['u_list'])[1:].copy()
        up[-1] = np.array(xr_pts['u_list'])[0].copy()
        vp = np.array(xr_pts['v_list']).copy()
        vp[:-1] = np.array(xr_pts['v_list'])[1:].copy()
        vp[-1] = np.array(xr_pts['v_list'])[0].copy()


        A = np.sum(( sx*syp - sxp*sy)/2.0)
        dudx = np.sum((up+u)*(syp-sy))/(2.0*A) 
        dudy = np.sum((up+u)*(sx-sxp))/(2.0*A)  #Note that sx-sxp = -Delta x here
        dvdx = np.sum((vp+v)*(syp-sy))/(2.0*A) 
        dvdy = np.sum((vp+v)*(sx-sxp))/(2.0*A) #Note that sx-sxp = -Delta x here

        eps_I = dudx + dvdy
        eps_II =  ((dudx - dvdy)**2 + (dudy + dvdx)**2)**0.5
        vrt = dvdx - dudy

        eps_tot = ( eps_I**2 + eps_II**2 )**0.5

        # Compute the propagation errors
        sxp2 = np.array(xr_pts['sx_list']).copy()
        sxp2[:-2] = np.array(xr_pts['sx_list'])[2:].copy()
        sxp2[-2] = np.array(xr_pts['sx_list'])[0].copy()
        sxp2[-1] = np.array(xr_pts['sx_list'])[1].copy()

        syp2 = np.array(xr_pts['sy_list']).copy()
        syp2[:-2] = np.array(xr_pts['sy_list'])[2:].copy()
        syp2[-2] = np.array(xr_pts['sy_list'])[0].copy()
        syp2[-1] = np.array(xr_pts['sy_list'])[1].copy()

        up2 = np.array(xr_pts['u_list']).copy()
        up2[:-2] = np.array(xr_pts['u_list'])[2:].copy()
        up2[-2] = np.array(xr_pts['u_list'])[0].copy()
        up2[-1] = np.array(xr_pts['u_list'])[1].copy()

        vp2 = np.array(xr_pts['v_list']).copy()
        vp2[:-2] = np.array(xr_pts['v_list'])[2:].copy()
        vp2[-2] = np.array(xr_pts['v_list'])[0].copy()
        vp2[-1] = np.array(xr_pts['v_list'])[1].copy()

        del11 = np.sum( (sy - syp2)**2.0/(2.0*A*A*dt*dt) + (u-up2)**2.0/(4.0*A*A) )
        del22 = np.sum( (sx - sxp2)**2.0/(2.0*A*A*dt*dt) + (v-vp2)**2.0/(4.0*A*A) )
        del12 = np.sum( (sx - sxp2)**2.0/(2.0*A*A*dt*dt) + (u-up2)**2.0/(4.0*A*A) )
        del21 = np.sum( (sy - syp2)**2.0/(2.0*A*A*dt*dt) + (v-vp2)**2.0/(4.0*A*A) )

        delA = np.sum( ((sx- sxp2)**2.0 + (syp2-sy)**2.0)/4.0 )
        
        del11 = dudx*dudx*delA/A**2.0 + del11*(sigx**2.0)
        del22 = dvdy*dvdy*delA/A**2.0 + del22*(sigx**2.0)
        del12 = dudy*dudy*delA/A**2.0 + del12*(sigx**2.0)
        del21 = dvdx*dvdx*delA/A**2.0 + del21*(sigx**2.0)
        delI = del11 + del22
        delvrt = del12 + del21
        term1 = (delI*(dudx - dvdy)**2.0 + (del12+del21)*(dudy + dvdx)**2.0) #this is eps_II^2 * delII^2
        term2 = eps_I**2.0 * delI
        epstot_deltot = term1 + term2

        if eps_II > 0.0:
            delII = (delI*(((dudx - dvdy)/eps_II)**2.0) + (del12+del21)*(((dudy + dvdx)/eps_II)**2.0))
            deltot = delII*((eps_II/eps_tot)**2.0) + delI*(( eps_I/eps_tot)**2.0)
            sig2noise = eps_tot**2.0 / epstot_deltot**0.5
        else:
            delII = 1000.0
            deltot = 1000.0
            sig2noise = 0.0

        #print("%s/%s" % (i,a1), j, len(xr_pts['sx_list']), eps_I,eps_II, vrt)
        shared_results[0,j] = eps_I*60*60*24
        shared_results[1,j] = eps_II*60*60*24
        shared_results[2,j] = vrt*60*60*24
        shared_results[3,j] = Atot
        shared_results[4,j] = dudx*60*60*24
        shared_results[5,j] = dudy*60*60*24
        shared_results[6,j] = dvdx*60*60*24
        shared_results[7,j] = dvdy*60*60*24
        shared_results[8,j] = delA**0.5
        shared_results[9,j] = (del11**0.5)*60*60*24
        shared_results[10,j] = (del12**0.5)*60*60*24
        shared_results[11,j] = (del21**0.5)*60*60*24
        shared_results[12,j] = (del22**0.5)*60*60*24


