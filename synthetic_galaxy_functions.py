#!/usr/bin/env python
# coding: utf-8


#import packages 

import sys
import os
import math
import logging
import galsim
import time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from scipy.ndimage.interpolation import rotate



#galsim script with minor adjustments 
def galaxy_gen(argv, params, output_file):
    """
    - Field
    - Galaxies are all bulge + disk
    - Galaxies are made with Sersic profiles (with diferent parameters)
    - psf is gaussian
    - Noise is poisson

    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("galaxies")

    # Define some parameters we'll use below.
    # Normally these would be read in from some parameter file.
    
    pixel_scale = params['pixel_scale']         # arcsec/pixel
    image_size = 2048                           # size of image in pixels
    image_size_arcsec = image_size*pixel_scale  # size of big image in each dimension (arcsec)
    nobj = params['nobj']                       # number of galaxies in entire field
                                                # (This corresponds to 8 galaxies / arcmin^2)
    center_ra = 19.3*galsim.hours               # The RA, Dec of the center of the image on the sky
    center_dec = -33.1*galsim.degrees
    

    # random_seed 
    random_seed = 24783923
    
    # filename
    filename = output_file
    
    
    file_name = os.path.join(filename + ".fits")
    
    logger.info('Starting first multiple galaxies fits')

    
    # Setup the image:
    full_image = galsim.ImageF(image_size, image_size)

    # The default convention for indexing an image is to follow the FITS standard where the
    # lower-left pixel is called (1,1).  However, this can be counter-intuitive to people more
    # used to C or python indexing, where indices start at 0.  It is possible to change the
    # coordinates of the lower-left pixel with the methods `setOrigin`.  For this demo, we
    # switch to 0-based indexing, so the lower-left pixel will be called (0,0).
    full_image.setOrigin(0,0)

    # As for demo10, we use random_seed for the random numbers required for the
    # whole image.  In this case, both the power spectrum realization and the noise on the
    # full image we apply later.
    rng = galsim.BaseDeviate(random_seed)


    # Make a slightly non-trivial WCS.  We'll use a slightly rotated coordinate system
    # and center it at the image center.
    theta = 0.17 * galsim.degrees
    
    # angle for random rotation of the galaxy 
    rand = galsim.UniformDeviate(random_seed+1)
    phi = rand() * 2.0 * np.pi * galsim.degrees
    
    # ( dudx  dudy ) = ( cos(theta)  -sin(theta) ) * pixel_scale
    # ( dvdx  dvdy )   ( sin(theta)   cos(theta) )
    # Aside: You can call numpy trig functions on Angle objects directly, rather than getting
    #        their values in radians first.  Or, if you prefer, you can write things like
    #        theta.sin() or theta.cos(), which are equivalent.
    dudx = np.cos(theta) * pixel_scale
    dudy = -np.sin(theta) * pixel_scale
    dvdx = np.sin(theta) * pixel_scale
    dvdy = np.cos(theta) * pixel_scale
    image_center = full_image.true_center
    affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=full_image.true_center)

    # We can also put it on the celestial sphere to give it a bit more realism.
    # The TAN projection takes a (u,v) coordinate system on a tangent plane and projects
    # that plane onto the sky using a given point as the tangent point.  The tangent
    # point should be given as a CelestialCoord.
    sky_center = galsim.CelestialCoord(ra=center_ra, dec=center_dec)

    # The third parameter, units, defaults to arcsec, but we make it explicit here.
    # It sets the angular units of the (u,v) intermediate coordinate system.
    wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
    full_image.wcs = wcs
    
    # Now we need to loop over our objects:
    for k in range(nobj):
        time1 = time.time()
        # The usual random number generator using a different seed for each galaxy.
        ud = galsim.UniformDeviate(random_seed+k+1)
        
        k = 23 #when working with single galaxies, use k to put it in the quadrant you want 
        if k in range(0,5):
            dec = center_dec + (3.5/10) * image_size_arcsec * galsim.arcsec
            ra = center_ra + (-3.5/10 + 1.75*k/10) * image_size_arcsec / np.cos(dec) * galsim.arcsec
        if k in range(5,10):
            dec = center_dec + (1.75/10) * image_size_arcsec * galsim.arcsec
            ra = center_ra + (-3.5/10 + 1.75*(k-5)/10) * image_size_arcsec / np.cos(dec) * galsim.arcsec
        if k in range(10,15):
            dec = center_dec + (0) * image_size_arcsec * galsim.arcsec
            ra = center_ra + (-3.5/10 + 1.75*(k-10)/10) * image_size_arcsec / np.cos(dec) * galsim.arcsec
        if k in range(15,20):
            dec = center_dec - (1.75/10) * image_size_arcsec * galsim.arcsec
            ra = center_ra + (-3.5/10 + 1.75*(k-15)/10) * image_size_arcsec / np.cos(dec) * galsim.arcsec
        if k in range(20,25):
            dec = center_dec - (3.5/10) * image_size_arcsec * galsim.arcsec
            ra = center_ra + (-3.5/10 + 1.75*(k-20)/10) * image_size_arcsec / np.cos(dec) * galsim.arcsec
            
        world_pos = galsim.CelestialCoord(ra,dec)

        # We will need the image position as well, so use the wcs to get that
        image_pos = wcs.toImage(world_pos)

        # We also need this in the tangent plane, which we call "world coordinates" here,
        # since the PowerSpectrum class is really defined on that plane, not in (ra,dec).
        uv_pos = affine.toWorld(image_pos)

        # We could just use `ud()<0.3` for this, but instead we introduce another Deviate type
        # available in GalSim that we haven't used yet: BinomialDeviate.
        # It takes an N and p value and returns integers according to a binomial distribution.
        # i.e. How many heads you get after N flips if each flip has a chance, p, of being heads.
        binom = galsim.BinomialDeviate(ud, N=1, p=0.3)
        real = binom()
        
        
        # galaxy 
        bulge_n = 4                           #sersic index #0.3 <= n <= 6.2
        bulge_re = params['bulge_effective']  #half light radius in arcsec
        
        disk_n = 1                            #sersic index 
        disk_r0 = params['disk_scale']        #scale radius in arcsec
        
        gal_q = params['inclination']         # (axis ratio 0 < q < 1) #inclination 
        gal_beta = params['sky_angle']        # degrees (position angle on the sky) 
    
        # Components individual fluxes
        # ADU  ("Analog-to-digital units", the units of the numbers on a CCD) 
        bd_ratio = params['bd_ratio']
        disk_flux = params['disk_flux']  
        bulge_flux = disk_flux * bd_ratio  
    
        # psf
        psf_flux = 1.0                    # PSF flux should always = 1
        psf_fwhm = params['psf_fwhm']

        # AGN 
        AGN_flux = params['agn_flux']
        
        # lensing
        g1 = 0.1                                                                                                                                       
        g2 = 0.1 
        mu = 1.0
        
        # Define the galaxy profile.
        # Normally Sersic profiles are specified by half-light radius, the radius that
        # encloses half of the total flux.  However, for some purposes, it can be
        # preferable to instead specify the scale radius, where the surface brightness
        # drops to 1/e of the central peak value.
        
        bulge = galsim.Sersic(bulge_n, flux = bulge_flux, half_light_radius=bulge_re)
        disk = galsim.Sersic(disk_n, flux = disk_flux, scale_radius=disk_r0)
    
        # Set the overall flux of the combined object.
        gal = galsim.Add([bulge, disk])
    
        gal_shape = galsim.Shear(q=gal_q, beta=gal_beta*galsim.degrees)
        gal = gal.shear(gal_shape)
        logger.debug('Made galaxy profile')

        # Apply a random rotation (same random for all galaxies)
        gal = gal.rotate(phi)   

        # Rescale the flux to match our telescope configuration.
        # This automatically scales up the noise variance by flux_scaling**2.
        #gal *= flux_scaling

        # Apply the cosmological (reduced) shear and magnification at this position using a single
        # GSObject method.
        gal = gal.lens(g1, g2, mu)
        
        # Define the PSF profile
        psf = galsim.Gaussian(fwhm=psf_fwhm)      
        logger.debug('Made PSF profile')

        # Define the AGN profile
        #AGN = galsim.Gaussian(flux=AGN_flux, sigma=AGN_sigma)
        AGN = galsim.DeltaFunction(flux=AGN_flux) 
        
        # Convolve psf and galaxy
        final_convolve = galsim.Convolve(psf, gal)
        
        # Adding the AGN
        final = galsim.Add(AGN, final_convolve)
        
        # Adding the AGN
        final_add = galsim.Add(AGN, gal)
        
        # Convolve with the psfp
        final = galsim.Convolve(psf, final_add)
        #final = galsim.Convolve(psf, gal)

        
        # Account for the fractional part of the position
        # cf. demo9.py for an explanation of this nominal position stuff.
        x_nominal = image_pos.x +0.5
        y_nominal = image_pos.y +0.5
        ix_nominal = int(math.floor(x_nominal +0.5))
        iy_nominal = int(math.floor(y_nominal +0.5))
        dx = x_nominal - ix_nominal
        dy = y_nominal - iy_nominal
        offset = galsim.PositionD(dx,dy)

        # stamp
        stamp = final.drawImage(wcs=wcs.local(image_pos), offset=offset)

        # Recenter the stamp at the desired position:
        stamp.setCenter(ix_nominal,iy_nominal)

        # Find the overlapping bounds:
        bounds = stamp.bounds & full_image.bounds
    
        # Finally, add the stamp to the full image.
        full_image[bounds] += stamp[bounds]

        time2 = time.time()
        tot_time = time2-time1
        logger.info('Galaxy %d: position relative to center = %s, t=%f s',
                    k, str(uv_pos), tot_time)
                          
    # Noise. We have to do this step at the end, rather than adding to individual postage stamps, 
    #in order to get the noise level right in the overlap regions between postage stamps.    
    sky_level_pixel = 0
    noise = galsim.PoissonNoise(rng,sky_level=sky_level_pixel)
    full_image.addNoise(noise) #come back to this
    logger.info('Added noise to final large image')

    # Now write the image to disk.  
    full_image.write(file_name)
    logger.info('Wrote image to %r',file_name)

    # Compute some sky positions of some of the pixels to compare with the values of RA, Dec
    # that ds9 reports.  ds9 always uses (1,1) for the lower left pixel, so the pixel coordinates
    # of these pixels are different by 1, but you can check that the RA and Dec values are
    # the same as what GalSim calculates.
    ra_str = center_ra.hms()
    dec_str = center_dec.dms()
    logger.info('Center of image    is at RA %sh %sm %ss, DEC %sd %sm %ss',
                ra_str[0:3], ra_str[3:5], ra_str[5:], dec_str[0:3], dec_str[3:5], dec_str[5:])
    for (x,y) in [ (0,0), (0,image_size-1), (image_size-1,0), (image_size-1,image_size-1) ]:
        world_pos = wcs.toWorld(galsim.PositionD(x,y))
        ra_str = world_pos.ra.hms()
        dec_str = world_pos.dec.dms()
        logger.info('Pixel (%4d, %4d) is at RA %sh %sm %ss, DEC %sd %sm %ss',x,y,
                    ra_str[0:3], ra_str[3:5], ra_str[5:], dec_str[0:3], dec_str[3:5], dec_str[5:])
    logger.info('ds9 reports these pixels as (1,1), (1,2048), etc. with the same RA, Dec.')
    

    

# take synthetic galaxy generated in galaxy_gen() function and add it to real image
def add_galaxy(synthetic, background, x, y):
    
    #open synthetic galaxy file
    synth_gal = fits.open(synthetic)

    #extract data from files
    bkg_data = background[1].data
    header = background[1].header
    gal = synth_gal[0].data
    coords = WCS(header)

    #make synthetic galaxy match background shape
    empty = np.zeros((2048,2048))
    tgal = np.append(empty, gal, axis=0)
    
    #translate sky coords into pixel coordinates
    pix, piy = coords.all_world2pix(x,y,1)

    #get simulated galaxy position from galsim script
    sx = 662.9743900 
    sy = 307.7052556 + 2048

    #mask out sources blocked by galaxy 
    cap = np.max(tgal) - 10
    mask = (tgal>cap)
    bkg_data[mask] = 0 

    #put the synthetic galaxy into the background image
    inject = tgal + bkg_data
    
    return inject




# takes window of only the real galaxy we're comparing and synthetic galaxy 
def galaxy_cutout(x, y, synthetic, background):
    
    #open synthetic galaxy file
    synth_gal = fits.open(synthetic)

    #extract data from files
    bkg_data = background[1].data
    header = background[1].header
    gal = synth_gal[0].data
    coords = WCS(header)

    #make synthetic galaxy match background shape
    empty = np.zeros((2048,2048))
    tgal = np.append(empty, gal, axis=0)
    
    #translate sky coords into pixel coordinates
    pix, piy = coords.all_world2pix(x,y,1)

    #get simulated galaxy position from galsim script
    sx = 662.9743900 
    sy = 307.7052556 + 2048

    #mask out sources blocked by galaxy 
    cap = np.max(tgal) - 10
    mask = (tgal>cap)
    bkg_data[mask] = 0 

    #put the synthetic galaxy into the background image
    inject = tgal + bkg_data
    
    #translate sky coords into pixel coordinates
    pix, piy = coords.all_world2pix(x,y,1)

    #get simulated galaxy position from galsim script
    sx = 662.9743900 
    sy = 307.7052556 + 2048
    
    
    #create window of only galaxy we're interested in
    window = 100
    
    #real galaxy cutout
    lowx = int(pix) - window
    highx = int(pix) + window

    lowy = int(piy) - window
    highy = int(piy) + window
    
    ogal = background[1].data
    ogal = ogal[lowy:highy,lowx:highx]

    #synthetic galaxy cutout
    slx = int(sx) - window
    shx = int(sx) + window

    sly = int(sy) - window
    shy = int(sy) + window

    isynth = inject[sly:shy,slx:shx]
    
    return isynth, ogal




# creates radial profile of major and minor axis of both galaxies and compares them
def radial_profile(x, y, angle, inject, background):
    
    header = background[1].header
    coords = WCS(header)
    
    #translate sky coords into pixel coordinates
    pix, piy = coords.all_world2pix(x,y,1)
    
    #get simulated galaxy position from galsim script
    sx = 662.9743900 
    sy = 307.7052556 + 2048
    
    #create window of only galaxy we're interested in
    window = 100
    
    #real galaxy cutout
    lowx = int(pix) - window
    highx = int(pix) + window

    lowy = int(piy) - window
    highy = int(piy) + window
    
    ogal = background[1].data
    ogal = ogal[lowy:highy,lowx:highx]

    #synthetic galaxy cutout
    slx = int(sx) - window
    shx = int(sx) + window

    sly = int(sy) - window
    shy = int(sy) + window

    isynth = inject[sly:shy,slx:shx]

    #rotate galaxies to be vertically oriented
    rot_synth = rotate(isynth, angle)
    rot_ogal = rotate(ogal, angle)

    #take slices along major and minor axes for both original and synthetic galaxies
    mino, majo = np.where(rot_ogal == np.max(rot_ogal)) 
    mins, majs = np.where(rot_synth == np.max(rot_synth))

    # plot radial profiles
    plt.plot(rot_ogal[mino[0],mino[0]-100:mino[0]+100], label='real')
    plt.plot(rot_synth[mins[0],mins[0]-100:mins[0]+100], label='synthetic', color='green')
    plt.title('Minor axis')
    plt.legend()
    plt.show()

    plt.plot(rot_ogal[majo[0]-100:majo[0]+100,majo[0]],label='real')
    plt.plot(rot_synth[majs[0]-100:majs[0]+100,majs[0]], label='synthetic', color='green')
    plt.title('Major axis')
    plt.legend()
    plt.show()
    
    return rot_ogal, rot_synth




# creates a range of flux values
def agn_flux(N,percent):
    low = N * (1 - percent)
    high = N * (1 + percent)
    return np.linspace(low,high,10)




# gets a very rough estimate of galaxy parameters
def galaxy_specs(ra,dec,dradius,bradius, header):
    coords = WCS(header)
    pix, piy = coords.all_world2pix(ra,dec,1)
    
    half = 0.5
    scale = 1/(np.exp(1))

    #turn arcsec into pixel
    disk_r = dradius / .27
    bulge_r = bradius / .27

    #calculate flux
    dflux, dfluxerr, dflag = sep.sum_circle(data, [pix], [piy], disk_r)
    bflux, bfluxerr, bflag = sep.sum_circle(data, [pix], [piy], bulge_r)

    #radius in pixels
    bulge_half, flag = sep.flux_radius(data, [pix], [piy], [bulge_r], half, 
                              normflux=bflux, subpix=5)
    disk_scale, flag = sep.flux_radius(data, [pix], [piy], [disk_r], scale,
                              normflux=dflux, subpix=5)
    disk_half, flag = sep.flux_radius(data, [pix], [piy], [disk_r], half, normflux=dflux, subpix=5)

    #radius in arcsecs
    br_half = bulge_half * 0.27
    dr_scale = disk_scale * 0.27
    dr_half = disk_half * .27     
    
    #disk concentration index
    dr_80, flag = sep.flux_radius(data, [pix], [piy], [disk_r], .8, normflux=dflux, subpix=5)
    dr_20, flag = sep.flux_radius(data, [pix], [piy], [disk_r], .2, normflux=dflux, subpix=5)
    dc_index = 5*np.log10(dr_80/dr_20) 
       
    #bulge concentration index
    br_80, flag = sep.flux_radius(data, [pix], [piy], [bulge_r], .8, normflux=bflux, subpix=5)
    br_20, flag = sep.flux_radius(data, [pix], [piy], [bulge_r], .2, normflux=bflux, subpix=5)
    bc_index = 5*np.log10(br_80/br_20)    
       
    #all parameters in a dictionary
    galaxy_specs = dict(bulge_flux = bflux[0], disk_flux = dflux[0] - bflux[0], bulge_half_r = br_half[0],                             disk_scale_r = dr_scale[0], bulge_concentration = bc_index, disk_concentration = dc_index)

    return galaxy_specs


