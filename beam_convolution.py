import numpy as np
import healpy as hp
import pickle
import os
import numpy.ma as ma
import yaml
import lmfit
from lmfit import Model, Parameters
import scipy
from scipy.special import jv
from scipy.interpolate import interp1d
import utils as ut
import astropy.units as u
import warnings
import noise_calc as nc 
import pysm
opj = os.path.join

def get_freqs(band, band_width=0.2, nsamp_freqs=5):
    """
    Returns the 'nsamp_freqs' sub-frequencies of a band around
    a center value 'band_c' with bandwidth 'band_width'. 
    """    
    band_c = {'LF1':27, 'LF2':39, 'MF1':93, 'MF2':145, 'UHF1':225, 'UHF2':280}
    freqs = np.linspace(band_c[band]-(band_width*band_c[band]/2),
                        band_c[band]+(band_width*band_c[band]/2), 
                        nsamp_freqs)
    return freqs

def frequency_scale(maps_in, band, beta, temp, comp, sed_type='plaw',
                    nu_c='band_centers',freqs=None):
    """
    Function that scales the sub-frequencies of a band by the sky component SED

    Args
    ----
    maps_in: Array (len(sub_freqs),3,hp.nside2npix(nside)) of sub-frequency 
             maps to be scaled
    beta: The spectral index of the SED of the sky component

    Output
    ------
    maps_scaled_band: Array (3,hp.nside2npix(nside)) of frequency scaled 
                      maps of the frequency band
    """
    band_c = {'LF1':40.125, 'LF2':40.125, 'MF1':140., 'MF2':140., 'UHF1':250., 'UHF2':250.}
    bpass_c = {'LF1':27, 'LF2':39, 'MF1':93, 'MF2':145, 'UHF1':225, 'UHF2':280}
    fg_c = {'dust':353, 'synchrotron':23, 'cmb':None}

    if nu_c=='band_centers':
        nu0 = band_c[band]
    elif nu_c=='bpass_centers':
        nu0 = bpass_c[band]
    elif nu_c=='fg_centers':
        nu0 = fg_c[comp]
    else:
        raise ValueError("You must specify the reference frequency to be\
                           used for scaling the maps")

    maps_scaled_sub = np.zeros_like(maps_in)

    if sed_type=='comp_sed':
        scale_T = [ut.comp_sed(freq,nu0,beta,temp,comp) for freq in freqs]
    elif sed_type=='plaw':
        scale_T = (freqs/nu0)**beta
    for freq_i in range(np.shape(maps_in)[0]):
        maps_scaled_sub[freq_i,:] = maps_in[freq_i,:]*scale_T[freq_i]
        
    maps_scaled_band = maps_scaled_sub.sum(axis=0)*((freqs[-1]-freqs[0])/len(freqs))

    return maps_scaled_band

def find_nearest(array, value):
    """
    Return the index of the array element
    closer to the desired value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def airy_disc(x, y, amp, x0, y0, R, theta):
    """Define an Airy pattern model for the beam"""
    
    Rz = 1.2196698912665045 
    r = np.sqrt((x-x0)**2+(y-y0)**2)
    z = np.pi*r/(R/Rz)
    j1 = scipy.special.jv(1, z)
    airy = (2 * j1/z)**2
    return amp*airy

def gaussian_2d_rot(x, y, x0, y0, sigx, sigy, theta, **kwargs):
    """2D Gaussian function including rotation. Inputs given in radians
    
    Args
    ----
    x0, y0: float centers of the Gaussian
    sigx, sigy: sigma of the distribution in x,y directions
    theta: orientation angle
    """

    try:
        amp = kwargs['amp']
    except:
        amp = 1 
    sigx2 = sigx**2
    sigy2 = sigy**2
    x_rot = np.cos(theta)**2/(2*sigx2) + np.sin(theta)**2/(2*sigy2)
    y_rot = np.sin(theta)**2/(2*sigx2) + np.cos(theta)**2/(2*sigy2)
    xy_rot = np.sin(2*theta)/(4*sigx2) - np.sin(2*theta)/(4*sigy2)
    
    d2_rot_gauss = -x_rot*(x-x0)**2 - y_rot*(y-y0)**2 - 2*xy_rot*(x-x0)*(y-y0) 
    return amp*np.exp(d2_rot_gauss) 

def make_model_params(dependent_params, b_acc):
    """Make the lmfit parameter object for the fitting function"""
    
    params=Parameters()
    for idx_key, key in enumerate(dependent_params.keys()):
        key_value = dependent_params[key]
        params.add(key, value=key_value, 
                        min=-b_acc*key_value+key_value, 
                        max=b_acc*key_value+key_value)
    return params


def mask_source(data, x, y, r_mask):
    """Mask a region around the source or radius r_mask and gap-fill"""
    
    mask = np.zeros_like(data)
    mask[np.where(np.sqrt(x**2+y**2) < r_mask)]=1
    
    return mask*data

def fit_main_lobe(data, x, y, res, init_params, n_iter, acc,
                  return_all=False):
    """Fit an Airy pattern to find the beam's first minimum,
       mask the sidelobes and then fit a 2D rotated Gaussian

    Args
    ----
    data: Array of npix x npix to be fitted
    res: the size of the radial bins
    init_params: initial guess for the fitted parameters
    n_iter: number of iterations
    acc: desired accuracy of the fit

    Returns
    -------
    Dictionary of the fitted parameters/ or only the best-fit 
    beam_size, in arcmins, and ellipticity
    """
    data /= np.nanmax(data)
    amp, x0, y0, R, theta = [init_params[key] for key in ['amp',
                                                          'x0',
                                                          'y0',
                                                          'R',
                                                          'theta']]                         
    error = np.sqrt(data.ravel()+1)

    # Roughly the dark ring is FWHM/0.8
    fmodel = Model(airy_disc, independent_vars=('x','y'))
    params = make_model_params(dependent_params={'amp':amp, 
                                                 'x0':x0, 
                                                 'y0':y0, 
                                                 'R':R,
                                                 'theta':theta}, 
                                                 b_acc=2*acc)
    result_a = fmodel.fit(data=data.ravel(), 
                          x=x.ravel(), 
                          y=y.ravel(), 
                          params=params, 
                          weights=1/error,
                          max_nfev=n_iter)

    amp, x0, y0, R, theta = result_a.best_values.values()

    # Define a 10% region around the first dark ring 
    bins, prof = radial_profile(x.ravel(), y.ravel(), data.ravel(), center = np.array([0,0]),
                                binsize=res, return_angle=True)
    bins_pos, prof_pos = bins[np.where(prof>=0)], prof[np.where(prof>=0)]

    idx_min, idx_max = find_nearest(bins, R-0.1*R), find_nearest(bins, R+0.1*R)
    trunc_prof = prof[idx_min:idx_max]

    ### find the minimum of positive values in -- sometimes filtering creates two negative bumps around the peak
    first_zero_touch = idx_min+np.where(np.logical_and(trunc_prof>0, trunc_prof==np.min(trunc_prof)))[0][0]
    ### interpolate over negative points
    sp1d = interp1d(bins_pos, prof_pos)
    prof[:idx_max] = sp1d(bins[:idx_max])

    # Mask after the first minimum to isolate main lobe
    rmask = bins[first_zero_touch]
    data_masked = mask_source(data, x, y, rmask)   
    prof_masked, bins_masked = radial_profile(x.ravel(), y.ravel(), data_masked.ravel(), center = np.array([0,0]),
                                              binsize=res, return_angle=True)

    # Fit a Gaussian to the masked map
    fact = np.sqrt(8*np.log(2))
    sigx, sigy = init_params['fwhm_x']/fact, init_params['fwhm_y']/fact 
    fmodel = Model(gaussian_2d_rot, independent_vars=('x','y'))

    params = make_model_params(dependent_params={'x0':x0, 
                                                 'y0':y0, 
                                                 'sigx':sigx, 
                                                 'sigy':sigy, 
                                                 'theta':theta}, 
                                                  b_acc=acc)

    result_g = fmodel.fit(data_masked.ravel(), 
                          x=x.ravel(), 
                          y=y.ravel(), 
                          params=params, 
                          weights=1/error, 
                          max_nfev=n_iter)  

    fitted_values_g = result_g.best_values  

    # Store the best-fit values 
    fwhm_x, fwhm_y = fitted_values_g['sigx']*fact, fitted_values_g['sigy']*fact
    fwhm = np.sqrt(8*np.log(2)*fitted_values_g['sigx']*fitted_values_g['sigy'])
    ell = np.abs(fwhm_x-fwhm_y)/(fwhm_x+fwhm_y)

    fitted_values_cp = {'fwhm_x': np.degrees(fwhm_x)*60, 
                        'fwhm_y': np.degrees(fwhm_y)*60,
                        'fwhm': np.degrees(fwhm)*60,
                        'theta': theta,
                        'amp':amp,
                        'ell':ell,
                        'R': R,
                        }
    if not return_all:
        return fitted_values_cp['fwhm'], fitted_values_cp['ell']
    else:
        return fitted_values_cp

def get_input_params(INITIAL_PARA_FILE, tele, band):
    """Get the initial parameters configuration for
    telescope 'tele' and frequency band 'band' 
    """

    if tele != "LAT":
        import re

        tele = re.split("(\d+)", tele)[0]
    with open(INITIAL_PARA_FILE, "r") as stream:
        try:
            config_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    band_idx = config_file['telescopes'][tele]['bands'].index(band)
    beamsize, band_c, d = config_file['telescopes'][tele]['beamsize'][band_idx], config_file['telescopes'][tele]['band_c'][band_idx],config_file['telescopes'][tele]['aperture']
    wlength = 3 / (band_c)
    wlength *= 1e-01
    R = 1.22*wlength/d
    init_params={'amp':1, 
                 'x0':1e-05, 
                 'y0':1e-05, 
                 'fwhm_x':np.radians(beamsize/60), 
                 'fwhm_y':np.radians(beamsize/60), 
                 'theta':1e-06,
                 'R': R}
    return init_params

def radial_profile(xi, eta, signal,
    center=np.array([0,0]),
    mask=None,
    return_angle=False,
    return_nr=False,
    binsize=0.5,
    weights=None,
    steps=False,
    interpnan=False,
    stddev=False,
    left=None,
    right=None):

    '''
    Calculate the azimuthally averaged radial profile.
    '''

    if center is None:
        center = np.array([0., 0.])

    if mask is None:
        mask = np.ones(signal.shape, dtype='bool')

    if weights is None:
        weights = np.ones(signal.shape)

    r = np.sqrt((xi - center[0])**2 + (eta - center[1])**2)
    try:
        nbins = int(np.round(r.max() / binsize)+1)
    except:
        return [0,0]
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    nr = np.histogram(r, bins, weights=mask.astype('int'))[0]
    
    if stddev:
        # Find out which radial bin each point in the map belongs to
        whichbin = np.digitize(r.flat, bins)
        # This method is still very slow; is there a trick to do this with histograms?
        radial_prof = np.array([signal.flat[mask.flat*(whichbin==b)].std() for b in range(1,nbins+1)])

    else:
        radial_prof = np.histogram(r, bins, weights=(signal*weights*mask))[0] \
            / np.histogram(r, bins, weights=(mask*weights))[0]

    if interpnan:
        radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof], \
            radial_prof[radial_prof==radial_prof], left=left, right=right)
        
    if steps:
        xarr = np.array(zip(bins[:-1],bins[1:])).ravel()
        yarr = np.array(zip(radial_prof,radial_prof)).ravel()
        return xarr, yarr


    elif return_angle:
        return bin_centers, radial_prof
    elif return_nr:
        return nr, bin_centers, radial_prof
    else:
        return radial_prof

def equiang2hp(bmap, delta_az, delta_el, nside_out, return_thetamax=False,
               apodize=False):
    """
    Simplified version of the corresponding hwp_sims function to convert 
    a grasp beam array to healpy map

    Args
    ----
    bmap: The beam map file
    delta_az, delta_el: The assumed azimuth and elevation range
    nside_out: The healpix map resolution
    return_thetamax: If True, return the maximum angle of the map
    apodize: If True, apodize the map with a Gaussian kernel

    Output
    ------
    bmap_out: The healpix map
    """
    naz, nel = bmap.shape

    delta_az = np.radians(delta_az)
    delta_el = np.radians(delta_el)

    az_arr = np.linspace(-delta_az / 2., delta_az / 2., naz)
    el_arr = np.linspace(-delta_el / 2., delta_el / 2., nel)

    bmap_out = np.zeros(12 * nside_out**2, dtype=bmap.dtype)

    theta, phi = hp.pix2ang(nside_out, np.arange(bmap_out.size))
    phi += np.pi
    az = np.arctan(np.cos(phi) * np.tan(theta))
    el = np.arcsin(np.sin(phi) * np.sin(theta))

    az_ind = np.digitize(az, az_arr, right=True)
    el_ind = np.digitize(el, el_arr, right=True)

    #fixing edge cases
    az_ind[az_ind == naz] = naz - 1
    el_ind[el_ind == nel] = nel - 1

    bmap_out[:] = bmap[az_ind,el_ind]

    max_az = np.max(az_arr)
    max_el = np.max(el_arr)

    max_rad = min(max_az, max_el)

    if apodize:
        # apodize outer edge with Gaussian
        mu = 0.8 * max_rad
        std = 0.05 * max_rad

        annulus = np.where(np.logical_and(theta<max_rad, theta>0.8*max_rad))
        gauss = np.exp(-0.5*(theta-mu)**2 / std**2)
        bmap_out[annulus] *= gauss[annulus]

    bmap_out[theta>max_rad] = 0

    if return_thetamax:
        return bmap_out, max_rad

    return bmap_out

def e2iqu(fields, nside_out, **kwargs):
    """
    Simplified version of the corresponding hwp_sims function to convert 
    the co- and cross-polar grasp fields to Stokes I,Q,U maps

    Args
    ----
    fields: .pkl file containing the co- and cross-polar grasp fields

    Output
    ------
    [I,Q,U]: Tuple of healpix Stokes maps
    """
  
    if kwargs.get('apodize'):
       apodize = kwargs['apodize']
    else:
       apodize = False

    e_co = fields['e_cx']
    e_cross = fields['e_co']

    cr = fields['cr'] # [azmin, elmin, azmax, elmax]
    delta_az = cr[2] - cr[0]
    delta_el = cr[3] - cr[1]
    
    e_co = equiang2hp(e_co, delta_az, delta_el, nside_out=nside_out,
                        apodize=apodize)
    e_cross = equiang2hp(e_cross, delta_az, delta_el, nside_out=nside_out,
                        apodize=apodize)

    nside = hp.get_nside(e_co)

    # squared combinations of fields
    e_co2 = np.abs(e_co)**2
    e_cross2 = np.abs(e_cross)**2
    e_cocr = e_co * np.conj(e_cross)

    # create Stokes parameters
    I = e_co2 + e_cross2
    Q = (e_co2 - e_cross2)
    U = -2 * np.real(e_cocr)

    return [I, Q, U]


def get_beam_healpix_maps(band_name, nside, do_scaling, comp=None, **kwargs):
    """
    Get the beam stokes I,Q,U maps given the sub frequencies and nside

    Args
    ----
    do_scaling: If True, frequency-scale the beam by the 
                sky component's SED. This scaling is applied
                to the sub_frequency beam maps which are after
                synthesized to the frequency band beam. If not
                simply average the sub frequency maps.
    """

    beam_path = kwargs['beam_path']

    # for key in healpix_keys:
    #     exec('{KEY} = {VALUE}'.format(KEY = key, VALUE = repr(kwargs[key])))
    #     kwargs.pop(key)
    sub_freqs = get_freqs(band_name)

    for freq_idx, freq_i in enumerate(sub_freqs):

        freq_n = str(freq_i).split('.')  
        healpix_beam_map = opj(beam_path, 'beam_fullsky_stokes_maps_nside'+str(nside)+\
                                         '_band'+band_name+'_pixel0cm_unscaled', 
                                         'map_'+str(nside)+'_'+str(freq_i)+'_GHz.fits')
        
        if not os.path.exists(healpix_beam_map.rsplit('/',1)[0]):
            os.makedirs(healpix_beam_map.rsplit('/',1)[0])

        try:
            stokes_sub = hp.read_map(healpix_beam_map, field=(0,1,2))
        except:
            warnings.warn('Converting pickle files to beam maps, might take time')
            beam_name = 'satpixel'+"{:03d}".format(int(freq_n[0]))+'pt'\
                        +str(freq_n[1])+'GHz_00pt00deg_fields.pkl'
            fields = pickle.load(open(opj(beam_path, 'grasp_files', beam_name),'rb'), 
                                encoding='latin1')
            stokes_sub = e2iqu(fields, nside_out=nside, **kwargs)
            hp.write_map(healpix_beam_map, stokes_sub)

        if freq_idx==0:
            stokes_matrix = np.zeros((len(sub_freqs), 3, hp.nside2npix(nside)))
        stokes_matrix[freq_idx,:,:] = stokes_sub

    if do_scaling == False:
        band_beam = stokes_matrix.sum(axis=0)*((sub_freqs[-1]-sub_freqs[0])/len(sub_freqs))
    else:
        beta = kwargs['betas'][comp]
        temp = kwargs['temps'][comp]
        sed_type = kwargs['sed_type']
        nu_c = kwargs['nu_c']
 
        band_beam= frequency_scale(stokes_matrix, 
                                 band_name, 
                                 beta,
                                 temp,
                                 comp,
                                 sed_type,
                                 nu_c,
                                 sub_freqs)

    return band_beam

def get_beam_fwhm(band_name, nside, do_scaling, comp=None, **kwargs):
    """
    Function that returns the beam FWHM of a frequency band either 
    from retrieving it or finding its best-fit value

    Args
    ----
    **lon_rot, **lat_rot: rotate the beam (by default is placed on
                          the North pole of healpix map)
    **lon_crop, **lat_crop: tuples of min/max longitude and latitude
                            of patch that fully contains the beam.

    Output
    ------
    fwhm: The retrieved/best-fit fwhm of the beam in arcmins.
    """

    fwhm = kwargs['bms_fwhm'][band_name]
    print('fwhm=',fwhm, fwhm is None) 
    if fwhm is None:
        fitting_keys = ['initial_param_file', 'tele', 'res', 'n_iter', 'acc']
        initial_param_file = kwargs['initial_param_file']
        tele = kwargs['tele']
        res = kwargs['res']
        n_iter = kwargs['n_iter']
        acc = kwargs['acc']

        for key in fitting_keys:
            # exec('{KEY} = {VALUE}'.format(KEY = key, VALUE = repr(kwargs[key])))
            kwargs.pop(key)

        band_beam = get_beam_healpix_maps(band_name=band_name, nside=nside, 
                                          do_scaling=do_scaling, comp=comp,
                                          **kwargs)

        stokes_sub = [hp.cartview(band_beam[i], 
                                  return_projected_map=True, 
                                  lonra=kwargs['lon_crop'], 
                                  latra=kwargs['lat_crop'],
                                  rot=[kwargs['lon_rot'], kwargs['lat_rot']])
                                      for i in range (np.shape(band_beam)[0])] 

        dim1, dim2 = np.shape(stokes_sub)[1], np.shape(stokes_sub)[2]
        x = np.linspace(kwargs['lon_crop'][0], kwargs['lon_crop'][1], dim1) 
        y = np.linspace(kwargs['lat_crop'][0], kwargs['lat_crop'][1], dim2)
        X, Y = np.meshgrid(x,y)
        X_rad, Y_rad = np.radians(X), np.radians(Y)

        init_params = get_input_params(initial_param_file, 
                                       tele, 
                                       band_name)
        
        fwhm, ell = fit_main_lobe(stokes_sub[0], 
                                  X_rad, 
                                  Y_rad, 
                                  res, 
                                  init_params,
                                  n_iter, 
                                  acc)

    return fwhm

def get_gauss_blms(bms_fwhm, ell):
    """
    Function that returns Gaussian beam harmonic transforms

    Args
    ----
    bms_fwhm: The FWHM of the beam
    ell: The multipole range (0, lmax+1)
    """

    sig = bms_fwhm / np.sqrt(8. * np.log(2)) /60. * np.pi/180.

    return np.exp(-0.5*ell*(ell+1)*sig**2.)

def get_beam_blms(band_name, nside, lmax, blm_mmax, do_scaling, beam_type, pol, 
                  comp=None, **kwargs):
    """
    Function that returns the beam harmonic coefficients for a frequency band

    Args
    ----
    beam_type: If 'Gaussian', the function returns the Gaussian beam's harmonic
               transform given retrieved or best-fit FWHM.
               If 'PO', the harmonic transform of the PO beam files is returned
               instead.
    **outdir: Path to save the beam harmonic transform if desired. If outdir is 
              None the beam transform will not be stored.
    """
    if lmax is None:
       lmax = 3*nside-1

    if beam_type == 'Gaussian':
        bms_fwhm = get_beam_fwhm(band_name=band_name, nside=nside,
                                 do_scaling=do_scaling, comp=comp,
                                 **kwargs)
        blms_from_fwhm = get_gauss_blms(bms_fwhm, np.arange(lmax))
        blms_from_fwhm /= np.nanmax(blms_from_fwhm)
        blms = np.full((3, lmax), blms_from_fwhm)

    elif beam_type == 'PO':
        sub_freqs = get_freqs(band_name)
        beam_maps = get_beam_healpix_maps(band_name=band_name, nside=nside, 
                                          do_scaling=do_scaling, comp=comp,
                                          **kwargs)
        blms = hp.map2alm(beam_maps, pol=pol, mmax=blm_mmax)
        blms *= np.sqrt(4*np.pi/(2*np.arange(lmax+1)+1))
        blms /= np.nanmax(blms[0])
    
    outdir = kwargs['outdir']
    prefix = kwargs['prefix']
    
    if outdir is not None:
        # find a better name 
        if prefix is None:
            prefix = band_name+'_'+str(nside)+'_'+str(int(do_scaling))
        np.save(opj(outdir, prefix+'.npy'), blms)
                
    return blms

def conv_sky(sky, band_names, nside, lmax=None, blm_mmax=0, 
             pol=False, **kwargs):
    """
    Return the 'beam-convolved' maps for all sky components described in
    'sky' and frequency bands in 'band_names'

    Args
    ----
    sky: Dictionary containing PySM foreground models
    band_names: Array of frequency band names
    betas: Dictionary of per component spectral index
    nside: Desired healpix map resolution
    lmax: Maximum multipole number to truncate harmonic coefficients
    blm_mmax: Maximum mode to include. Default:0 for a symmetric beam
    pol: If True (False) hp.map2alm() returns [almT,almE,almB] ([almI, almQ, almU]).

    Output
    ------
    sig: array of size (len(band_names), 3, hp.nside2npix(nside))
         The sky signal per frequency where all sky components have been
         added after being bandpassed and 'beam-convolved'.
    """
    
    conv_sky_keys = ['space', 'do_scaling', 'beam_type']

    space = kwargs['space']
    do_scaling = kwargs['do_scaling']
    beam_type = kwargs['beam_type']

    for key in conv_sky_keys:
    #     exec('{KEY} = {VALUE}'.format(KEY = key, VALUE = repr(kwargs[key])))
        kwargs.pop(key)

    if lmax is None:
       lmax = 3*nside-1

    freqs = ut.get_freqs(band_names)
    n_freqs = len(freqs)
    sig = np.zeros((len(freqs), 3, hp.nside2npix(nside)))

    if np.any(do_scaling==True):
        beta_bf = {'cmb':1, 'synchrotron':-3, 'dust':1.59}
        temp_bf = {'cmb':None, 'synchrotron':None, 'dust':19.6}

        for component in sky.Components:
     
            if component not in kwargs['betas'].keys():
                kwargs['betas'][component] = beta_bf[component]
            if component not in kwargs['betas'].keys():
                kwargs['temps'][component] = temps[component]

    # # if apply_bandpass:
    # bpss = np.array([np.loadtxt('/cfs/home/koda4949/simonsobs/beam_chromaticity/data/bandpasses/'+n+'.txt') for n in band_names])
    # bpss_nu = [bpss[i][:,0] for i in range(len(bpss))]
    # bpss_bnu = [bpss[i][:,1] for i in range(len(bpss))]
    
    for band_name,freq,freq_idx,scale_beam in zip(band_names,freqs,range(n_freqs),do_scaling):
 
        if space=='map':

            for component in sky.Components:

                sig_i = getattr(sky, component)(freq, **kwargs)

                # sig_i = sky3.get_emission(bpss_nu[freq_idx] * u.GHz, bpss_bnu[freq_idx])

                beam_fwhm = get_beam_fwhm(band_name=band_name, nside=nside,
                                          do_scaling=do_scaling[freq_idx], 
                                          comp=component, **kwargs)
                print('before smoothing and shape of sig_i=',np.shape(sig_i))
                sig[freq_idx,:,:] += hp.smoothing(sig_i, np.radians(beam_fwhm/60.))

        elif space=='harmonic':

            for component in sky.Components:

                sig_i = getattr(sky, component)(freq, **kwargs)
                # sig_i = sky3.get_emission(bpss_nu[freq_idx] * u.GHz, bpss_bnu[freq_idx])

                alms = hp.map2alm(sig_i, lmax=lmax, pol=pol)
                beam_blm = get_beam_blms(band_name=band_name, nside=nside, 
                                         lmax=lmax, blm_mmax=blm_mmax,
                                         do_scaling=do_scaling[freq_idx], 
                                         beam_type=beam_type[freq_idx], 
                                         pol=pol, comp=component, **kwargs)

                alms_conv = np.zeros_like(alms)

                # first order perurbation of cross-pol
                alms_conv[0] = hp.almxfl(alms[0,:],beam_blm[0,:])
                alms_conv[1] = hp.almxfl(alms[1,:],beam_blm[1,:]) - hp.almxfl(alms[2,:],beam_blm[2,:])
                alms_conv[2] = hp.almxfl(alms[2,:],beam_blm[1,:]) - hp.almxfl(alms[1,:],beam_blm[2,:])

                sig[freq_idx,:,:] += hp.alm2map(alms_conv, nside=nside, lmax=lmax, pol=pol)
          
    return sig
