import utils as ut
import healpy as hp
import numpy as np
import os
import argparse as ap
import os 
import sys

opj = os.path.join

parser = ap.ArgumentParser(formatter_class=\
ap.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data-path', dest='data_path', default='none',
                  type=str, help='Directory containing the simulation data')
parser.add_argument('--initial-param-file', dest='initial_param_file', default='none',
                  type=str, help='Directory containing the initial fitting params')
parser.add_argument('--band-names', dest='band_names', action='store', default=None,
                  type=str, nargs='+', help='Frequency band names')
parser.add_argument('--parse-bms-fwhm', dest='parse_bms_fwhm', action='store', default=[True,True,False,True,True,True],
                  help='Simply parse the beam best-fit FWHM to smooth the maps')
parser.add_argument('--do-freq-scaling', dest='do_freq_scaling', action='store', default=[False,False,False,False,False,False],
                  nargs='+', help='Frequency scale beams of which frequency bands')
parser.add_argument('--beam-type', dest='beam_type', action='store',
                  default = ['Gaussian','Gaussian','PO','Gaussian','Gaussian','Gaussian'],
                  help='Work with Gaussian or PO beams.')
parser.add_argument('--save-params', dest='save_params', action='store',
                  help='Boolean, save simulation parameters.')
parser.add_argument('--simple-sky', dest='simple_sky', default=False, action='store_true',
                  help='Spectral index of dust and synchrotron are fixed.')
parser.add_argument('--beam-path', dest='beam_path', action='store',
                  default='/cfs/home/koda4949/simonsobs/beam_chromaticity/input_beams',
                  help='Provide beam path, if available, of full beam healpix maps.')
parser.add_argument('--conv_space', dest='conv_space', action='store',
                  default='harmonic', help='Beam convolution happens on map or harmonic domain.')
parser.add_argument('--conv-sed-type', dest='conv_sed_type', action='store',
                  default='plaw', help='Scale by a simple power low if True else use compoenents SED.\
                  Choose between plaw or comp_sed.')
parser.add_argument('--conv-nu-c', dest='conv_nu_c', action='store',
                  default='band_centers', help='The reference frequencies to be used for scaing.\
                  Choose from band_centers, bpass_centers or fg_centers.')
parser.add_argument('--output-dir', dest='dirname', default='none',
                  type=str, help='Output directory')
parser.add_argument('--prefix', dest='prefix', default='none',
                  type=str, help='Prefix for storing file')
parser.add_argument('--seed', dest='seed',  default=1000, type=int,
                  help='Set to define seed, default=1000')
parser.add_argument('--lmax', dest='lmax',  default=None, type=int,
                  help='Set to maximum multipole number. If None lmax=3*nside-1.')
parser.add_argument('--nside', dest='nside', default=256, type=int,
                  help='Set to define Nside parameter, default=256')
parser.add_argument('--std-dust', dest='std_dust', default=0., type=float,
                  help='Deviation from the mean value of beta dust, default = 0.')
parser.add_argument('--std-sync', dest='std_sync', default=0., type=float,
                  help='Deviation from the mean value of beta synchrotron, default = 0.')
parser.add_argument('--gamma-dust', dest='gamma_dust', default=-3., type=float,
                  help='Exponent of the beta dust power law, default=-3.')
parser.add_argument('--gamma-sync', dest='gamma_sync', default=-3., type=float,
                  help='Exponent of the beta sync power law, default=-3.')
parser.add_argument('--include-cmb', dest='include_cmb', default=True, action='store_false',
                  help='Set to remove CMB from simulation, default=True.')
parser.add_argument('--include-sync', dest='include_sync', default=True, action='store_false',
                  help='Set to remove synchrotron from simulation, default=True.')
parser.add_argument('--include-dust', dest='include_dust', default=True, action='store_false',
                  help='Set to remove dust from simulation, default=True.')
parser.add_argument('--include-E', dest='include_E', default=True, action='store_false',
                  help='Set to remove E-modes from simulation, default=True.')
parser.add_argument('--include-B', dest='include_B', default=True, action='store_false',
                  help='Set to remove B-modes from simulation, default=True.')
parser.add_argument('--mask', dest='add_mask', default=False, action='store_true',
                  help='Set to add mask to observational splits, default=False.')
parser.add_argument('--dust-vansyngel', dest='dust_vansyngel', default=False, action='store_true',
                  help='Set to use Vansyngel et al\'s dust model, default=False.')
parser.add_argument('--beta', dest='gaussian_beta', default=True, action='store_false',
                  help='Set for non-gaussian beta variation.')
parser.add_argument('--nu0-dust', dest='nu0_dust', default=353., type=int,
                  help='Set to change dust pivot frequency, default=353 GHz.')
parser.add_argument('--nu0-sync', dest='nu0_sync', default=23., type=int,
                  help='Set to change synchrotron pivot frequency, default=23 GHz.')
parser.add_argument('--A-dust-BB', dest='Ad', default=5, type=float,
                  help='Set to modify the B-mode dust power spectrum amplitude, default=5')
parser.add_argument('--alpha-dust-BB', dest='alpha_d', default=-0.42, type=float,
                  help='Set to mofify tilt in D_l^BB for dust, default=-0.42')
parser.add_argument('--A-sync-BB', dest='As', default=2, type=float,
                  help='Set to modify the B-mode dust power spectrum amplitude, default=2')
parser.add_argument('--alpha-sync-BB', dest='alpha_s', default=-0.6, type=float,
                  help='Set to mofify tilt in D_l^BB for synchrotron, default=-0.42')
parser.add_argument('--plaw-amp', dest='plaw_amps', default=True, action='store_false',
                  help='Set to use realistic amplitude maps for dust and synchrotron.')
parser.add_argument('--r-tensor', dest='r_tensor', default=0.0, type=float,
                  help='Set to mofify tensor-to-scalar ratio')
parser.add_argument('--dust-sed', dest='dust_sed', default='mbb', type=str,
                  help='Dust SED (\'mbb\' or \'hensley_draine\' or \'curved_plaw\')')
parser.add_argument('--sync-sed', dest='sync_sed', default='plaw', type=str,
                  help='Synchrotron SED (\'plaw\' or \'curved_plaw\')') #TODO: curved to add
## NEW
parser.add_argument('--dust-beta', dest='dust_beta', default='none', type=str,
                  help='Non-plaw dust beta map: leave none for d1, use GNILC for d10.')
parser.add_argument('--dust-amp', dest='dust_amp', default='none', type=str,
                  help='Non-plaw dust amplitude map: leave none for d1, use GNILC for d10.')
parser.add_argument('--unit-beams', dest='unit_beams', default=False, action='store_true',
                  help='Set to include unitary beams instead of SO-like beams, default=False.')

o = parser.parse_args()
nside = o.nside
seed = o.seed

if o.dirname == 'none':
    o.dirname = "./" 

if not os.path.exists(o.dirname):
  os.makedirs(o.dirname)

o.dirname += "sim_ns%d" %o.nside

if o.r_tensor != 0.:
    o.dirname+= f"_r%.2f"%o.r_tensor

#o.dirname+= f"_whitenoiONLY" #check noise_calc
    
if o.add_mask:
    o.dirname+= "_msk"
else:
    o.dirname+= "_fullsky"
if o.include_E:
    o.dirname+= "_E"
if o.include_B:
    o.dirname+= "_B"
if o.include_cmb:
    o.dirname+= "_cmb"
if o.include_dust:
    o.dirname+= "_dust"
if o.include_sync:
    o.dirname+= "_sync"

if not o.gaussian_beta:
    if o.dust_beta == 'GNILC':
        o.dirname+= "_GNILCbetaD"
        o.dirname+= "_PySMbetaS"
    else:
        o.dirname+= "_PySMBetas"
else:
    o.dirname+= "_stdd%.1lf_stds%.1lf"%(o.std_dust, o.std_sync)
    o.dirname+= "_gdm%.1lf_gsm%.1lf"%(-int(o.gamma_dust), -int(o.gamma_sync))
if not o.plaw_amps:
    if o.dust_amp == 'GNILC':
        o.dirname+= "_GNILCampD"
        o.dirname+= "_PySMampS"
    else:
        #o.dirname+= "_realAmps"
        o.dirname+= "_PySMAmps" #_d1s1"
else:
    o.dirname+= "_Ad%.1f" %(o.Ad)
    o.dirname+= "_As%.1f" %(o.As)
    o.dirname+= "_ad%.2f" %(-o.alpha_d)
    o.dirname+= "_as%.2f" %(-o.alpha_s)

o.dirname+= "_nu0d%d_nu0s%d" %(o.nu0_dust, o.nu0_sync)

if o.unit_beams:
    o.dirname+= "_unitBeams"
else:
    o.dirname+= "_SObeams"

o.dirname+="/s%d" % o.seed
os.system('mkdir -p ' + o.dirname)
print(o.dirname)

if o.save_params:
  arg_dict = {}
  for arg in vars(o):
    arg_dict[arg] = getattr(o, arg)
  np.save(opj(o.dirname, 'simulation_parameters.txt'), arg_dict)


# Decide whether spectral index is constant or varying
mean_p, moment_p = ut.get_default_params()
if o.std_dust > 0. :
    # Spectral index variantions for dust with std
    amp_beta_dust = ut.get_delta_beta_amp(sigma=o.std_dust, gamma=o.gamma_dust)
    moment_p['amp_beta_dust'] = amp_beta_dust
    moment_p['gamma_beta_dust'] = o.gamma_dust
if o.std_sync > 0. :
    # Spectral index variantions for sync with std
    amp_beta_sync = ut.get_delta_beta_amp(sigma=o.std_sync, gamma=o.gamma_sync)
    moment_p['amp_beta_sync'] = amp_beta_sync
    moment_p['gamma_beta_sync'] = o.gamma_sync

# Define parameters for the simulation:
# Which components do we want to include?
mean_p['include_CMB'] = o.include_cmb
mean_p['include_sync'] = o.include_sync
mean_p['include_dust'] = o.include_dust

# Which polarizations do we want to include?
mean_p['include_E'] = o.include_E
mean_p['include_B'] = o.include_B

# Modify r
mean_p['r_tensor'] = o.r_tensor

# Dust & Sync SED
mean_p['dust_SED'] = o.dust_sed
mean_p['sync_SED'] = o.sync_sed

# Modify SED template plaws for dust and sync
# i.e. define amp_d_bb, amp_s_bb, alpha_d, alpha_s
mean_p['A_dust_BB'] = o.Ad
mean_p['alpha_dust_BB'] = o.alpha_d
mean_p['A_sync_BB'] = o.As
mean_p['alpha_sync_BB'] = o.alpha_s

# Define pivot freqs
mean_p['nu0_dust'] = o.nu0_dust
mean_p['nu0_sync'] = o.nu0_sync

# Modify template of dust amplitude map (None or GNILC)
mean_p['dust_amp_map'] = o.dust_amp
mean_p['dust_beta_map'] = o.dust_beta

# Define if we're using unit or SO-like beams
mean_p['unit_beams'] = o.unit_beams

lmax = o.lmax
if lmax is None:
   lmax = 3*o.nside-1

### Theory prediction, simulation and noise
scc = ut.get_theory_sacc(o.data_path, o.band_names, o.nside, lmax=lmax, mean_pars=mean_p,
                         moment_pars=moment_p, add_11=True, add_02=False)
##scc.saveToHDF(o.dirname+"/cells_model.sacc") #old sacc
scc.save_fits(o.dirname+'/cls_fid.fits', overwrite=True)

if o.dust_vansyngel:
    mean_p['include_dust'] = False

sim = ut.get_sky_realization(o.data_path, o.band_names, o.nside, seed=o.seed,
                             mean_pars=mean_p,
                             moment_pars=moment_p,
                             gaussian_betas=o.gaussian_beta,
                             plaw_amps=o.plaw_amps,
                             compute_cls=True,
                             lmax=lmax,
                             simple_sky=o.simple_sky,
                             **{'parse_bms_fwhm':np.array(o.parse_bms_fwhm),
                                'space':o.conv_space,
                                'do_scaling':np.array(o.do_freq_scaling),
                                'beam_type':o.beam_type,
                                'beam_path':o.beam_path,
                                'sed_type':o.conv_sed_type,
                                'nu_c':o.conv_nu_c,
                                'outdir': '../input_beams/test_fscaled_beams',
                                'prefix':o.prefix,
                                'tele':'SAT',
                                'lon_rot':0,
                                'lat_rot':90,
                                'lon_crop':[-3,3],
                                'lat_crop':[-3,3],
                                'initial_param_file':o.initial_param_file,
                                'n_iter':100,
                                'acc':0.1,
                                'res':0.001})

if o.dust_vansyngel:
    import utils_vansyngel as uv
    nus = ut.get_freqs(o.band_names)
    qud = np.transpose(np.array(uv.get_dust_sim(nus, o.nside)),
                       axes=[1, 0, 2])
    units = ut.fcmb(nus)
    qud = qud/units[:, None, None]

    lmax = 3*o.nside-1
    if not (mean_p['include_E'] and mean_p['include_B']):
        for inu, nu in enumerate(nus):
            ebd = ut.qu2eb(qud[inu], o.nside, lmax)
            if not mean_p['include_E']:
                ebd[0] *= 0
            if not mean_p['include_B']:
                ebd[1] *= 0
            qud[inu, :, :] = ut.eb2qu(ebd, o.nside, lmax)
    sim['freq_maps'] += qud

noi = ut.create_noise_splits(o.data_path, o.band_names, o.nside, lmax=lmax)

# Define maps signal and noise
mps_signal = sim['freq_maps']
mps_noise = noi['maps_noise']

# Save beam maps
nu = ut.get_freqs(o.band_names)
nfreq = len(nu)
npol = 2
nmaps = nfreq*npol
npix = hp.nside2npix(o.nside)

for i in range(np.shape(mps_signal)[0]):
    hp.write_map(o.dirname+"/maps_sky_signal_Q_"+str(i)+".fits", mps_signal[i,0,:],
             overwrite=True)
    hp.write_map(o.dirname+"/maps_sky_signal_U_"+str(i)+".fits", mps_signal[i,1,:],
             overwrite=True)

###Added
mps_dust_amp = sim['maps_dust']
mps_sync_amp = sim['maps_sync']
hp.write_map(o.dirname+"/maps_dust_QU.fits", mps_dust_amp,
             overwrite=True)
hp.write_map(o.dirname+"/maps_sync_QU.fits", mps_sync_amp,
             overwrite=True)

# Create splits
nsplits = len(mps_noise)
for s in range(nsplits):
    maps_signoi = mps_signal[:,:,:]+mps_noise[s,:,:,:]
    if o.add_mask:
        maps_signoi *= noi['mask']
    hp.write_map(o.dirname+"/obs_split%dof%d.fits.gz" % (s+1, nsplits),
                 (maps_signoi).reshape([nmaps,npix]),
                 overwrite=True)

# Write splits list
f=open(o.dirname+"/splits_list.txt","w")
Xout=""
for i in range(nsplits):
    Xout += o.dirname+'/obs_split%dof%d.fits.gz\n' % (i+1, nsplits)
f.write(Xout)
f.close()
