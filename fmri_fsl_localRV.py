#!/usr/bin/env python
# %%
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Created on Sun Aug 23 , 2020

@author: Or Duek
1st level analysis using FSL
"""

from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range

import os  # system functions

import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.fsl as fsl  # fsl
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.modelgen as model  # model generation
#import nipype.algorithms.rapidart as ra  # artifact detection
from nipype.interfaces.utility import Function


fsl.FSLCommand.set_default_output_type('NIFTI_GZ')


# %%
data_dir = os.path.abspath('/gpfs/gibbs/pi/levy_ifat/Nachshon/shared/reversal/fmriprep')
output_dir = '/gpfs/gibbs/pi/levy_ifat/Or/RV_FSL'
fwhm = 6
tr = 2 



removeTR = 4 #Number of TR's to remove before initiating the analysis
lastTR = -1 # total number of frames in the scan, after removing removeTR (i.e. if we have a 500 frames scan and we removed 5 frames and the start of scan it should be 495, unless we also want to remove some from end of scan)
thr = 0.5 # scrubbing threshold



# %% Methods
def _bids2nipypeinfo(in_file, events_file, regressors_file,
                     regressors_names=None,
                     motion_columns=None,
                     decimals=3, amplitude=1.0, removeTR=4):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from scrubFunc import scrub
    from nipype.interfaces.base.support import Bunch
    # Process the events file
    events = pd.read_csv(events_file)
    bunch_fields = ['onsets', 'durations', 'amplitudes']
    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]
    out_motion = Path('motion.par').resolve()
    regress_data = pd.read_csv(regressors_file, sep=r'\s+')
    #regress_data = scrub(regressors_file, thr) # grab also per which will be saved as file
    np.savetxt(out_motion, regress_data[motion_columns].values[removeTR:,], '%g')
    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))
    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']
    runinfo = Bunch(
        scans=in_file,
        conditions=list(set(events.trial_type.values)),
        **{k: [] for k in bunch_fields})
    for condition in runinfo.conditions:
        event = events[events.trial_type.str.match(condition)]
        runinfo.onsets.append(np.round(event.onset.values-removeTR, 3).tolist()) # added -removeTR to align to the onsets after removing X number of TRs from the scan
        runinfo.durations.append(np.round(event.duration.values, 3).tolist())
        if 'amplitudes' in events.columns:
            runinfo.amplitudes.append(np.round(event.amplitudes.values, 3).tolist())
        else:
            runinfo.amplitudes.append([amplitude] * len(event))
    if 'regressor_names' in bunch_fields:
        runinfo.regressor_names = regressors_names
        runinfo.regressors = regress_data[regressors_names].fillna(0.0).values[removeTR:,].T.tolist() # adding removeTR to cut the first rows
    return runinfo, str(out_motion)

def saveScrub(regressors_file, thr):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from scrubFunc import scrub
    # this function will call scrub and save a file with precentage of scrubbed framewise_displacement
    perFile = Path('percentScrub.txt').resolve()
    regress_data = pd.read_csv(regressors_file, sep=r'\s+')
    regress_data  = scrub(regressors_file, thr) # grab also per which will be saved as file
    x = regress_data.scrub
    per = np.array([sum(x)/len(x)])
    np.savetxt(perFile, per, '%g')
    return str(perFile)

# %%
subject_list = ['001', '004', '005', '008', '010', '011', '013', '016',
'020', '021', '022', '024', '027', '030', '038', '043',
'047', '048', '053', '055', '059', '062', '063', '065',
'071', '072', '080', '081', '082', '083', '086', '087',
'089', '095', '100', '101', '102', '103', '104', '105',
'106', '107', '108', '112', '113', '114', '118', '121',
'122', '124', '126', '130', '131', '132', '136', '144',
'146', '150', '152', '153', '154', '158', '160', '161',
'165', '166', '167', '168', '169', '170', '171', '172',
'173', '177', '178', '179', '182', '183', '184', '185',
'189', '193', '196', '200', '202', '203', '204', '205',
'1205', '1207', '1222', '1223', '1232',] # 
# Map field names to individual subject runs.


infosource = pe.Node(util.IdentityInterface(fields=['subject_id'
                                            ],
                                    ),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list)]

# SelectFiles - to grab the data (alternativ to DataGrabber)
templates = {'func': os.path.join(data_dir, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'),
             'mask': os.path.join(data_dir, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz'),
             'regressors': os.path.join(data_dir, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task*_desc-confounds_regressors.tsv'),
             'events': os.path.join('/gpfs/gibbs/pi/levy_ifat/Or/RV_FSL', 'eventfiles', 'sub-{subject_id}.csv')}


selectfiles = pe.Node(nio.SelectFiles(templates,
                               base_directory=data_dir),
                   name="selectfiles")

# %%

# Extract motion parameters from regressors file
runinfo = pe.Node(util.Function(
    input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names', 'removeTR'],
    function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
    name='runinfo')


# Set the column names to be used from the confounds file
runinfo.inputs.regressors_names = ['std_dvars', 'framewise_displacement'] + \
                                   ['a_comp_cor_%02d' % i for i in range(6)]

runinfo.inputs.motion_columns   = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

runinfo.inputs.removeTR = removeTR

#runinfo.inputs.thr = thr # set threshold of scrubbing

## adding node for the saveScrub functions
svScrub = pe.Node(util.Function(
    input_names = ['regressors_file', 'thr'], output_names = ['perFile'],
    function = saveScrub), name = 'svScrub'
    )

svScrub.inputs.thr = thr
# %%
skip = pe.Node(interface=fsl.ExtractROI(), name = 'skip')
skip.inputs.t_min = removeTR
skip.inputs.t_size = lastTR

# %%
susan =  pe.Node(interface=fsl.SUSAN(), name = 'susan') #create_susan_smooth()
susan.inputs.fwhm = fwhm
susan.inputs.brightness_threshold = 1000.0
# %%
modelfit = pe.Workflow(name='fsl_fit', base_dir= output_dir)
"""
Use :class:`nipype.algorithms.modelgen.SpecifyModel` to generate design information.
"""

modelspec = pe.Node(interface=model.SpecifyModel(),
                    name="modelspec")

modelspec.inputs.input_units = 'secs'
modelspec.inputs.time_repetition = tr
modelspec.inputs.high_pass_filter_cutoff= 120
"""
Use :class:`nipype.interfaces.fsl.Level1Design` to generate a run specific fsf
file for analysis
"""

## Building contrasts
level1design = pe.Node(interface=fsl.Level1Design(), name="level1design")



# set contrasts, depend on the condition
cond_names = ['CSplusUS1','CSminus1','CSplus1', 'CSminusUS2','CSminus2','CSplus2']
# CS+US vs CS+ all
cont1 = ('Shock_NoShockGeneral', 'T', cond_names, [.5, 0, -.5, .5, -.5, 0])
# add CS+ vs. CS- all exp.
cont2 = ('CS+ > CS-', 'T', cond_names, [0, -.5, .5, 0, .5 , -.5])
# CS+ vs. baseline all
cont3 = ('CS+ > nothing', 'T', cond_names, [0, 0, .5, 0, .5 , 0])

# CS+US vs CS+ 1st half
cont4 = ('Shock_NoShocGeneral1stHalf', 'T', cond_names, [1, 0, -1, 0, 0, 0])
# CS+US vs CS+ 2nd Half
cont5 = ('Shock_NoShockGenerals2tHalf', 'T', cond_names, [0, 0, 0, 1, -1, 0])
# add CS+ vs. CSminus 1st half.
cont6 = ('CS+ > CS-1stHalf', 'T', cond_names, [0, -1, 1, 0, 0 , 0])
# CS+ vs. CS- 2nd half
cont7 = ('CS+ > CS-2stHalf', 'T', cond_names, [0, 0, 0, 0, 1 , -1])

# CS- vs. baseline all 
cont8 = ('CS- > baselineAll', 'T', cond_names, [.5, 0, 0, 0, 0 , .5])
# CS- vs. baseline 1st half
cont9 = ('CS- > baseline1stHalf', 'T', cond_names, [1, 0, 0, 0, 0 , 0])
# CS- vs. baseline 2nd half
cont10 = ('CS- > baseline2ndHalf', 'T', cond_names, [0, 0, 0, 0, 0 , 1])

## adding CS+ vs. baseline in the two halfs
cont11 = ('CS+ > nothing_1stHalf', 'T', cond_names, [0, 0, 1, 0, 0 , 0])
cont12 = ('CS+ > nothing_2ndHalf', 'T', cond_names, [0, 0, 0, 0, 1 , 0])

contrasts = [cont1, cont2, cont3, cont4, cont5, cont6, cont7, 
cont8, cont9, cont10, cont11, cont12]

level1design.inputs.interscan_interval = tr
level1design.inputs.bases = {'dgamma': {'derivs': False}}
level1design.inputs.contrasts = contrasts
level1design.inputs.model_serial_correlations = True
"""
Use :class:`nipype.interfaces.fsl.FEATModel` to generate a run specific mat
file for use by FILMGLS
"""

modelgen = pe.Node(
    interface=fsl.FEATModel(),
    name='modelgen',
    )
"""
Use :class:`nipype.interfaces.fsl.FILMGLS` to estimate a model specified by a
mat file and a functional run
"""
mask =  pe.Node(interface= fsl.maths.ApplyMask(), name = 'mask')


modelestimate = pe.Node(
    interface=fsl.FILMGLS(smooth_autocorr=True, mask_size=5, threshold=100),
    name='modelestimate')


# %%
modelfit.connect([
    (infosource, selectfiles, [('subject_id', 'subject_id')]),
    (selectfiles, runinfo, [('events','events_file'),('regressors','regressors_file')]),
    (selectfiles, svScrub, [('regressors', 'regressors_file')]),
    (selectfiles, skip,[('func','in_file')]),
    (skip,susan,[('roi_file','in_file')]),
    #(selectfiles, susan, [('mask','mask_file')]),
    (susan, runinfo, [('smoothed_file', 'in_file')]),
    (susan, modelspec, [('smoothed_file', 'functional_runs')]),
    (runinfo, modelspec, [('info', 'subject_info'), ('realign_file', 'realignment_parameters')]),
    (modelspec, level1design, [('session_info', 'session_info')]),
    (level1design, modelgen, [('fsf_files', 'fsf_file'), ('ev_files',
                                                          'ev_files')]),
    (susan, mask, [('smoothed_file', 'in_file')]),
    (selectfiles, mask, [('mask', 'mask_file')]),
    (mask, modelestimate, [('out_file','in_file')]),
    (modelgen, modelestimate, [('design_file', 'design_file'),('con_file', 'tcon_file'),('fcon_file','fcon_file')]),

])

# %% Adding data sink
# Datasink
datasink = pe.Node(nio.DataSink(base_directory=os.path.join(output_dir, 'Sink_respNOScrub')),
                                         name="datasink")


modelfit.connect([
        (modelestimate, datasink, [('results_dir','1stLevel.@results')])



])
# %%
modelfit.run('MultiProc', plugin_args={'n_procs': 10})
