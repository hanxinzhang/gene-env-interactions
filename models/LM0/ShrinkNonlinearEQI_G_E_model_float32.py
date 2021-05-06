# %env MKL_NUM_THREADS=2
%env THEANO_FLAGS=device=cpu, floatX=float32

import multiprocessing as mp
mp.set_start_method('forkserver')

import numpy as np
import pickle
import pymc3 as pm
import scipy.linalg
import theano.tensor as tt
import warnings
import sys
import patch_mp_connection_bpo_17560 as patch
patch.patch_mp_connection_bpo_17560()

warnings.filterwarnings('ignore')

EPS = np.finfo(float).eps
NUM_CHAINS = 4

PHE_INDEX = int(sys.argv[1])

with open('../make phenotype 09242020 DBlair/sample_phe.bpkl3', 'rb') as f:
    sample_phe = pickle.load(f)
    sample_phe_mat = sample_phe['sample_phe_mat']
    
OBS_Y = sample_phe_mat[:, PHE_INDEX]

with open('../data/sel_sample_model_variables2.bpkl3', 'rb') as f:
    var_dict = pickle.load(f)
    sex_age_fam21 = var_dict['fam21_demo']
    sex_age_fam11 = var_dict['fam11_demo']
    sex_age_fam22 = var_dict['fam22_demo']

TOTAL_SAMPLE_NUM = sample_phe_mat.shape[0]

NUM_FAM21 = sex_age_fam21.shape[0] // 3
NUM_FAM11 = sex_age_fam11.shape[0] // 2
NUM_FAM22 = sex_age_fam22.shape[0] // 4

BIAS_SEX_AGE = np.ones((TOTAL_SAMPLE_NUM, 3))
BIAS_SEX_AGE[:, 1:] = np.concatenate((sex_age_fam21, sex_age_fam11, sex_age_fam22))

with open('../data/sample_eqi_design_matrix.bpkl3', 'rb') as f:
    SAMPLE_EQI = pickle.load(f)
    SAMPLE_EQI_1 = SAMPLE_EQI[:, :5]
    
BIAS_SEX_AGE_EQI = np.hstack((BIAS_SEX_AGE, 
                              SAMPLE_EQI_1,
                              SAMPLE_EQI_1**2.0,
                              SAMPLE_EQI_1**3.0))


# -----------------------------------------------------------------------------

# Make model

basic_model = pm.Model()
    
with basic_model:
    
    # Fixed effect: sex + age, shrinkage horseshoe prior
    b_shrinkage = pm.HalfCauchy('b_shrinkage', beta=1.0, shape=18)
    fixed_b = pm.Normal('fixed_b', mu=0, sigma=20.0*b_shrinkage, shape=18)
        
    # Random effect: genetic relationship
    g_var = pm.Gamma('g_var', alpha=2.0, beta=1.0)
    
    g21_mu = np.zeros(3)
    grm21 = np.array([[1.0, 0.0, 0.5], 
                      [0.0, 1.0, 0.5],
                      [0.5, 0.5, 1.0]]) 
    g21 = pm.MvNormal('g21', mu=g21_mu, cov=grm21, shape=(NUM_FAM21, 3))
    
    g11_mu = np.zeros(2)
    grm11 = np.array([[1.0, 0.5],
                      [0.5, 1.0]]) 
    g11 = pm.MvNormal('g11', mu=g11_mu, cov=grm11, shape=(NUM_FAM11, 2))
    
    g22_mu = np.zeros(4)
    grm22 = np.array([[1.0, 0.0, 0.5, 0.5], 
                      [0.0, 1.0, 0.5, 0.5],
                      [0.5, 0.5, 1.0, 0.5],
                      [0.5, 0.5, 0.5, 1.0]]) 
    g22 = pm.MvNormal('g22', mu=g22_mu, cov=grm22, shape=(NUM_FAM22, 4))
    
    g = tt.sqrt(g_var) * tt.concatenate([g21.flatten(), g11.flatten(), g22.flatten()])
    
    # Residuals
    e_var = pm.Gamma('e_var', alpha=2.0, beta=1.0)
    e_sd = tt.sqrt(e_var)
    e_offset = pm.Normal('e_offset', mu=0.0, sd=1.0, shape=TOTAL_SAMPLE_NUM)
    e = pm.Deterministic('e', e_offset*e_sd)
    
    # Additive model
    l = tt.dot(BIAS_SEX_AGE_EQI, fixed_b) + g + e
    y = pm.Bernoulli('y', logit_p=l, observed=OBS_Y)
    
        
with basic_model:
    start, step = pm.init_nuts(init='advi+adapt_diag', 
                               chains=NUM_CHAINS, 
                               n_init=200000,
                               target_accept=0.95)
    trace = pm.sample(draws=500,
                      chains=NUM_CHAINS,
                      tune=2500,
                      start=start,
                      step=step,
                      discard_tuned_samples=True)
    
with open('tune2500draw500/trace_{}.bpkl3'.format(PHE_INDEX), 'wb') as buff:
    pickle.dump({'model': basic_model, 
                 'trace': trace,
                 'step': step}, buff)
    

# -----------------------------------------------------------------------------

# Diagnoses and results

import seaborn as sb
import matplotlib.pyplot as plt
# plt.rc('font', family='Helvetica')
from pathlib import Path
import pandas as pd
import arviz as az

results_outpath = './results/cond_{}/'.format(PHE_INDEX)
Path(results_outpath).mkdir(parents=True, exist_ok=True)

# Trace plot
pm.traceplot(trace, ['g_var', 'e_var',
                     'b_shrinkage', 'fixed_b'])
plt.savefig(results_outpath + 'trace_plot.pdf', format='pdf', bbox_inches='tight')
plt.close()

# Variable summary
var_summ = pm.summary(trace, ['g_var', 'e_var',
                              'b_shrinkage', 'fixed_b'])
var_summ.to_csv(results_outpath + 'variable_summary.csv')

# Energy plot
energy = trace['energy']
energy_diff = np.diff(energy)
sb.distplot(energy - energy.mean(), label='energy')
sb.distplot(energy_diff, label='energy diff')
plt.legend()
plt.savefig(results_outpath + 'energy_plot.pdf', format='pdf', bbox_inches='tight')
plt.close()

# Final results: h2, e2, WAIC
g_var_trace = trace['g_var']
e_var_trace = trace['e_var']
total_var_trace = g_var_trace + e_var_trace

h2_trace = g_var_trace / total_var_trace
e2_trace = e_var_trace / total_var_trace

h2_hpd = pm.hpd(h2_trace, hdi_prob=0.95)
e2_hpd = pm.hpd(e2_trace, hdi_prob=0.95)

h2_mean = h2_trace.mean()
e2_mean = e2_trace.mean()

waic = pm.waic(trace, basic_model, scale='deviance')

key_results = pd.DataFrame(data={'h2 mean': [h2_mean],
                                 'h2 95% HPD lo': [h2_hpd[0]],
                                 'h2 95% HPD up': [h2_hpd[1]],
                                 
                                 'e2 mean': [e2_mean],
                                 'e2 95% HPD lo': [e2_hpd[0]],
                                 'e2 95% HPD up': [e2_hpd[1]],
                                 
                                 'WAIC': [waic.waic],
                                 'pWAIC': [waic.p_waic]})
key_results.to_csv(results_outpath + 'key_results.csv')