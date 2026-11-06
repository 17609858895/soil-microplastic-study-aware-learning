"""Independent density-range-endpoint sensitivity; original inputs/results untouched."""
from pathlib import Path
import importlib.util
import json
import re
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'results/causal').mkdir(parents=True,exist_ok=True)
(ROOT/'manuscript/checks').mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'scripts'))
import study_level as sl

spec=importlib.util.spec_from_file_location('specs',ROOT/'scripts/07_study_level_effects.py')
specs=importlib.util.module_from_spec(spec);spec.loader.exec_module(specs)
start=time.perf_counter()
d=pd.read_pickle(ROOT/'data/processed/model_table.pkl')
d=d[d.y.notna() & d.y_raw.notna()].reset_index(drop=True)
# Explicit hyphen/en-dash/minus-separated ranges with a density unit.
pattern=re.compile(r'(\d\.\d+)\s*(?:[-–−]\s*(\d\.\d+))?\s*g\s*/?\s*(?:cm|mL|ml)')
has_range=d.pretreat_text.str.contains(r'\d\.\d+\s*[-–−]\s*\d\.\d+\s*g',na=False)
dd=d.copy()
for i in d.index[has_range]:
    # Take the largest upper endpoint/scalar in the entire pretreatment text,
    # rather than selecting a value independently for each salt.
    matches=pattern.findall(d.at[i,'pretreat_text'])
    values=[max(float(a),float(b)) if b else float(a) for a,b in matches]
    if values:
        dd.at[i,'max_density']=max(values)
changed=~np.isclose(d.max_density,dd.max_density,equal_nan=True,rtol=0,atol=1e-12)
change_table=d.loc[has_range,['ID','study_id','max_density','pretreat_text']].copy()
change_table['density_upper_endpoint']=dd.loc[has_range,'max_density']
change_table['changed']=changed[has_range]
change_table.to_csv(ROOT/'results/causal/density_endpoint_record_changes.csv',index=False,encoding='utf-8-sig')

S=sl.study_table(dd,'y')
T,Y,ctrl,binary,label=next(x for x in specs.SPECS if x[0]=='max_density')
before=pd.read_csv(ROOT/'results/causal/study_level_table.csv',index_col=0)
assert np.allclose(S.alpha,before.loc[S.index,'alpha'],rtol=0,atol=1e-10), 'Outcomes must remain unchanged'
for c in ctrl:
    assert np.allclose(S[c],before.loc[S.index,c],equal_nan=True,rtol=0,atol=1e-10), ('Controls must remain unchanged',c)
print(json.dumps({'range_records':int(has_range.sum()),'range_studies':int(d.loc[has_range,'study_id'].nunique()),
    'changed_records':int(changed.sum()),'changed_studies':int(d.loc[changed,'study_id'].nunique())}),flush=True)
theta,se=sl.rlearner(S,T,ctrl,outcome=Y,K=5,R=10,seed=0)
theta_perm,p_perm,null_sd=sl.perm_test(S,T,ctrl,outcome=Y,n_perm=1000,seed=0)
base=pd.read_csv(ROOT/'results/causal/study_level_protocol_effects.csv')
base=base[base.treatment=='max_density'].iloc[0]
rows=[{'density_coding':'Original encoded descriptor','n_studies':int(base.n_studies),
       'theta_R':base.theta_R,'se_R':base.se_R,'lo_R':base.lo_R,'hi_R':base.hi_R,
       'p_perm_study':base.p_perm_study,'n_records_changed':0,'n_studies_changed':0},
      {'density_coding':'Largest explicit range endpoint','n_studies':len(S),
       'theta_R':theta,'se_R':se,'lo_R':theta-1.96*se,'hi_R':theta+1.96*se,
       'p_perm_study':p_perm,'n_records_changed':int(changed.sum()),
       'n_studies_changed':int(d.loc[changed,'study_id'].nunique())}]
pd.DataFrame(rows).to_csv(ROOT/'results/causal/density_endpoint_sensitivity.csv',index=False)
details={'source_field':'Sample pretreatment (retained as pretreat_text)',
  'original_rule':'Find scalar density values with an optional ASCII-hyphen range whose upper endpoint is not captured; take the maximum captured value across the complete pretreatment text. En-dash ranges are matched from the upper endpoint.',
  'sensitivity_rule':'For explicit ranges separated by hyphen/en dash/minus, take the larger endpoint, then take the largest endpoint/scalar across the whole pretreatment description. All other values retain the original coding.',
  'range_records':int(has_range.sum()),'range_studies':int(d.loc[has_range,'study_id'].nunique()),
  'changed_records':int(changed.sum()),'changed_studies':int(d.loc[changed,'study_id'].nunique()),
  'outcomes_and_controls_identical':True,'K':5,'R':10,'seed':0,'n_permutations':1000,
  'theta_permutation_single_crossfit':theta_perm,'permutation_null_sd':null_sd,
  'results':rows,'elapsed_seconds':time.perf_counter()-start,
  'status':'Independent coding sensitivity, not included in the original nine-test multiplicity family.'}
(ROOT/'manuscript/checks/pass2_density_endpoint_audit.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
print(json.dumps(details,indent=2))
