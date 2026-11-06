"""Replay calibration draws and score existing predictions on identical queries.

No model is fitted.  The draw order matches 05_prism_model.py exactly:
GroupKFold(5); studies in sorted order within a fold; k=1,3,5;
five draws for each eligible study/k; one shared default_rng(2026).
The historical all-records/same-studies comparison is retained separately.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
(ROOT/'results/prism').mkdir(parents=True,exist_ok=True)
(ROOT/'manuscript/checks').mkdir(parents=True,exist_ok=True)
SEED, K_VALUES, N_DRAWS = 2026, (1, 3, 5), 5
d = pd.read_pickle(ROOT / 'data/processed/model_table.pkl')
d = d[d.y.notna()].reset_index(drop=True)
A = pd.read_csv(ROOT / 'results/prism/anchored_predictions.csv')
R = pd.read_csv(ROOT / 'results/prism/prism_oof_k0_records.csv')
y, sid = d.y.to_numpy(), d.study_id.to_numpy()
folds = list(GroupKFold(5).split(d, y, sid))
rng = np.random.default_rng(SEED)
mu, record_rows = np.empty(len(d)), []
for fi, (_, te) in enumerate(folds):
    rr = R[R.fold == fi]
    assert np.array_equal(rr.study.to_numpy(), sid[te]), ('k0 study alignment', fi)
    assert np.allclose(rr.y.to_numpy(), y[te], rtol=0, atol=1e-12), ('k0 outcome alignment', fi)
    mu[te] = rr.mu.to_numpy()
    loc = pd.DataFrame({'i': np.arange(len(te)), 's': sid[te]})
    for s, g in loc.groupby('s'):
        ix = g.i.to_numpy()
        for k in K_VALUES:
            if len(ix) < k + 2:
                continue
            for rep in range(N_DRAWS):
                anchor = rng.choice(ix, k, replace=False)
                query = np.setdiff1d(ix, anchor)
                ai, qi = te[anchor], te[query]
                assert not np.intersect1d(ai, qi).size
                asites = set(zip(d.loc[ai, 'lat'].round(4), d.loc[ai, 'lon'].round(4)))
                same_site = [(a, b) in asites for a, b in zip(d.loc[qi, 'lat'].round(4), d.loc[qi, 'lon'].round(4))]
                record_rows.append(pd.DataFrame({'k': k, 'fold': fi, 'study': s, 'draw': rep,
                    'index': qi, 'y': y[qi], 'mu0': mu[qi], 'same_site_as_anchor': same_site,
                    'same_y_as_anchor': np.isin(y[qi], y[ai])}))
Q = pd.concat(record_rows, ignore_index=True)
assert len(Q) == len(A)
for col in ['k', 'fold', 'study']:
    assert np.array_equal(Q[col], A[col]), ('anchored row alignment', col)
assert np.allclose(Q.y, A.y, rtol=0, atol=1e-12), 'anchored outcome row alignment'
Q['PRISM'], Q['PRISM_var'] = A.PRISM, A.PRISM_var

out = {'row_wise_alignment': {'passed': True, 'n_prediction_occasions': len(Q),
           'exact_columns': ['k', 'fold', 'study'], 'outcome_absolute_tolerance': 1e-12},
       'draw_design': {'seed': SEED, 'k_values': list(K_VALUES), 'draws_per_study_k': N_DRAWS,
           'eligibility': 'At least k+2 records per study', 'outer_split': 'GroupKFold(5)',
           'iteration_order': 'fold, sorted study, k, draw',
           'scoring': 'Both predictors scored on identical non-anchor query prediction occasions; repetitions have the same weights.'},
       'within_update': 'The update adds one constant to every query prediction of a study/draw. It preserves pairwise differences, ranking and within-draw centred predictions by algebra.',
       'scores': []}
csv_rows = []
sizes = d.groupby('study_id').size()
for k, q in Q.groupby('k'):
    eligible = d.study_id.map(sizes) >= k + 2
    row = {'k': int(k), 'studies': int(q.study.nunique()), 'query_prediction_occasions': len(q),
        'unique_query_records': int(q['index'].nunique()),
        'k0_all_records_same_studies_R2': r2_score(y[eligible], mu[eligible]),
        'k0_exact_same_queries_R2': r2_score(q.y, q.mu0),
        'k0_exact_same_queries_RMSE': mean_squared_error(q.y, q.mu0)**0.5,
        'anchored_exact_same_queries_R2': r2_score(q.y, q.PRISM),
        'anchored_exact_same_queries_RMSE': mean_squared_error(q.y, q.PRISM)**0.5,
        'query_fraction_sharing_anchor_coordinate': float(q.same_site_as_anchor.mean()),
        'query_fraction_matching_an_anchor_y': float(q.same_y_as_anchor.mean())}
    gg = q.groupby(['fold', 'study', 'draw'])
    yc = q.y - gg.y.transform('mean')
    zc = q.mu0 - gg.mu0.transform('mean')
    pc = q.PRISM - gg.PRISM.transform('mean')
    row['within_query_draw_R2_k0'] = r2_score(yc, zc)
    row['within_query_draw_R2_anchored'] = r2_score(yc, pc)
    row['max_centered_prediction_change'] = float(np.max(np.abs(zc-pc)))
    assert row['max_centered_prediction_change'] < 1e-10, 'Constant-offset update invariant'
    out['scores'].append(row)
    csv_rows.append({'k': int(k), 'n_studies': row['studies'], 'n_prediction_occasions': len(q),
        'R2_k0_same_queries': row['k0_exact_same_queries_R2'],
        'RMSE_k0_same_queries': row['k0_exact_same_queries_RMSE']})

pd.DataFrame(csv_rows).to_csv(ROOT/'results/prism/prism_k0_on_query_occasions.csv', index=False)
(ROOT/'manuscript/checks/pass2_transfer_score_audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
