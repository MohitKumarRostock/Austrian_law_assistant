# kahm_query_regressors_by_law_n_clusters_ablation

Generated at: 2026-04-07T21:38:16.555520+00:00

## Configuration

- n_clusters values: 100, 200, 300, 400
- Temporary artifacts kept: False
- Combined inference mode: soft
- Validation fraction: 0.05
- Tune soft: True
- Tune NLMS: True

## Aggregate results

| requested_n_clusters | status | cos_mean | mse | r2_overall | num_clamped_laws | total_train_time_s |
|---:|---|---:|---:|---:|---:|---:|
| 100 | ok | 0.938682 | 0.000117 | 0.879808 | 0 | 436.81 |
| 200 | ok | 0.949674 | 0.000096 | 0.902101 | 0 | 856.31 |
| 300 | ok | 0.953579 | 0.000088 | 0.909525 | 0 | 1343.09 |
| 400 | ok | 0.954752 | 0.000086 | 0.911746 | 0 | 1687.04 |

## Best run

Best requested n_clusters: **400**

- cos_mean: 0.954752
- mse: 0.000086
- r2_overall: 0.911746
- clamped laws: 0

## Per-run notes

### n_clusters = 100

- status: ok
- models trained: 84
- temporary root: `/var/folders/f7/xlmxw2s12tx1sw_n425xb_c00000gn/T/kahm_ablation_44j4mt5z/n_clusters_100`
- combined cos_mean: 0.938682
- combined mse: 0.000117
- combined r2_overall: 0.879808
- number of recorded errors: 0

### n_clusters = 200

- status: ok
- models trained: 84
- temporary root: `/var/folders/f7/xlmxw2s12tx1sw_n425xb_c00000gn/T/kahm_ablation_44j4mt5z/n_clusters_200`
- combined cos_mean: 0.949674
- combined mse: 0.000096
- combined r2_overall: 0.902101
- number of recorded errors: 0

### n_clusters = 300

- status: ok
- models trained: 84
- temporary root: `/var/folders/f7/xlmxw2s12tx1sw_n425xb_c00000gn/T/kahm_ablation_44j4mt5z/n_clusters_300`
- combined cos_mean: 0.953579
- combined mse: 0.000088
- combined r2_overall: 0.909525
- number of recorded errors: 0

### n_clusters = 400

- status: ok
- models trained: 84
- temporary root: `/var/folders/f7/xlmxw2s12tx1sw_n425xb_c00000gn/T/kahm_ablation_44j4mt5z/n_clusters_400`
- combined cos_mean: 0.954752
- combined mse: 0.000086
- combined r2_overall: 0.911746
- number of recorded errors: 0

