# mlhep2018-starterkit
Starter kit for MLHEP-18 challenge

# About

Repository contains starter kit for [Semantic Segmentation of LArTPC tracks](https://competitions.codalab.org/competitions/19818) competition


# Phases

| Phase № | Competition Type | Problem Type                     | Description                                                                                                                                             | Data                                                    |
|---------|------------------|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| 1       | Public           | 3-class 3D semantic segmentation | You have events, each event is (192,192,192) 3D-tensor. Each cell in event can have signal and could be classified as 0 (no signal), 1 (electron/positron), 2 (all other particles)       | train_1-2.hdf5 (6k events), test_1-2.hdf5 (4k events)   |
| 2       | Private          | 3-class 3D semantic segmentation | Private phase of competition. Closed for submissions. Submission from previous will be automatically migrated to this phase and will be validated on private dataset            | ---                                                     |
| 3       | Public           | 4-class 3D semantic segmentation | You have events, each event is (192,192,192) 3D-tensor. Each cell in event can have signal and could be classified as 0 (no signal), 1 (electron/positron), 3 (proton), 2 (all other particles) | train_3-4.hdf5 (10k events), test_3-4.hdf5 (10k events) |
| 4       | Private          | 4-class 3D semantic segmentation | Private phase of competition. Closed for submissions. Submission from previous will be automatically migrated to this phase and will be validated on private dataset            | ---                                                     |
| 5       | Public           | 4-class 3D semantic segmentation with gaps |     The same problem as in phases 3 and 4, but data contains the gaps                                                                                                                                                    |  Data is not available yet                                                       |
| 6       | Private          | 4-class 3D semantic segmentation |                                                                                         Private phase of competition. Closed for submissions. Submission from previous will be automatically migrated to this phase and will be validated on private dataset                                                                |                                                       ---  |

# Metrics

All phases are evaluated with the following metrics with priority descend:

1) **acc_at_80** - 80-percentile of classification accuraces, averaged across each event (one event is (192,192,192) tensor)
2) **acc_at_50** - 50-percentile (median) of classification accuraces, averaged across each event
2) **acc_mean** - mean classification accuracy, each accuracy is averaged across each event (one event is (192,192,192) tensor)

All metrics implemented in [metrics.py](metrics.py) script

# Baseline

## Phases 1-2
Baselines for phases 1 (Public) and 2 (Private) are placed in [phase_1-2](phase_1-2/) directory:

* [Baseline MPNN](phase_1-2/baseline_mpnn/baseline_mpnn.ipynb)
* [Baseline CNN](phase_1-2/baseline_cnn/baseline_cnn.ipynb)

## Phases 3-4
Baselines for phases 3 (Public) and 4 (Private) are placed in [phase_3-4](phase_3-4/) directory:

* [Baseline MPNN](phase_3-4/baseline_mpnn/baseline_mpnn.ipynb)


# Environment setup

```
pip install -r requirements.txt
```
