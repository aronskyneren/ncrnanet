# ncRNANet

ncRNANet contains modules for non-coding RNA interaction prediction and competing endogenous RNA network analysis in spinal cord injury research.

## Installation

```bash
pip install -e .
```

## Data

The repository does not include cohort records or foundation-model embeddings. For each dataset, provide `<data-root>/<dataset>.csv` with the columns `src_id`, `src_kind`, `src_seq`, `src_expression`, `src_pathway`, `dst_id`, `dst_kind`, `dst_seq`, `dst_expression`, `dst_pathway`, and `label`. Labels must be 0 or 1. Expression values and pathway indices must be supplied for every node and remain consistent across rows. Pathway indices range from 0 to 5, with -1 indicating no assigned pathway. Supported node kinds are `lncrna`, `circrna`, `mirna`, and `mrna`.

For foundation-enabled configurations, provide `<data-root>/<dataset>.embeddings.npz` with arrays `ids` and `embeddings`. The IDs must be strings in the format `kind:id`, ordered first by kind (`lncrna`, `circrna`, `mirna`, `mrna`) and then by identifier. The matrix must contain one embedding per ID and have the dimension specified by `model.fm_dim`. Supply embeddings from the intended foundation model; its weights and inference implementation are not included.

The loader requires explicit labels and does not derive negative interactions from unlabeled pairs. Input data and any associated evidence must be obtained and prepared separately.

## Commands

```bash
ncrnanet train recipes/pour/main.yaml --data-root DATA_DIR --out-dir runs
ncrnanet eval recipes/pour/main.yaml --data-root DATA_DIR
ncrnanet transfer recipes/pour/transfer_9905_9589.yaml --data-root DATA_DIR
```

Network inference requires an integration for independently sourced database evidence and is unavailable until that integration is implemented. Results require independent validation.

## Modules

`batch` loads and prepares data; `furnace` defines model components; `forming` provides training; `calipers` and `inspection` contain evaluation utilities; `coldwork` provides network-analysis components.
