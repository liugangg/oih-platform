# Paper Metrics Extraction Workflow

## Script
`/data/oih/outputs/paper_data/extract_all_metrics.py`
Output: `/data/oih/outputs/paper_data/all_designs_complete_metrics.tsv`

## Binder/Antigen Chain Assignment — CRITICAL

Default: shortest chain = binder. **This breaks when antigen is truncated shorter than binder.**

HER2 example: binder=214aa (RFD backbone), truncated antigen=202aa → default logic swaps labels.

Fix: WHITELIST `binder_chain_len` field overrides default logic. HER2 has `binder_chain_len: 214`.

**Rule**: When adding targets with domain truncation (`DOMAIN_REGISTRY`), check if truncated domain < binder length. If so, add `binder_chain_len`.

## Run command
```bash
cd /data/oih/outputs/paper_data && python3 extract_all_metrics.py
```

## Verify
1. HER2 `binder_len` always = 214
2. Truncated antigen rows (val_4/6/7/8/9): `antigen_len=202`, not swapped
3. No unexpected SKIP/WARNING lines
