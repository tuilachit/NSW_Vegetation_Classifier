# Model Assets

This folder is where the backend expects the trained model weights and metadata.
The large `.h5` files are intentionally ignored by Git so the public portfolio
repository stays lightweight.

Required files:

```text
binary_veg_other_p2_best.weights.h5
group9_after_binary_p2_best.weights.h5
binary_run_metadata.json
group9_run_metadata.json
```

Optional evaluation summaries:

```text
binary_eval_summary.csv
group9_eval_summary.csv
group9_p2_best_test_per_class_iou.csv
```

Output labels:

```text
0  = no data
1  = Alpine
2  = Arid Shrublands
3  = Dry Sclerophyll Forests
4  = Wetlands
5  = Grass / Grassy Woodlands
6  = Heathlands
7  = Rainforests
8  = Semi-arid Woodlands
9  = Wet Sclerophyll Forests
17 = Other / non-target land cover
```
