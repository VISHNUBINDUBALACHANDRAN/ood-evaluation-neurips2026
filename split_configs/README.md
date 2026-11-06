# Near-OOD split configurations

The splits behind the near-OOD results in the paper, written by `make_split_configs.py`. The script rebuilds each split with the procedure used in the runs. It checks every split size against the stored results before writing.

| File | Setting | Splits | Results it backs |
|---|---|---|---|
| `tobacco3482_leave_class_out.json` | LayoutLMv3, Tobacco3482. One class held out of fine-tuning (Memo, Form, Resume or Scientific) and used as OOD. | 4 classes × 5 seeds | `verification_2026-10/results/E1_doc_nearood/lco/` |
| `cifar100_small_mammals.json` | ViT-B/16. ID: hamster, mouse, rabbit, shrew. Near-OOD: squirrel. Far-OOD: rocket. | 5 seeds | `outputs/nearood_polarity/vision_v2_seed*.json` |
| `20newsgroups_comp.json` | RoBERTa-base. ID: the 5 comp.* newsgroups. Near-OOD: sci.electronics. Far-OOD: talk.politics.mideast. | 5 seeds | `outputs/nearood_polarity/text_seed*.json` |

Each file states:
- its procedure;
- its label map;
- the split sizes;
- the exact members of every split.

**Index conventions**

- **CIFAR-100:** indices refer to the official train and test splits, in the same order as `torchvision.datasets.CIFAR100`.
- **20 Newsgroups:** indices refer to the list returned by `sklearn.datasets.fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))` with its default shuffle.
- **Tobacco3482:** indices are rows of the processed dataset written by `run_prepare_data.py`. Its row order follows the file listing at build time, so `labels_by_row` gives the class of every row. The split procedure itself is a stratified 80/20 split of the non-held-out classes with `random_state = seed`. It is fully specified in the file.

The leave-class-out models were fine-tuned on these train rows, and they are available in the checkpoint record. Their `training_metadata.json` has the same label map.

**CIFAR-100 class-count subsets.** These subsets (10, 20 and 50 classes) are not near-OOD splits. They keep CIFAR-100 label ids `0 … N−1` (`run_cifar100_benchmark.py --num_classes N`).

**Regenerate** (needs the three datasets locally; no model is run):
```bash
python split_configs/make_split_configs.py --tobacco <processed Tobacco3482 folder> \
    --cifar100 <folder containing cifar-100-python> --news_home <scikit-learn data_home>
```
