# Model Checkpoints

Zenodo holds the fine-tuned checkpoints of the runs reported in the paper with seeds 42, 123, 456, 789 and 1024:

**DOI: [10.5281/zenodo.23338711](https://doi.org/10.5281/zenodo.23338711)**

The record holds 90 checkpoints (44.26 GB of weights), packed as 19 uncompressed ZIP archives. Each archive holds complete runs. Alongside the archives the record has:
- `README.md`;
- `checkpoint_index.json`, which lists every weight file with its archive, licence, size and SHA-256;
- SHA-256 checksum lists.

| Archives | Models | Runs |
|---|---|---|
| `promised40_layoutlmv3_tobacco3482_intra.zip`, `promised40_layoutlmv3_tobacco3482_cross.zip`, `promised40_layoutlmv3_financedocs.zip` | LayoutLMv3-base: Tobacco3482 and FinanceDocs, intra- and cross-domain | 20 |
| `promised40_vit_b16_cifar10_intra.zip`, `promised40_vit_b16_cifar10_svhn.zip`, `promised40_vit_b16_cifar100.zip` | ViT-B/16: CIFAR-10 intra, CIFAR-10 vs SVHN, CIFAR-100 | 15 |
| `promised40_roberta_base_sst2.zip` | RoBERTa-base: SST-2 | 5 |
| `rebuttal_vit_b16_cifar100_c10.zip`, `rebuttal_vit_b16_cifar100_c20.zip`, `rebuttal_vit_b16_cifar100_c50.zip` | ViT-B/16: CIFAR-100 subsets of 10, 20 and 50 classes | 15 |
| `rebuttal_roberta_large_sst2_part1of3.zip`, `…_part2of3.zip`, `…_part3of3.zip` | RoBERTa-large: SST-2 | 5 |
| `rebuttal_nearood_vit_smallmammals.zip` | ViT-B/16: near-OOD, CIFAR-100 small mammals | 5 |
| `rebuttal_nearood_roberta_20news_comp.zip` | RoBERTa-base: near-OOD, 20 Newsgroups comp.* | 5 |
| `camera_ready_layoutlmv3_tobacco3482_leaveout_{Memo,Form,Resume,Scientific}.zip` | LayoutLMv3-base: Tobacco3482 with one class held out of fine-tuning | 20 |

## Download and verify
```bash
shasum -a 256 -c SHA256SUMS_zip_files.txt            # the ZIP archives
for z in *.zip; do unzip -q "$z" -d checkpoints_release; done
cd checkpoints_release
shasum -a 256 -c ../SHA256SUMS_promised40.txt          # the 40 checkpoints behind the main tables
shasum -a 256 -c ../SHA256SUMS_additional.txt          # the other checkpoints
```
On Linux, `sha256sum -c` works the same way.

`manifests/SHA256SUMS.txt` and `manifests/checkpoint_manifest.json` in this repository are byte-identical to `SHA256SUMS_promised40.txt` and `checkpoint_manifest_promised40.json` on Zenodo.

## Loading
- **LayoutLMv3** (Tobacco3482, FinanceDocs, leave-class-out): these are standard Hugging Face folders. Load them with `LayoutLMv3ForSequenceClassification.from_pretrained("<run>/best_model")`.
- **ViT-B/16 and RoBERTa:** these runs store the state dict of the repository's wrapper module. Every key starts with `model.`, and no `config.json` is stored. Load them in one of two ways:
  - Into the wrapper classes `models/vit_classifier.py:ViTClassifier` or `run_roberta_experiments.py:RoBERTaClassifier`, built from `google/vit-base-patch16-224`, `roberta-base` or `roberta-large` with the run's number of labels.
  - Or strip the `model.` prefix and load into `ViTForImageClassification` or `RobertaForSequenceClassification` with the matching `num_labels`. The classifier shape in each file gives this number.
- **Leave-class-out runs** also contain `training_metadata.json`, which records the held-out class and the label map. They were trained with transformers 4.28.0 (optimizer `adamw_hf`). The paper's other runs used transformers 4.57.6. `train_leave_class_out.py` retrains them.

## Manifests in this repository
| File / folder | Contents |
|---|---|
| `manifests/checkpoint_manifest.json` | One record per checkpoint of the main tables: run id, path, size in bytes, SHA-256 |
| `manifests/SHA256SUMS.txt` | SHA-256 of these 40 `model.safetensors` files |
| `manifests/configs/` | `config.json` of the 20 LayoutLMv3 checkpoints of the main tables |
| `manifests/ahm_masks/`, `manifests/ahm_masks_manifest.json`, `manifests/AHM_MASKS_SHA256SUMS.txt` | The AHM-selected head masks of the LayoutLMv3 runs and their hashes |

The random head masks of the ViT-B/16 and RoBERTa runs are in `random_head_masks/`.

## Licences
| Backbone | Base licence | Licence of the fine-tuned checkpoints |
|---|---|---|
| ViT-B/16 (`google/vit-base-patch16-224`) | Apache-2.0 | CC-BY-4.0 |
| RoBERTa-base (`roberta-base`) | MIT | CC-BY-4.0 |
| RoBERTa-large (`roberta-large`) | MIT | CC-BY-4.0 |
| LayoutLMv3-base (`microsoft/layoutlmv3-base`) | CC-BY-NC-SA-4.0 | CC-BY-NC-SA-4.0 (inherited; non-commercial, share-alike) |

Hashes and configuration files in this repository are released under CC-BY-4.0.
