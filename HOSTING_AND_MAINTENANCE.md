# Hosting and Maintenance Plan

## Long-term hosting

| Asset | Platform | Persistence mechanism |
|-------|----------|-----------------------|
| Source code | GitHub (https://github.com/VISHNUBINDUBALACHANDRAN/ood-evaluation-neurips2026) | Git tags per release |
| Fine-tuned checkpoints (SafeTensors) | Zenodo (DOI [10.5281/zenodo.23338711](https://doi.org/10.5281/zenodo.23338711)) | Per-version DOI is immutable once published |
| Audit documentation, split configs | GitHub release archive | Tagged releases |
| Aggregated and per-seed result JSONs | `outputs/` directory in the public code repository | Git history under tagged releases |

The audit suite is designed to live on platforms whose persistence is not contingent on the authors. Once a Zenodo DOI is minted, that record is immutable and continues to resolve regardless of changes in author affiliation. The GitHub release tags are likewise persistent and can be re-archived by any downstream user. We deliberately rely on these public archival platforms rather than personal cloud storage; no part of the public release is hosted on personal Google Drive, Dropbox, OneDrive, or any other account-based cloud service.

## Versioning

The audit suite uses semantic versioning of the form `MAJOR.MINOR.PATCH`:

- **MAJOR**: incompatible changes to the protocol, including changes to split definitions, mask generators, or evaluation metrics.
- **MINOR**: additive changes such as new methods or new (ID, OOD) configurations.
- **PATCH**: bug fixes that do not change reported numbers.

Each version is tagged on GitHub and archived on Zenodo with a fresh DOI; the parent record on Zenodo always resolves to the most recent release.

## What the authors commit to

The following are best-effort author commitments, scoped to what we can actually deliver as individual researchers without speaking on behalf of any institution:

1. **Issue tracking and bug fixes.** We will monitor the public issue tracker on GitHub and respond to reproducibility-related issues on a best-effort basis for at least 24 months after publication. We do not commit to a fixed response SLA.
2. **Erratum.** We will publish corrections to reported numbers or methodology in a public `ERRATA.md` file in the code repository as issues are confirmed.
3. **Versioned releases.** We will continue to tag versions on GitHub and mint Zenodo DOIs for each version.

We deliberately do *not* claim institutional backing, multi-year funding, or transfer agreements with any third-party benchmark organisation. Any such arrangements would have to be confirmed independently by those organisations, and we are not in a position to make those commitments on their behalf.

## Release scope

The public release covers:

- Source code that produced the reported numbers.
- Fine-tuned model checkpoints (LayoutLMv3-base, ViT-B/16, RoBERTa derivatives), each released under the licence inherited from its base backbone.
- Per-seed evaluation outputs (JSON only — no raw data).
- Configuration files describing the (ID, OOD) splits.

Source datasets are not redistributed; they are obtained from their original publishers (see [DATA_ACCESS.md](DATA_ACCESS.md)).

## Adding new methods or configurations

External contributors may submit pull requests adding new OOD methods or new (ID, OOD) configurations. Contributions must include a reproducible run script, evaluation across the five fixed seeds, and a licence compatible with the licences declared in `LICENSE`. We review such pull requests on the same best-effort basis as bug fixes.

## Removal and deprecation

If a source dataset is withdrawn by its original maintainers — for licence withdrawal, ethical concerns, or any other reason — we will tag a new MAJOR version that removes the affected configuration and document the removal in `CHANGELOG.md`. Earlier MAJOR versions remain on Zenodo, immutable, with a deprecation notice prepended to the README.
