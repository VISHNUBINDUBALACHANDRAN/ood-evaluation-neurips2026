"""make_split_configs.py — write the near-OOD split configurations used in the paper.

Rebuilds every split with the same procedure as the runs that produced the reported numbers and checks the
resulting sizes against the stored results before writing anything.

  tobacco3482_leave_class_out.json   documents: 4 held-out classes x 5 seeds (LayoutLMv3 leave-class-out models)
  cifar100_small_mammals.json        vision: ID = hamster, mouse, rabbit, shrew; near = squirrel; far = rocket
  20newsgroups_comp.json             text: ID = comp.* (5 classes); near = sci.electronics; far = talk.politics.mideast

  python split_configs/make_split_configs.py --tobacco <processed Tobacco3482 dataset (save_to_disk folder)> \
      --cifar100 <folder containing cifar-100-python> --news_home <scikit-learn data_home with 20news-bydate_py3.pkz>
"""
import argparse, json, os, pickle
import numpy as np
from sklearn.model_selection import train_test_split

SEEDS = [42, 123, 456, 789, 1024]

# Sizes recorded by the runs (verification_2026-10/results/E1_doc_nearood/lco/*.json and outputs/nearood_polarity/*.json).
DOC_EXPECTED = {"Memo": (2289, 573, 620, 114), "Form": (2440, 611, 431, 114),
                "Resume": (2689, 673, 120, 114), "Scientific": (2576, 645, 261, 114)}   # train, eval, ood, Letter-only eval
VISION_EXPECTED = {"eval": 200, "near": 600, "far": 600}
TEXT_EXPECTED = {"eval": 468, "near": 800, "far": 800}

CIFAR_ID, CIFAR_NEAR, CIFAR_FAR = ["hamster", "mouse", "rabbit", "shrew"], "squirrel", "rocket"
NEWS_ID_PREFIX, NEWS_NEAR, NEWS_FAR = "comp.", "sci.electronics", "talk.politics.mideast"
N_EVAL = 800        # run_nearood_polarity.py --n_eval (cap on ID eval and on each OOD set)


def tobacco(path):
    from datasets import load_from_disk
    labels = list(load_from_disk(path)["labels"])
    classes = sorted(set(labels))
    out = {"dataset": "Tobacco3482 (10 classes), processed by run_prepare_data.py",
           "row_indices_refer_to": "rows of the processed dataset written by run_prepare_data.py (colated_tensors). "
                                   "Its row order follows the file listing at build time, so the class sequence of the rows "
                                   "is given in labels_by_row.",
           "procedure": "for held-out class h: ID rows = rows whose class is not h, in dataset order; label2id = sorted ID "
                        "classes; train/eval = sklearn train_test_split(train_size=0.8, stratify=label, random_state=seed); "
                        "OOD = all rows of class h; Letter-only eval = eval rows of class Letter",
           "classes": classes, "labels_by_row": labels, "splits": {}}
    for h, (n_tr, n_ev, n_ood, n_let) in DOC_EXPECTED.items():
        train_classes = [c for c in classes if c != h]
        l2i = {c: i for i, c in enumerate(sorted(train_classes))}
        id_rows = [i for i, l in enumerate(labels) if l in train_classes]
        y = np.array([l2i[labels[i]] for i in id_rows])
        ood = [i for i, l in enumerate(labels) if l == h]
        for s in SEEDS:
            tr, ev = train_test_split(np.arange(len(id_rows)), train_size=0.8, random_state=s, stratify=y)
            tr, ev = [id_rows[i] for i in tr], [id_rows[i] for i in ev]
            letter = [i for i in ev if labels[i] == "Letter"]
            got = (len(tr), len(ev), len(ood), len(letter))
            if got != (n_tr, n_ev, n_ood, n_let):
                raise SystemExit(f"Tobacco {h} seed {s}: sizes {got} != stored {(n_tr, n_ev, n_ood, n_let)}")
            out["splits"][f"holdout_{h}_seed{s}"] = {"holdout": h, "seed": s, "label2id": l2i,
                                                     "n_train": n_tr, "n_eval": n_ev, "n_ood": n_ood, "n_eval_letter": n_let,
                                                     "train_rows": tr, "eval_rows": ev, "ood_rows": ood, "eval_letter_rows": letter}
    return out


def cifar100(root):
    base = os.path.join(root, "cifar-100-python")
    load = lambda n: pickle.load(open(os.path.join(base, n), "rb"), encoding="latin1")
    names = load("meta")["fine_label_names"]
    tr_t, te_t = load("train")["fine_labels"], load("test")["fine_labels"]
    n2i = {c: i for i, c in enumerate(names)}
    lmap = {n2i[c]: i for i, c in enumerate(CIFAR_ID)}
    out = {"dataset": "CIFAR-100 (python version; indices refer to the official train / test splits, as in torchvision.datasets.CIFAR100)",
           "id_classes": CIFAR_ID, "near_ood_class": CIFAR_NEAR, "far_ood_class": CIFAR_FAR,
           "label2id": {c: i for i, c in enumerate(CIFAR_ID)},
           "procedure": "ID train-split images of the 4 ID classes, shuffled with numpy RandomState(seed); first 90% = train, "
                        "rest = eval (cap 800). Each OOD class = all its test-split images followed by all its train-split images "
                        "(600 <= cap 800, so no subsampling).",
           "splits": {}}
    for s in SEEDS:
        idx = [i for i, t in enumerate(tr_t) if t in lmap]
        np.random.RandomState(s).shuffle(idx)
        cut = int(0.9 * len(idx))
        tr, ev = idx[:cut], idx[cut:][:N_EVAL]
        ood = {}
        for tag, c in (("near", CIFAR_NEAR), ("far", CIFAR_FAR)):
            te_i = [i for i, t in enumerate(te_t) if t == n2i[c]]
            tr_i = [i for i, t in enumerate(tr_t) if t == n2i[c]]
            if len(te_i) + len(tr_i) > N_EVAL:
                raise SystemExit("unexpected OOD subsampling")
            ood[tag] = {"test_split_indices": te_i, "train_split_indices": tr_i}
        got = {"eval": len(ev), "near": sum(map(len, ood["near"].values())), "far": sum(map(len, ood["far"].values()))}
        if got != VISION_EXPECTED:
            raise SystemExit(f"CIFAR-100 seed {s}: sizes {got} != stored {VISION_EXPECTED}")
        out["splits"][f"seed{s}"] = {"seed": s, "n_train": len(tr), "n_eval": len(ev), "n_near_ood": got["near"],
                                     "n_far_ood": got["far"], "train_indices": tr, "eval_indices": ev,
                                     "near_ood": ood["near"], "far_ood": ood["far"]}
    return out


def news(home):
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="all", data_home=home, remove=("headers", "footers", "quotes"), download_if_missing=False)
    names = d.target_names
    pick = lambda ok: [i for i, (t, y) in enumerate(zip(d.data, d.target)) if ok(names[y]) and len(t.strip()) > 50]
    id_all = pick(lambda n: n.startswith(NEWS_ID_PREFIX))
    cats = sorted({names[d.target[i]] for i in id_all})
    out = {"dataset": "20 Newsgroups via sklearn.datasets.fetch_20newsgroups(subset='all', remove=('headers', 'footers', "
                      "'quotes')) with its default shuffle (random_state=42); indices refer to that list",
           "id_categories": cats, "near_ood_category": NEWS_NEAR, "far_ood_category": NEWS_FAR,
           "label2id": {c: i for i, c in enumerate(cats)},
           "procedure": "documents with more than 50 non-blank characters; ID = comp.* documents permuted with numpy "
                        "RandomState(seed); first 90% = train, rest = eval (cap 800); each OOD category = 800 documents drawn "
                        "without replacement with RandomState(seed).choice",
           "splits": {}}
    for s in SEEDS:
        perm = np.random.RandomState(s).permutation(len(id_all))
        cut = int(0.9 * len(perm))
        tr = [id_all[i] for i in perm[:cut]]
        ev = [id_all[i] for i in perm[cut:][:N_EVAL]]
        ood = {}
        for tag, c in (("near", NEWS_NEAR), ("far", NEWS_FAR)):
            pool = pick(lambda n, c=c: n == c)
            ood[tag] = [pool[i] for i in np.random.RandomState(s).choice(len(pool), N_EVAL, replace=False)] if len(pool) > N_EVAL else pool
        got = {"eval": len(ev), "near": len(ood["near"]), "far": len(ood["far"])}
        if got != TEXT_EXPECTED:
            raise SystemExit(f"20 Newsgroups seed {s}: sizes {got} != stored {TEXT_EXPECTED}")
        out["splits"][f"seed{s}"] = {"seed": s, "n_train": len(tr), "n_eval": len(ev), "n_near_ood": len(ood["near"]),
                                     "n_far_ood": len(ood["far"]), "train_indices": tr, "eval_indices": ev,
                                     "near_ood_indices": ood["near"], "far_ood_indices": ood["far"]}
    return out


def to_py(o):
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    return o.item() if isinstance(o, np.generic) else o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tobacco", required=True)
    ap.add_argument("--cifar100", required=True)
    ap.add_argument("--news_home", required=True)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    files = {"tobacco3482_leave_class_out.json": tobacco(a.tobacco),
             "cifar100_small_mammals.json": cifar100(a.cifar100),
             "20newsgroups_comp.json": news(a.news_home)}
    for name, obj in files.items():           # written only after every size check has passed
        with open(os.path.join(a.out, name), "w") as f:
            json.dump(to_py(obj), f)
        print(f"wrote {name}: {len(obj['splits'])} splits")


if __name__ == "__main__":
    main()
