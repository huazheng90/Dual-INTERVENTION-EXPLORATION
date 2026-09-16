#!/usr/bin/env python3
"""Prepare the DIE benchmark datasets under ``./data``.

The CLIP ViT-B/16 weights and the benchmark images are external assets that are
NOT bundled with this repository (same policy as the reference release). This
script prints the expected layout, validates an existing tree, and optionally
downloads the official DomainNet train/test lists and the 345-class label file.

Expected layout under DATA_ROOT (default ./data):

    office_home/
        Art/           class_name/*.jpg
        Clipart/       class_name/*.jpg
        Product/       class_name/*.jpg
        Real_World/    class_name/*.jpg
    visda/
        synthetic/     class_name/*.jpg   (labeled source)
        real/          class_name/*.jpg   (unlabeled target)
    domainnet/
        clipart/       class_name/*.jpg
        painting/      class_name/*.jpg
        real/          class_name/*.jpg
        sketch/        class_name/*.jpg
        labels_126.txt (optional: one class name per line -> 126-class subset)

Download sources:
    Office-Home  https://www.hemanthdv.org/officeHomeDataset.html
    VisDA-2017   http://ai.bu.edu/visda-2017/
    DomainNet    https://ai.bu.edu/M3SDA/   (345 classes; the widely used
                  126-class subset follows the SSDA-DomainNet convention)
"""
from __future__ import annotations

import argparse
import os
import urllib.request

LAYOUT = """Expected layout under DATA_ROOT (default ./data):

office_home/
    Art/           class_name/*.jpg
    Clipart/       class_name/*.jpg
    Product/       class_name/*.jpg
    Real_World/    class_name/*.jpg
visda/
    synthetic/     class_name/*.jpg   (labeled source)
    real/          class_name/*.jpg   (unlabeled target)
domainnet/
    clipart/       class_name/*.jpg
    painting/      class_name/*.jpg
    real/          class_name/*.jpg
    sketch/        class_name/*.jpg
    labels_126.txt (optional: one class name per line -> 126-class subset)
"""

DOMAINS = ["clipart", "painting", "real", "sketch"]
LIST_BASE = "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/"


def download_domainnet_lists(data_root: str) -> None:
    dn = os.path.join(data_root, "domainnet")
    os.makedirs(dn, exist_ok=True)
    for d in DOMAINS:
        for split in ("train", "test"):
            url = f"{LIST_BASE}{d}_{split}.txt"
            out = os.path.join(dn, f"{d}_{split}.txt")
            if os.path.exists(out):
                print(f"exists: {out}")
                continue
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, out)
    # 345-class label file (fallback only; dirs are authoritative).
    out = os.path.join(dn, "labels.txt")
    if not os.path.exists(out):
        url = f"{LIST_BASE}labels.txt"
        print(f"downloading {url}")
        try:
            urllib.request.urlretrieve(url, out)
        except Exception as e:
            print(f"labels.txt failed ({e}); class names fall back to dirs")


def validate(data_root: str) -> None:
    print(LAYOUT)
    ok = True
    for bench, domains in {
        "office_home": ["Art", "Clipart", "Product", "Real_World"],
        "visda": ["synthetic", "real"],
        "domainnet": ["clipart", "painting", "real", "sketch"],
    }.items():
        base = os.path.join(data_root, bench)
        if not os.path.isdir(base):
            print(f"[missing] {base}")
            ok = False
            continue
        for d in domains:
            n = 0
            for _, _, fs in os.walk(os.path.join(base, d)):
                n += sum(1 for f in fs if f.lower().endswith((".jpg", ".jpeg", ".png")))
            print(f"  [{bench}/{d}] {n} images" if n else f"  [missing] {bench}/{d}")
            ok = ok and n > 0
    dn126 = os.path.join(data_root, "domainnet", "labels_126.txt")
    if os.path.exists(dn126):
        k = sum(1 for _ in open(dn126, encoding="utf-8") if _.strip())
        print(f"  [domainnet/labels_126.txt] {k} classes")
    else:
        print("  [note] labels_126.txt absent -> full 345-class domainnet")
    print("\nAll good." if ok else "\nSome domains are missing; download them from the"
          " official sources listed in the header of this script.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data")
    p.add_argument("--domainnet-lists", action="store_true",
                   help="download official DomainNet train/test lists + labels")
    args = p.parse_args()
    if args.domainnet_lists:
        download_domainnet_lists(args.data_root)
    validate(args.data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
