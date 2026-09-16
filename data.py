"""Dataset loaders for DIE (Office-Home, VisDA-2017, DomainNet-126).

Frozen CLIP features are cached once per (domain, split) as ``.pt`` files
alongside a ``.npy`` label/path table, so repeated runs reuse the cache.  When
``strong_aug: image`` the module also retains target image paths and provides a
callable that reloads the images and applies RandAugment on the fly (used by
the core/frontier objectives); with ``strong_aug: feature`` a feature-space
noise surrogate is used instead.
"""
from __future__ import annotations

import json
import os
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

LOGGER = __import__("logging").getLogger("die")

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_transform() -> "torch.nn.Module":
    """CLIP's canonical preprocessing (no augmentation)."""
    import torchvision.transforms.v2 as T2
    return T2.Compose([
        T2.Resize((224, 224)),
        T2.ToImage(), T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(_CLIP_MEAN, _CLIP_STD),
    ])


class DataModule:
    def __init__(self, cfg: dict, backend, device: torch.device):
        self.cfg = cfg
        self.backend = backend
        self.device = device
        self.data_root = cfg.get("data_root", "./data")
        self.cache_dir = os.path.join(self.data_root, ".feat_cache")

    # ------------------------------------------------------------- loaders
    def _class_dirs(self, bench: str, domain: str) -> Tuple[List[str], List[str]]:
        """Return (class_names, dir_paths) for one domain directory."""
        base = os.path.join(self.data_root, bench, domain)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"missing domain directory: {base}")
        cls_names = sorted([d for d in os.listdir(base)
                            if os.path.isdir(os.path.join(base, d))])
        return cls_names, [os.path.join(base, c) for c in cls_names]

    def _domainnet126(self, bench: str, domain: str) -> Tuple[List[str], List[str]]:
        """DomainNet with the widely used 126-class subset when available."""
        base = os.path.join(self.data_root, bench, domain)
        all_cls = sorted([d for d in os.listdir(base)
                          if os.path.isdir(os.path.join(base, d))])
        list_file = os.path.join(self.data_root, bench, "labels_126.txt")
        if os.path.exists(list_file):
            keep = [ln.strip() for ln in open(list_file, encoding="utf-8")
                    if ln.strip()]
            cls_names = [c for c in all_cls if c in keep]
            if len(cls_names) == 126:
                return cls_names, [os.path.join(base, c) for c in cls_names]
            LOGGER.warning("labels_126.txt has %d classes; falling back to dirs",
                           len(cls_names))
        return all_cls, [os.path.join(base, c) for c in all_cls]

    def load_split(self, bench: str, domain: str) -> Tuple[List[str], List[str],
                                                          List[str]]:
        """Return (class_names, image_paths, labels) for a domain."""
        if bench == "domainnet":
            cls_names, dirs = self._domainnet126(bench, domain)
        else:
            cls_names, dirs = self._class_dirs(bench, domain)
        paths, labels = [], []
        for c, d in enumerate(dirs):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(d, f))
                    labels.append(c)
        return cls_names, paths, labels

    def _cache_files(self, bench: str, domain: str) -> Tuple[str, str]:
        os.makedirs(self.cache_dir, exist_ok=True)
        tag = f"{bench}__{domain}".replace("/", "_")
        return (os.path.join(self.cache_dir, f"{tag}.pt"),
                os.path.join(self.cache_dir, f"{tag}.json"))

    def extract_and_cache(self, bench: str, domain: str,
                          class_names: List[str], paths: List[str],
                          labels: List[int]) -> torch.Tensor:
        z_path, meta_path = self._cache_files(bench, domain)
        if os.path.exists(z_path) and os.path.exists(meta_path):
            z = torch.load(z_path, map_location="cpu", weights_only=True)
            LOGGER.info("cache hit %s (%d)", z_path, z.shape[0])
            return z
        feats = []
        bs = 64
        tform = clip_transform()
        for i in range(0, len(paths), bs):
            batch_paths = paths[i:i + bs]
            imgs = torch.stack([tform(Image.open(p).convert("RGB"))
                                for p in batch_paths])
            feats.append(self.backend.image_features(imgs.to(self.device)).cpu())
        z = torch.cat(feats, dim=0)
        torch.save(z, z_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"class_names": class_names, "paths": paths,
                       "labels": labels}, f)
        LOGGER.info("cached %s (%d feats)", z_path, z.shape[0])
        return z

    # --------------------------------------------------------------- prepare
    def prepare(self, src: str, tgt: str, inductive: bool = False) -> Dict:
        """Assemble a (z_s, y_s, z_t, ...) dict for one transfer."""
        bench = self.cfg["benchmark"]
        try:
            cls_s, paths_s, labels_s = self.load_split(bench, src)
            cls_t, paths_t, labels_t = self.load_split(bench, tgt)
        except FileNotFoundError:
            if getattr(self.backend, "backend_kind", "") == "mock":
                # Smoke fallback: synthetic features for mock backends only,
                # so the CLI exercises run without downloading datasets.
                LOGGER.warning("data dirs missing; generating synthetic mock data")
                return self._mock_data(bench, src, tgt)
            raise
        class_names = cls_t if len(cls_t) >= len(cls_s) else cls_s
        # Align the class vocabulary to the union ordering (shared K classes).
        assert sorted(cls_s) == sorted(cls_t), "class vocabularies differ"

        z_s = self.extract_and_cache(bench, src, cls_s, paths_s, labels_s)
        z_t = self.extract_and_cache(bench, tgt, cls_t, paths_t, labels_t)
        z_s = F.normalize(z_s, dim=-1)
        z_t = F.normalize(z_t, dim=-1)
        y_s = torch.tensor(labels_s, dtype=torch.long)
        y_t = torch.tensor(labels_t, dtype=torch.long)
        if inductive:
            # Official train/test lists (already applied at cache time by
            # filtering paths); here we only support the transductive split.
            raise NotImplementedError("inductive split requires list-based "
                                      "loading; see docs/REPRODUCE.md")
        strong_aug_fn = self._make_strong_aug(paths_t, cls_t)
        return {"z_s": z_s, "y_s": y_s, "z_t": z_t, "y_t": y_t,
                "z_t_test": None, "y_t_test": None,
                "class_names": class_names,
                "strong_aug_fn": strong_aug_fn}

    def _mock_data(self, bench: str, src: str, tgt: str) -> Dict:
        """Deterministic synthetic features (mock backend / smoke only)."""
        import torch.nn.functional as Fn

        K = self.cfg["num_classes"]
        d = self.cfg["feat_dim"]
        seed = (sum(ord(c) for c in bench) * 31 + sum(ord(c) for c in src) * 7
                + sum(ord(c) for c in tgt)) % (2 ** 31)
        g = torch.Generator().manual_seed(seed)
        class_names = [f"{bench}_{i}" for i in range(K)]
        z_s = Fn.normalize(torch.randn(300, d, generator=g), dim=-1)
        z_t = Fn.normalize(torch.randn(200, d, generator=g), dim=-1)
        y_s = torch.randint(0, K, (300,), generator=g)
        y_t = torch.randint(0, K, (200,), generator=g)
        return {"z_s": z_s, "y_s": y_s, "z_t": z_t, "y_t": y_t,
                "z_t_test": None, "y_t_test": None,
                "class_names": class_names, "strong_aug_fn": None}

    def _make_strong_aug(self, paths_t: List[str],
                         cls_t: List[str]) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        if self.cfg.get("strong_aug", "image") != "image":
            return None  # trainer falls back to feature-space noise
        import torchvision.transforms.v2 as T2

        base = clip_transform()
        rand = T2.RandAugment(num_ops=2, magnitude=9)

        def _fn(idx: torch.Tensor) -> torch.Tensor:
            idx = idx.cpu().tolist()
            imgs = []
            for i in idx:
                img = Image.open(paths_t[i]).convert("RGB")
                imgs.append(rand(base(img)))
            return self.backend.image_features(torch.stack(imgs).to(self.device))

        return _fn
