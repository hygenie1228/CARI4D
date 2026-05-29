"""NFS-safe helpers for mask HDF5 files (h5py read/write on NFS can hang)."""
from __future__ import annotations

import hashlib
import os
import os.path as osp
import shutil
import tempfile

import h5py


def _local_cache_path(src: str) -> str:
    st = os.stat(src)
    key = hashlib.sha256(
        f"{osp.abspath(src)}:{st.st_size}:{st.st_mtime_ns}".encode()
    ).hexdigest()[:20]
    return osp.join(tempfile.gettempdir(), f"cari4d_masks_{key}.h5")


def open_masks_h5(path: str, mode: str = "r"):
    """Open mask H5; for read mode, copy to local disk first if needed."""
    if mode != "r":
        return h5py.File(path, mode)
    src = osp.abspath(path)
    if not osp.isfile(src):
        raise FileNotFoundError(src)
    cache = _local_cache_path(src)
    if not osp.isfile(cache) or os.stat(cache).st_size != os.stat(src).st_size:
        partial = cache + ".partial"
        shutil.copy2(src, partial)
        os.replace(partial, cache)
    return h5py.File(cache, "r")
