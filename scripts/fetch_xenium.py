"""Selectively download members of a 10x Xenium output bundle (.zip) over HTTP Range.

The full Xenium Prime breast bundle is 41 GB; we only need ~11 GB of it. The 10x CDN
honours Range requests, so we read the zip central directory remotely, then stream just
the members we want (deflate-decompressing on the fly) into data/raw/<dataset>/.

Usage:
    python -X utf8 scripts/fetch_xenium.py                # breast, default member list
    python -X utf8 scripts/fetch_xenium.py --list         # show all members and sizes
    python -X utf8 scripts/fetch_xenium.py --members cells.parquet gene_panel.json
    python -X utf8 scripts/fetch_xenium.py --dataset lung  # 8 GB lung multimodal bundle

Resumable: a finished member is skipped if the on-disk size matches; a partial member
(.part file) is restarted from scratch (deflate streams cannot be resumed mid-way).
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130 Safari/537.36"

DATASETS = {
    "breast": {
        "url": "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Breast_Cancer_FFPE/"
               "Xenium_Prime_Breast_Cancer_FFPE_outs.zip",
        "out": ROOT / "data" / "raw" / "xenium_breast_prime",
        # small-first so ROI picking can start while images stream
        "members": [
            "gene_panel.json",
            "experiment.xenium",
            "cells.parquet",
            "cell_feature_matrix.h5",
            "cell_boundaries.parquet",
            "nucleus_boundaries.parquet",
            "analysis_summary.html",
            "morphology_focus/morphology_focus_0000.ome.tif",
            "morphology_focus/morphology_focus_0001.ome.tif",
            "morphology_focus/morphology_focus_0002.ome.tif",
            "morphology_focus/morphology_focus_0003.ome.tif",
            "transcripts.parquet",
        ],
    },
    "lung": {
        "url": "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_humanLung_Cancer_FFPE/"
               "Xenium_V1_humanLung_Cancer_FFPE_outs.zip",
        "out": ROOT / "data" / "raw" / "xenium_lung_multimodal",
        "members": [
            "gene_panel.json",
            "experiment.xenium",
            "cells.parquet",
            "cell_feature_matrix.h5",
            "cell_boundaries.parquet",
            "nucleus_boundaries.parquet",
            "analysis_summary.html",
            "morphology_focus/morphology_focus_0000.ome.tif",
            "morphology_focus/morphology_focus_0001.ome.tif",
            "morphology_focus/morphology_focus_0002.ome.tif",
            "morphology_focus/morphology_focus_0003.ome.tif",
            "transcripts.parquet",
        ],
    },
}


def http_range(url: str, start: int, end: int, stream: bool = False):
    # CloudFront occasionally answers a Range request with a full 200 (seen once on a
    # cache refresh). Never read that body for a 41 GB object: close and retry.
    for attempt in range(6):
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": UA})
        resp = urllib.request.urlopen(req, timeout=120)
        if resp.status == 206:
            return resp if stream else resp.read()
        resp.close()
        print(f"      range request got {resp.status}, retrying ({attempt+1}/6)", flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("server keeps ignoring Range header")


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def read_central_directory(url: str, size: int) -> dict[str, dict]:
    tail = http_range(url, max(0, size - 1_000_000), size - 1)
    i = tail.rfind(b"PK\x06\x06")  # zip64 end of central directory
    if i >= 0:
        (_sig, _sz, _vm, _vn, _dn, _dc, _ent_disk, ent_total, cd_size, cd_off) = struct.unpack(
            "<IQHHIIQQQQ", tail[i : i + 56]
        )
    else:
        j = tail.rfind(b"PK\x05\x06")
        (_sig, _dn, _dc, _ent_disk, ent_total, cd_size, cd_off, _cl) = struct.unpack(
            "<IHHHHIIH", tail[j : j + 22]
        )
    cd = http_range(url, cd_off, cd_off + cd_size - 1)
    members: dict[str, dict] = {}
    p = 0
    while p < len(cd) and cd[p : p + 4] == b"PK\x01\x02":
        (_sig, _vm, _vn, flag, comp, _mt, _md, crc, csz, usz, fnl, exl, cml, _dsk, _ia, _ea, lho) = (
            struct.unpack("<IHHHHHHIIIHHHHHII", cd[p : p + 46])
        )
        name = cd[p + 46 : p + 46 + fnl].decode("utf-8", "replace")
        extra = cd[p + 46 + fnl : p + 46 + fnl + exl]
        if 0xFFFFFFFF in (usz, csz, lho):
            q = 0
            while q < len(extra):
                hid, hsz = struct.unpack("<HH", extra[q : q + 4])
                body = extra[q + 4 : q + 4 + hsz]
                if hid == 1:
                    vals = struct.unpack("<" + "Q" * (hsz // 8), body[: (hsz // 8) * 8])
                    k = 0
                    if usz == 0xFFFFFFFF:
                        usz = vals[k]; k += 1
                    if csz == 0xFFFFFFFF:
                        csz = vals[k]; k += 1
                    if lho == 0xFFFFFFFF:
                        lho = vals[k]; k += 1
                q += 4 + hsz
        members[name] = dict(comp=comp, crc=crc, csize=csz, usize=usz, lho=lho, flag=flag)
        p += 46 + fnl + exl + cml
    assert len(members) == ent_total, (len(members), ent_total)
    return members


def fetch_member(url: str, name: str, m: dict, out_dir: Path, chunk: int = 8 << 20) -> None:
    dest = out_dir / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == m["usize"]:
        print(f"skip  {name}  (already {m['usize']/1e9:.2f} GB)", flush=True)
        return
    # local file header: 30 bytes + name + extra, then the data
    lh = http_range(url, m["lho"], m["lho"] + 29)
    assert lh[:4] == b"PK\x03\x04", "bad local header"
    fnl, exl = struct.unpack("<HH", lh[26:30])
    data_start = m["lho"] + 30 + fnl + exl
    data_end = data_start + m["csize"] - 1
    part = dest.with_suffix(dest.suffix + ".part")
    dec = zlib.decompressobj(-15) if m["comp"] == 8 else None
    if m["comp"] not in (0, 8):
        raise RuntimeError(f"{name}: unsupported compression method {m['comp']}")
    crc = 0
    written = 0
    t0 = time.time()
    last = t0
    print(f"fetch {name}  {m['usize']/1e9:.2f} GB (compressed {m['csize']/1e9:.2f} GB)", flush=True)
    for attempt in range(5):
        try:
            with http_range(url, data_start, data_end, stream=True) as resp, open(part, "wb") as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    out = dec.decompress(buf) if dec else buf
                    f.write(out)
                    crc = zlib.crc32(out, crc)
                    written += len(out)
                    now = time.time()
                    if now - last > 15:
                        rate = written / (now - t0) / 1e6
                        print(f"      {name}: {written/1e9:.2f}/{m['usize']/1e9:.2f} GB  {rate:.1f} MB/s", flush=True)
                        last = now
                if dec:
                    out = dec.flush()
                    f.write(out)
                    crc = zlib.crc32(out, crc)
                    written += len(out)
            break
        except Exception as e:  # noqa: BLE001
            print(f"      retry {attempt+1}/5 after error: {e}", flush=True)
            dec = zlib.decompressobj(-15) if m["comp"] == 8 else None
            crc, written = 0, 0
            time.sleep(10)
    else:
        raise RuntimeError(f"{name}: gave up after 5 attempts")
    if written != m["usize"] or (crc & 0xFFFFFFFF) != m["crc"]:
        raise RuntimeError(f"{name}: size/CRC mismatch (got {written} bytes, crc {crc:#x} vs {m['crc']:#x})")
    part.replace(dest)
    print(f"done  {name}  in {(time.time()-t0)/60:.1f} min", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="breast")
    ap.add_argument("--members", nargs="*", help="override the default member list")
    ap.add_argument("--list", action="store_true", help="list members and exit")
    args = ap.parse_args()
    ds = DATASETS[args.dataset]
    size = remote_size(ds["url"])
    members = read_central_directory(ds["url"], size)
    if args.list:
        for name, m in sorted(members.items(), key=lambda kv: -kv[1]["usize"]):
            print(f"{m['usize']/1e9:9.3f} GB  comp={m['comp']}  {name}")
        return 0
    wanted = args.members or ds["members"]
    missing = [w for w in wanted if w not in members]
    if missing:
        print("not in bundle:", missing, file=sys.stderr)
        return 1
    total = sum(members[w]["usize"] for w in wanted)
    print(f"{args.dataset}: {len(wanted)} members, {total/1e9:.2f} GB uncompressed -> {ds['out']}", flush=True)
    for w in wanted:
        fetch_member(ds["url"], w, members[w], ds["out"])
    print("all done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
