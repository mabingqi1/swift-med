import json

import numpy as np
import zstandard as zstd

def read_zst(path_image: str):
    with open(path_image, 'rb') as f:
        shape_str = f.readline().decode('utf-8').strip()
        dtype_str = f.readline().decode('utf-8').strip()
        compressed_data = f.read()
    dctx = zstd.ZstdDecompressor()
    decompressed_data = dctx.decompress(compressed_data)
    shape = eval(shape_str)
    img = np.frombuffer(decompressed_data, dtype=dtype_str).reshape(shape)
    return img


def read_jsonl(path_jsonl: str):
    with open(path_jsonl) as f:
        datas = []
        for line in f:
            datas.append(json.loads(line))
    return datas


def write_jsonl(datas, path_jsonl: str):
    with open(path_jsonl, 'w') as f:
        f.write("\n".join([json.dumps(x, ensure_ascii=False) for x in datas]))