"""序列身份键:响应 token id 的 64 位滚动哈希,driver(traj_dump)与 worker(div_panel)各自算,
离线用它把两边的记录对上 —— 两边没有共同的序号(worker 只看到 micro-batch),但看得到同一串 id。

    key = ( L * M^L + sum_i id_i * M^(L-1-i) ) mod 2^64,  M = 1000003

纯整数运算,numpy 版与 python 版逐位一致(uint64 自然回绕);**输出截到 63 位**,让它落在有符号
int64 里 —— pandas.read_json(ujson)读不了 > 2^63 的整数,parquet 也省得走 uint64。同一步内两条
不同响应撞键的概率 ~ B^2 / 2^64,可忽略;不同步之间还有 step 列兜底。"""
M = 1000003
MASK = (1 << 64) - 1
OUT = (1 << 63) - 1


def py_key(ids):
    h = len(ids) & MASK
    for x in ids:
        h = (h * M + int(x)) & MASK
    return h & OUT


def np_key(ids):
    """ids: 1-D 整数数组(响应段,已去 padding)。返回 python int(无符号)。"""
    import numpy as np

    a = np.asarray(ids, dtype=np.uint64)
    n = a.size
    if n == 0:
        return 0
    m = np.uint64(M)
    with np.errstate(over="ignore"):
        p = np.cumprod(np.full(n, m, dtype=np.uint64))      # M^1 .. M^n(回绕)
        pw = np.concatenate([np.uint64(1)[None], p[:-1]])[::-1]  # M^(n-1-i)
        h = np.uint64(n) * p[-1] + (a * pw).sum(dtype=np.uint64)
    return int(h) & OUT


def keys_from_padded(resp, lens):
    """resp: [B, T] 整数数组;lens: [B]。逐行 np_key(resp[b, :lens[b]])。"""
    return [np_key(resp[b, : int(lens[b])]) for b in range(len(lens))]
