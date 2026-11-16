"""Feasibility probe: drive the REAL vLLM 0.27.1 BlockPool prefix cache on CPU."""
import itertools
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash
from vllm.utils.hashing import sha256_cbor
from vllm.v1.request import Request
from vllm.sampling_params import SamplingParams

BLOCK = 16
init_none_hash(sha256_cbor)

def chain_hasher(req):
    out, parent = [], None
    toks = req.prompt_token_ids or []
    for i in range(0, len(toks) - BLOCK + 1, BLOCK):
        h = hash_block_tokens(sha256_cbor, parent, toks[i:i+BLOCK])
        out.append(h); parent = h
    return out

def make_req(rid, tokens):
    return Request(rid, tokens, SamplingParams(max_tokens=1), None, block_hasher=chain_hasher)

pool = BlockPool(num_gpu_blocks=6, enable_caching=True, hash_block_size=BLOCK)
print("null_block id:", pool.null_block.block_id, "free:", pool.get_num_free_blocks())
toks = list(range(0, 32))
req = make_req("A", toks)
print("block_hashes:", [h.hex()[:8] for h in req.block_hashes])
blocks = pool.get_new_blocks(2)
print("alloc ids:", [b.block_id for b in blocks])
pool.cache_full_blocks(req, blocks, 0, 2, BLOCK, 0)
print("hash set:", [(b.block_id, None if b.block_hash is None else b.block_hash.hex()[:8]) for b in blocks])
hit = pool.get_cached_block(req.block_hashes[0], [0])
print("lookup block0 ->", None if hit is None else hit[0].block_id)
for b in blocks: b.ref_cnt = 0
pool.free_blocks(blocks)
print("free after release:", pool.get_num_free_blocks())
pool.evict_blocks({blocks[0].block_id})
print("evict_blocks cleared hash:", blocks[0].block_hash is None)
print("lookup after evict:", pool.get_cached_block(req.block_hashes[0], [0]))
free = []
b = pool.free_block_queue.free_list_head
while b is not None:
    free.append(b.block_id); b = b.next_free_block
print("free queue order:", free)
