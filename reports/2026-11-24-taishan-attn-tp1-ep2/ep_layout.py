"""Fixed global residency and per-rank ownership for the bounded EP2 trial."""
def layout(rank):
    if rank not in (0, 1):
        raise ValueError('Only ranks 0 and 1 are supported')
    start = rank * 32
    cpu_skip = [i < 64 for i in range(256)]
    local_mask = [start <= i < start + 32 for i in range(256)]
    slots = [i - start if local_mask[i] else -1 for i in range(256)]
    return cpu_skip, local_mask, slots
