from typing import List, Tuple


def default_sizes() -> List[Tuple[int, int, int]]:
    return [
        (4096, 4096, 4096),
        (4096, 1024, 4096),
        (1024, 4096, 4096),
        (4096, 4096, 1024),
        (4032, 4128, 3968),
    ]
