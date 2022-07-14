from typing import Callable

from worker.train_worker import train_worker
from worker.eval_worker import eval_worker


WORKER_DICT = {
    "train": train_worker,
    "eval": eval_worker,
}


def build_worker(task: str) -> Callable:

    WORKER_FN = WORKER_DICT[task]
    
    return WORKER_FN
