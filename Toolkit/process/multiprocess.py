from multiprocessing import Pool
import tqdm
import time
import argparse

def multiprocessor(func, iter, core=2, return_value=True):
    result = None
    with Pool(core) as p:
        result = list(tqdm.tqdm(p.imap(func, iter), total=len(iter)))
    return result if return_value is True else None

