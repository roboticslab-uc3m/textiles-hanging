import multiprocessing
import subprocess
import logging
import os

import begin


def work(cmd):
    logging.debug(cmd)
    return subprocess.call(cmd)


@begin.start(auto_convert=True)
@begin.logging
def main(out_dir: 'Output directory where files will be written to'='./',
         start: 'Index of the first element of the dataset to be generated'=0,
         number_per_job: 'Number of elements to be generated'=1,
         total: 'Number of consecutive jobs per Blender instance (to avoid RAM exaustion)'=5):
    """
    batch_generation.py
    ------------------------
    Multiprocess generation of training data via blender.
    """
    count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=count)
    tasks = map(lambda i: ['blender',
                           os.path.abspath(os.path.expanduser('~/repositories/textiles-hanging/blender/scene.blend')),
                           '-b', '-P',
                           os.path.abspath(os.path.expanduser('~/repositories/textiles-hanging/blender/script.py')),
                           '--', '-n', str(number_per_job), '-s', str(start+i*number_per_job), '-o', out_dir],
                range(int(total/number_per_job)))

    results = []
    r = pool.map_async(work, tasks, callback=results.append)
    r.wait()  # Wait on the results
