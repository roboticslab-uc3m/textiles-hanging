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
def main(out_dir='./', start=0, number_per_job=1, total=5):
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
