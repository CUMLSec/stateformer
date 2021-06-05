from command.params import fields, byte_start_pos, field_cf
from multiprocessing import Pool
import subprocess


def run(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--trainpref', f'data-src/pretrain/train.{field}',
         '--validpref',
         f'data-src/pretrain_cf/valid.{field}', '--destdir', f'data-bin/pretrain/{field}', '--workers',
         '40'])


def run_byte_with_dict(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', 'data-bin/byte_dict.txt', '--trainpref',
         f'data-src/pretrain/train.{field}',
         '--validpref',
         f'data-src/pretrain/valid.{field}', '--destdir', f'data-bin/pretrain/{field}',
         '--workers',
         '40'])


with Pool() as pool:
    pool.map(run_byte_with_dict, fields[byte_start_pos:])

with Pool() as pool:
    pool.map(run, fields[:byte_start_pos] + [field_cf])
