from command.params import fields
from multiprocessing import Pool
import subprocess
from itertools import product

import os


def run(arch_opt, field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/pretrain/{field}/dict.txt', '--trainpref',
         f'data-src/finetune/{arch_opt}/train.{field}',
         '--validpref',
         f'data-src/finetune/{arch_opt}/valid.{field}', '--destdir', f'data-bin/finetune/{arch_opt}/{field}',
         '--workers',
         '40'])


arch_opts = ['mips-O0']

with Pool() as pool:
    pool.starmap(run, product(arch_opts, fields))

for arch_opt in arch_opts:
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/label_dict.txt', '--trainpref',
         f'data-src/finetune/{arch_opt}/train.label',
         '--validpref',
         f'data-src/finetune/{arch_opt}/valid.label', '--destdir', f'data-bin/finetune/{arch_opt}/label',
         '--workers', '40'])

    subprocess.run(
        ['cp', '-r', f'data-bin/pretrain/cover', f'data-bin/finetune/{arch_opt}/'])
