import os
import shutil
from utils import typecheck_all_proofs
from config import typecheck_all_proofs_num_processes as num_processes


# empty and recreate output directories
if os.path.isdir('output/checked_proofs'):
    shutil.rmtree('output/checked_proofs')
os.mkdir('output/checked_proofs')
if os.path.isdir('output/checked_theorems'):
    shutil.rmtree('output/checked_theorems')
os.mkdir('output/checked_theorems')

# typecheck all proofs in output/modified_proofs
source_dirpath = 'output/modified_proofs'
output_dirpath = 'output'
typecheck_all_proofs(source_dirpath, output_dirpath, num_processes)


