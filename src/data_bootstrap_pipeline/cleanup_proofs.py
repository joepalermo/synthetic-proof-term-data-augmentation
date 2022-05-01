import os
import shutil

# empty and recreate output/modified_proofs directory
if os.path.isdir('output/modified_proofs'):
    shutil.rmtree('output/modified_proofs')
os.mkdir('output/modified_proofs')


def replace_universes(proof, uvs_in_proof):
    fixed_universe_variables = ["u_1001", "u_1002", "u_1003", "u_1004", "u_1005", "u_1006"]
    uv_mapping = {uv: fixed_universe_variables[i] for i, uv in enumerate(uvs_in_proof)}
    replace_dict = {}
    for uv, fuv in uv_mapping.items():
        replace_dict["Type %s}"%uv] = "Type %s}"%fuv
        replace_dict["Sort %s}"%uv] = "Sort %s}"%fuv
        replace_dict["{%s}"%uv] = "{%s}"%fuv
        replace_dict["{%s "%uv] = "{%s "%fuv
        replace_dict[" %s}"%uv] = " %s}"%fuv
        replace_dict["{%s+"%uv] = "{%s+"%fuv
        replace_dict["+%s}"%uv] = "+%s}"%fuv
        replace_dict[" %s "%uv] = " %s "%fuv
    for key, value in replace_dict.items():
        proof = proof.replace(key, value)
    return proof


proofs_dirpath = 'output/proofs'
type_universe_variables_dirpath = 'output/type_universe_variables'

proofs_filenames = [fn for fn in os.listdir(proofs_dirpath) if '.txt' in fn]
tuv_filenames = [fn for fn in os.listdir(type_universe_variables_dirpath) if '.txt' in fn]
assert proofs_filenames == tuv_filenames
proofs_filepaths = [os.path.join(proofs_dirpath, fn) for fn in proofs_filenames]
tuv_filepaths = [os.path.join(type_universe_variables_dirpath, fn) for fn in tuv_filenames]

num_proofs = 0
num_unique_proofs = 0
for proofs_filepath, tuv_filepath in zip(proofs_filepaths, tuv_filepaths):
    filename = proofs_filepath.split('/')[-1]
    declaration_name = filename.split('.txt')[0]

    # read proofs
    with open(proofs_filepath) as f:
        proofs = f.read().split(';\n')
        proofs = [p.strip() for p in proofs if p.strip() != ""]

    # read type_universe_variables 
    with open(tuv_filepath) as f:
        universe_variables_as_strings = f.read().split('\n')
        universe_variables_as_strings = [uv for uv in universe_variables_as_strings if uv.strip() != ""]
        universe_variables = []
        for uvs_string in universe_variables_as_strings:
            uvs = uvs_string.strip()[1:-1].split(',')
            uvs = [uv.strip() for uv in uvs if len(uv) > 0]
            universe_variables.append(uvs)

    # replace type universe variable names + remove duplicate proofs
    unique_proofs = set()
    assert len(universe_variables) == len(proofs), f"{len(universe_variables)} == {len(proofs)}"
    for uvs, proof in list(zip(universe_variables, proofs)):
        mod_proof = replace_universes(proof, uvs)
        unique_proofs.add(mod_proof)

    # write modified proofs to file
    modified_filename = filename.replace("\'", "_")
    with open(f'output/modified_proofs/{modified_filename}', 'w') as f:
        for i, mod_proof in enumerate(unique_proofs):
            f.write(mod_proof)
            if i != len(unique_proofs)-1:
                f.write(';\n')

    # update overall counts
    num_proofs += len(proofs)
    num_unique_proofs += len(unique_proofs)

print(f'num proofs {num_proofs}')
print(f'num unique proofs {num_unique_proofs}')