import subprocess


def build_batch(actives):
    batch_pairs = list(actives.keys())
    batch = [" ".join(["THEOREM", theorem, "PROOF"])
             for theorem, _ in batch_pairs]
    return batch, batch_pairs


def load_test_set_from_lm_task(filepath):
    with open(filepath) as f:
        text = f.read()
    examples = text.split('EOT')
    cleaned_examples = []
    for ex in examples:
        ex = ex.strip()
        if len(ex) > 0:
            theorem, proof = ex.split('PROOF')
            theorem = theorem.replace('THEOREM', '')
            theorem = theorem.strip()
            proof = proof.strip()
            cleaned_examples.append((theorem, proof))
    return cleaned_examples


def check_proof_correctness(ground_truth_filepath, attempted_filepath, results_filepath):
    subprocess.run(['lean', '--run', 'src/backwards_typecheck.lean', ground_truth_filepath, attempted_filepath,
                    results_filepath])