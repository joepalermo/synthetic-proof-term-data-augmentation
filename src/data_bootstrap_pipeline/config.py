# 1 - extract_subproofs.lean
max_num_declarations_to_take=1000
max_proof_length=2048

# 3 - typecheck_all_proofs.py
typecheck_all_proofs_num_processes=150

# 4 - create_lm_tasks.py
subsample_rate=1  # proportion of declarations to take
max_proof_length=2048
max_theorem_length=2048
distance_threshold=0.15  # minimum levenstein distance to max cosine similarity train example
train_test_split_size=0.65  # percentage to take as train set
test_val_split_size=0.5  # percentage to take as val/test (before filtering)
