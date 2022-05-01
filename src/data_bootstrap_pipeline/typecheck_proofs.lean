import all
import system.io
import data.list
import ..utils.tactic

meta def mmap_valid_pair {α β : Type} (f : α → tactic β) (l : list α) : tactic (list (prod α β)) := do
  res ← list.mmap (λ x, (do y ← f x, return (some (x,y))) <|> return none) l,
  return (list.filter_map id res)

meta def typecheck_filter (proofs : list string) : tactic (list string × list expr) := do {
  checked_proofs_and_theorems <- mmap_valid_pair parse_and_typecheck proofs,
  let (checked_proofs, checked_theorems) := checked_proofs_and_theorems.unzip,
  tactic.trace checked_theorems,
  return checked_proofs_and_theorems.unzip
}

section list_nth_except

-- convenience function for command-line argument parsing
meta def list.nth_partial {α} : list α → ℕ → string → io α := λ xs pos msg,
  match (xs.nth pos) with
  | (some result) := pure result
  | none := do {
    io.fail' format!"must supply {msg} as argument {pos}"
  }
  end

end list_nth_except

def put_str_ln_flush (h : io.handle) (s : string) : io unit :=
io.fs.put_str h s *> io.fs.put_str h "\n" *> io.fs.flush h

meta def write (dest : string) (msg : string) : io unit := do
  dest_handle ← io.mk_file_handle dest io.mode.write,
  put_str_ln_flush dest_handle msg

meta def main : io unit := do {
    args ← io.cmdline_args,
    source_dirpath ← args.nth_partial 0 "source_dirpath",
    output_dirpath ← args.nth_partial 1 "output_dirpath",
    source_filename ← args.nth_partial 2 "source_filename",
    let source_filepath := source_dirpath ++ "/" ++ source_filename,
    proofs ← (io.mk_file_handle source_filepath io.mode.read >>= λ f,
      (string.split (λ c, c = ';') <$> buffer.to_string <$> io.fs.read_to_end f)),
    (checked_proofs, checked_theorems) <- io.run_tactic $ typecheck_filter proofs,
    let ps := list.foldl (λ x y, x ++ ";\n" ++ y) checked_proofs.head checked_proofs.tail,
    write (output_dirpath ++ "/checked_proofs/" ++ source_filename) ps
}
