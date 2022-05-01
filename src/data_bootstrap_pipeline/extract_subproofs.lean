import system.io
import data.list
import ..utils.util

-- file system utils

meta def write_line_to_file (dest : string) (msg : string) : io unit := do
  dest_handle ← io.mk_file_handle dest io.mode.write,
  io.fs.put_str_ln_flush dest_handle msg

meta def write_to_file (dest : string) (msg : string) : io unit := do
  dest_handle ← io.mk_file_handle dest io.mode.write,
  io.fs.put_str_flush dest_handle msg

meta def append_line_to_file (dest : string) (msg : string) : io unit := do
  dest_handle ← io.mk_file_handle dest io.mode.append,
  io.fs.put_str_ln_flush dest_handle msg

meta def append_to_file (dest : string) (msg : string) : io unit := do
  dest_handle ← io.mk_file_handle dest io.mode.append,
  io.fs.put_str_flush dest_handle msg

-- printing utils

section pp

meta def enable_verbose : tactic unit := do {
  -- jesse's pp settings
  tactic.set_bool_option `pp.all true,
  tactic.set_bool_option `pp.implicit true, -- TODO(): can we get away with setting this `false`? this blows up the proof terms by a LOT
  tactic.set_bool_option `pp.universes false,
  tactic.set_bool_option `pp.notation true,
  tactic.set_bool_option `pp.generalized_field_notation true,
  tactic.set_bool_option `pp.structure_projections true,
  tactic.set_bool_option `pp.beta true,
  tactic.set_bool_option `pp.binder_types true,
  tactic.set_nat_option `pp.max_depth 128,
  tactic.set_nat_option `pp.max_steps 10000
}

meta def with_verbose {α} (tac : tactic α) : tactic α :=
tactic.save_options $ enable_verbose *> tac

end pp

meta def pp_expr (e : expr): tactic string := do
  fmt <- tactic.pp e,
  return (to_string fmt)

meta def print_decl_info (d : declaration) : tactic unit := do
  let tp := d.type,
  let v := d.value,
  let univ_params := v.collect_univ_params,
  tactic.trace v.to_raw_fmt

-- extracting library theorems

meta def process_thm (d : declaration) : option declaration :=
let n := d.to_name in
  if ¬ d.is_trusted ∨ n.is_internal then none
  else match d with
       | declaration.defn _ _ _ _ _ _ := none
       | t@(declaration.thm n ns e te) := some t
       | declaration.cnst _ _ _ _ := none
       | declaration.ax _ _ _ := none
       end

meta def library_thms : tactic $ list declaration :=
  environment.decl_filter_map <$> tactic.get_env <*> return process_thm

meta def filter_non_theorems (d : declaration) : option declaration :=
  if d.is_theorem then some d else none

meta def decl_to_proof (d : declaration) : tactic expr := return d.value

meta def decl_to_name (d : declaration) : tactic string := return d.to_name.to_string

meta def decl_to_theorem (d : declaration) : tactic expr := return d.type

meta def decl_to_inferred_theorem (d : declaration) : tactic expr := tactic.infer_type d.value

meta def cache_subproof_if_prop : expr -> string -> tactic unit := λ e decl_name, do
  is_proof <- tactic.is_proof e <|> return ff,
  proof <- pp_expr e,
  if is_proof && (proof.length < 2048)
  then tactic.unsafe_run_io (append_line_to_file ("output/proofs/" ++ decl_name ++ ".txt") (proof ++ ";\n"))
    >> tactic.unsafe_run_io (append_line_to_file ("output/type_universe_variables/" ++ decl_name ++ ".txt") e.collect_univ_params.to_string)
  else return unit.star

namespace expr
open tactic

meta def replace_body : expr -> expr -> expr := λ bindings new_body,
  -- bindings, body => new expr where body is attached to bindings
  match bindings with
    -- CASE: lam ... (lam ...)
    | lam var_name b_info var_type (lam var_name' b_info' var_type' body') := 
        lam var_name b_info var_type (replace_body (lam var_name' b_info' var_type' body') new_body)
    -- CASE: lam ... (pi ...)
    | lam var_name b_info var_type (pi var_name' b_info' var_type' body') := 
        lam var_name b_info var_type (replace_body (pi var_name' b_info' var_type' body') new_body)
    -- CASE: pi ... (lam ...)
    | pi var_name b_info var_type (lam var_name' b_info' var_type' body') := 
        pi var_name b_info var_type (replace_body (lam var_name' b_info' var_type' body') new_body)
    -- CASE: pi ... (pi ...)
    | pi var_name b_info var_type (pi var_name' b_info' var_type' body') := 
        pi var_name b_info var_type (replace_body (pi var_name' b_info' var_type' body') new_body)
    -- BASE CASE: lam ... (not lam/pi ...)
    | lam var_name b_info var_type _ := 
        lam var_name b_info var_type new_body
    -- BASE CASE: pi ... (not lam/pi ...)
    | pi var_name b_info var_type _ := 
        pi var_name b_info var_type new_body
    -- dummy case (this should never happen)
    -- TODO: throw exception
    | _ := new_body
  end

meta def traverse_subexpressions_aux : expr -> expr -> string -> tactic unit :=  λ bindings tree decl_name,

  match tree with

    -- CASE: lam 
    | e@(lam var_name b_info var_type body) := do
      let this_lam := (lam var_name b_info var_type (var 0)),
      let new_bindings := replace_body bindings this_lam,
      cache_subproof_if_prop (replace_body bindings e) decl_name,
      traverse_subexpressions_aux new_bindings var_type decl_name,
      traverse_subexpressions_aux new_bindings body decl_name

    -- CASE: pi 
    | e@(pi var_name b_info var_type body) := do
      let this_pi := (lam var_name b_info var_type (var 0)),
      let new_bindings := replace_body bindings this_pi,
      cache_subproof_if_prop (replace_body bindings e) decl_name,
      traverse_subexpressions_aux new_bindings var_type decl_name,
      traverse_subexpressions_aux new_bindings body decl_name

    -- CASE : app
    | e@(app func arg) := do
      cache_subproof_if_prop (replace_body bindings e) decl_name,
      traverse_subexpressions_aux bindings func decl_name,
      traverse_subexpressions_aux bindings arg decl_name

    -- CASE : elet
    | e@(elet var_name type assignment body) := do
      cache_subproof_if_prop (replace_body bindings e) decl_name,
      traverse_subexpressions_aux bindings type decl_name,
      traverse_subexpressions_aux bindings assignment decl_name,
      traverse_subexpressions_aux bindings body decl_name

    | _ := do
      return unit.star
  end

meta def traverse_subexpressions : nat -> (expr × string) -> tactic unit := 
  λ max_proof_len proof_and_name, do
    let proof := proof_and_name.1,
    let name := proof_and_name.2,
    proof_string <- pp_expr proof,
    if proof_string.length < 4096
      then 
        traverse_subexpressions_aux (var 0) proof name
      else 
        return unit.star

end expr

meta def main : tactic unit := do

  -- get cmdline args
  args <- tactic.unsafe_run_io (io.cmdline_args),
  num_proofs ← tactic.unsafe_run_io (args.nth_partial 0 "num_proofs"),
  max_proof_len ← tactic.unsafe_run_io (args.nth_partial 1 "max_proof_len"),
  let num_proofs := string.to_nat num_proofs, 
  let max_proof_len := string.to_nat max_proof_len, 

  -- set pp options
  os <- tactic.get_options,
  let os := os.set_bool `pp.all true,
  let os := os.set_bool `pp.implicit true, -- TODO(): can we get away with setting this `false`? this blows up the proof terms by a LOT
  let os := os.set_bool `pp.universes false,
  let os := os.set_bool `pp.notation true,
  let os := os.set_bool `pp.generalized_field_notation true,
  let os := os.set_bool `pp.structure_projections true,
  let os := os.set_bool `pp.beta true,
  let os := os.set_bool `pp.binder_types true,
  let os := os.set_nat `pp.max_depth 128,
  let os := os.set_nat `pp.max_steps 10000,
  tactic.set_options os,
  
  -- get proof terms
  env <- tactic.get_env,
  let ds := env.decl_filter_map filter_non_theorems,
  proofs <- ds.traverse decl_to_proof,
  names <- ds.traverse decl_to_name,
  let proofs_and_names := list.zip proofs names,
  -- traverse subexpressions in each proof term
  (list.take num_proofs proofs_and_names).traverse (expr.traverse_subexpressions max_proof_len),
  tactic.trace ""


-- TESTS ----------------------------------------------------------------------------------------

-- #eval do
--   let e := (expr.lam (mk_simple_name "b") binder_info.implicit (expr.const `bool [])
--             (expr.app
--               (expr.app
--               (expr.app (expr.const `eq.mp [level.zero])
--                 (expr.app (expr.const `not []) (expr.app (expr.app (expr.app (expr.const `eq [level.succ level.zero]) (expr.const `bool [])) (@expr.var tt 0)) (expr.const `bool.tt []))))
--               (expr.app (expr.app (expr.app (expr.const `eq [level.succ level.zero]) (expr.const `bool [])) (@expr.var tt 0)) (expr.const `bool.ff [])))
--               (expr.app (expr.const `eq_ff_eq_not_eq_tt []) (@expr.var tt 0)))),
--   traverse_subexpressions_aux (var 0) e 

-- run_cmd trace $
--     replace_body (
--       extract_bindings $
--       lam `α binder_info.default (expr.sort level.zero) $
--       lam `β binder_info.default (expr.sort level.zero) $
--       var 0) $ (const `eq.mp [])
