/-
Copyright (c) 2021 OpenAI. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Author(s): Stanislas Polu

Helper functions to work with the io monad.
-/

import system.io
import tactic.gptf.utils.util

section io
open interaction_monad interaction_monad.result
namespace io

/-- verion of io.run_tactic' which does not suppress the exception msg -/
meta def run_tactic'' {α} (tac :tactic α) : io α := do {
  io.run_tactic $ do {
    result ← tactic.capture tac,
    match result with
    | (success val _) := pure val
    | (exception m_fmt _ _) := do {
      let fmt_msg := (m_fmt.get_or_else (λ _, format!"n/a")) (),
      let msg := format!"[fatal] {fmt_msg}",
      tactic.trace msg,
      tactic.fail msg
    }
    end
  }
}

end io
end io

-- convenience function for command-line argument parsing
meta def list.nth_except {α} : list α → ℕ → string → io α := λ xs pos msg,
  match (xs.nth pos) with
  | (some result) := pure result
  | none := do
    io.fail' format!"must supply {msg} as argument {pos}"
  end