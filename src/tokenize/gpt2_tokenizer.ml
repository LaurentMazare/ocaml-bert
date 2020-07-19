(* TODO: Unicode is not properly supported here. *)
open! Base

type t =
  { vocab : Gpt2_vocab.t
  ; pattern_re : Re.re
  ; lower_case : bool
  }

let pattern_re = {|'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\d+| ?[^\s\[a-zA-Z]\d]+|\s+|}

let create vocab ~lower_case =
  let pattern_re = Re.Perl.re pattern_re in
  let pattern_re = Re.compile pattern_re in
  { vocab; pattern_re; lower_case }

let tokenize t str =
  Re.matches t.pattern_re str
  |> List.concat_map ~f:(fun str ->
         let str = if t.lower_case then String.lowercase str else str in
         Gpt2_vocab.bpe t.vocab str)
