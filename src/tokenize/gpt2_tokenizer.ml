(* TODO: Unicode is not properly supported here. *)
open! Base

type t =
  { vocab : Gpt2_vocab.t
  ; pattern_re : Re.re
  ; lower_case : bool
  }

let pattern_re =
  {|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+|}

let create vocab ~lower_case =
  let pattern_re = Re.Perl.re pattern_re |> Re.compile in
  { vocab; pattern_re; lower_case }

let tokenize t str =
  Re.matches t.pattern_re str
  |> List.concat_map ~f:(fun str -> String.lowercase str |> Gpt2_vocab.bpe t.vocab)
