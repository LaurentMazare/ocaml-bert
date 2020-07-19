(* Use a pre-trained GPT2 model.
   vocab file: https://cdn.huggingface.co/distilgpt2-vocab.json
   merge file: https://cdn.huggingface.co/distilgpt2-merges.txt
*)

open! Base
module Tokenizer = Bert_tokenize.Gpt2_tokenizer
module Vocab = Bert_tokenize.Gpt2_vocab

let () =
  let module Sys = Caml.Sys in
  if Array.length Sys.argv <> 3
  then Printf.failwithf "usage: %s vocab.txt merge.txt" Sys.argv.(0) ();
  Stdio.printf "Loading vocab/merge from %s and %s\n%!" Sys.argv.(1) Sys.argv.(2);
  let _vocab = Vocab.load ~vocab_filename:Sys.argv.(1) ~merge_filename:Sys.argv.(2) in
  Stdio.printf "Done loading\n%!"
