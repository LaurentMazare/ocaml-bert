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
  let vocab = Vocab.load ~vocab_filename:Sys.argv.(1) ~merge_filename:Sys.argv.(2) in
  let tokenizer = Tokenizer.create vocab ~lower_case:false in
  Stdio.printf "Done loading\n%!";
  List.iter
    ~f:(fun str ->
      let token_ids =
        Tokenizer.tokenize tokenizer str
        |> List.map ~f:(Printf.sprintf "'%d'")
        |> String.concat ~sep:", "
      in
      Stdio.printf "\"%s\" %s\n%!" str token_ids)
    [ "This is a sample sentence to be tokénized"
    ; "Wondering how this will get tokenized 🤔 ?"
    ]
