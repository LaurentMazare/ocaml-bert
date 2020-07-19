(* Use a pre-trained GPT2 model.
   vocab file: https://cdn.huggingface.co/distilgpt2-vocab.json
   merge file: https://cdn.huggingface.co/distilgpt2-merges.txt

   Weights from HuggingFace, published under the Apache 2.0 license.
     https://cdn.huggingface.co/distilgpt2-rust_model.ot
*)

open! Base
open Torch
module Model = Bert_model.Gpt2
module Tokenizer = Bert_tokenize.Gpt2_tokenizer
module Vocab = Bert_tokenize.Gpt2_vocab

let _tokenization () =
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

let () =
  let module Sys = Caml.Sys in
  if Array.length Sys.argv <> 4
  then Printf.failwithf "usage: %s weights.ot vocab.txt merge.txt" Sys.argv.(0) ();
  let vs = Var_store.create ~name:"db" ~device:Cpu () in
  let model = Model.lm_model vs Model.Config.distilgpt2 in
  Stdio.printf "Loading weights from %s\n%!" Sys.argv.(1);
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:Sys.argv.(1);
  Stdio.printf "Loading vocab/merge from %s and %s\n%!" Sys.argv.(2) Sys.argv.(3);
  let vocab = Vocab.load ~vocab_filename:Sys.argv.(2) ~merge_filename:Sys.argv.(3) in
  let tokenizer = Tokenizer.create vocab ~lower_case:true in
  Stdio.printf "Done loading\n%!";
  let token_ids =
    Tokenizer.tokenize tokenizer "The best way to go there is" |> Array.of_list
  in
  let rec loop n ~token_ids ~layer_past =
    if n <> 0
    then (
      let token_ids = Tensor.of_int2 [| token_ids |] in
      let output, layer_past =
        model
          token_ids
          ~layer_past
          ~attention_mask:None
          ~token_type_ids:None
          ~position_ids:None
          ~is_training:false
      in
      let next_token_id =
        (* TODO: rather than taking the argmax, add a notion of temperature and
           randomly sample the next word based on the probability distribution. *)
        Tensor.get (Tensor.get output 0) (-1) |> Tensor.argmax |> Tensor.to_int0_exn
      in
      Stdio.printf "Next token id: %d\n%!" next_token_id;
      Stdio.printf "Next token: %s\n%!" (Vocab.token vocab next_token_id);
      loop (n - 1) ~token_ids:[| next_token_id |] ~layer_past:(Some layer_past))
  in
  loop 10 ~token_ids ~layer_past:None
