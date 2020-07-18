(* Use a pre-trained BERT model for the masked-LM task.

   The weights and vocabulary files can be downloaded from the following links.
   These are published under the Apache 2.0 license by Google.

   Weight file:
     https://cdn.huggingface.co/bert-base-uncased-rust_model.ot
   Vocabulary file:
     https://cdn.huggingface.co/bert-base-uncased-vocab.txt

   This is a direct port of rust-bert example:
   https://github.com/guillaume-be/rust-bert/blob/master/examples/bert.rs
*)
open! Base
open! Torch
module Model = Bert_model.Bert
module Token = Bert_tokenize.Token
module Tokenizer = Bert_tokenize.Bert_tokenizer
module Vocab = Bert_tokenize.Bert_vocab

let () =
  let module Sys = Caml.Sys in
  if Array.length Sys.argv <> 3
  then Printf.failwithf "usage: %s weights.ot vocab.txt" Sys.argv.(0) ();
  let vs = Var_store.create ~name:"db" ~device:Cpu () in
  let model = Model.masked_lm vs Model.Config.base in
  Stdio.printf "Loading weights from %s\n%!" Sys.argv.(1);
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:Sys.argv.(1);
  Stdio.printf "Loading vocab from %s\n%!" Sys.argv.(2);
  let vocab = Vocab.load ~filename:Sys.argv.(2) in
  let tokenizer = Tokenizer.create vocab ~lower_case:true in
  Stdio.printf "Done loading\n%!";
  let predict_masked str ~masked_index =
    let tokens = Tokenizer.tokenize tokenizer str ~include_special_characters:true in
    let token_ids =
      List.filter_map tokens ~f:(fun token ->
          token.Token.with_id
          |> Option.map ~f:(fun with_id -> with_id.Token.With_id.token_id))
      |> Array.of_list
    in
    let masked_token_id = token_ids.(masked_index) in
    token_ids.(masked_index) <- 103;
    let token_ids = Tensor.of_int2 [| token_ids |] in
    let output =
      model
        ~input_ids:token_ids
        ~mask:None
        ~encoder_hidden_states:None
        ~encoder_mask:None
        ~is_training:false
    in
    let output_token =
      Tensor.get (Tensor.get output 0) masked_index
      |> Tensor.argmax
      |> Tensor.to_int0_exn
      |> Vocab.token vocab
    in
    Stdio.printf
      "Prediction for masked value \"%s\" in \"%s\": %s\n%!"
      (Vocab.token vocab masked_token_id)
      str
      output_token
  in
  predict_masked "Looks like one thing is missing" ~masked_index:4;
  predict_masked "It was a very nice and sunny day" ~masked_index:7;
  predict_masked "It's like comparing oranges to apples" ~masked_index:6;
  predict_masked "X is the best programming language" ~masked_index:1
