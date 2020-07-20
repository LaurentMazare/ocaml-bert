open Base
open Torch
module Config = Albert_config

let embeddings vs (config : Config.t) =
  let word_e =
    Layer.embeddings
      Var_store.(vs / "word_embeddings")
      ~num_embeddings:config.vocab_size
      ~embedding_dim:config.embedding_size
  in
  let pos_e =
    Layer.embeddings
      Var_store.(vs / "position_embeddings")
      ~num_embeddings:config.max_position_embeddings
      ~embedding_dim:config.embedding_size
  in
  let token_type_e =
    Layer.embeddings
      Var_store.(vs / "token_type_embeddings")
      ~num_embeddings:config.vocab_size
      ~embedding_dim:config.embedding_size
  in
  let layer_norm =
    Layer.layer_norm Var_store.(vs / "LayerNorm") config.embedding_size ~eps:1e-12
  in
  fun input_ids ~token_type_ids ~position_ids ~is_training ->
    let input_shape = Tensor.size input_ids in
    let seq_len = List.last_exn input_shape in
    let position_ids =
      match position_ids with
      | Some position_ids -> position_ids
      | None ->
        Tensor.arange ~end_:(Scalar.i seq_len) ~options:(T Int64, Var_store.device vs)
        |> Tensor.unsqueeze ~dim:0
        |> Tensor.expand ~size:input_shape ~implicit:true
    in
    let token_type_ids =
      match token_type_ids with
      | Some token_type_ids -> token_type_ids
      | None -> Tensor.zeros_like input_ids
    in
    let word_e = Layer.forward word_e input_ids in
    let pos_e = Layer.forward pos_e position_ids in
    let token_type_e = Layer.forward token_type_e token_type_ids in
    Layer.forward layer_norm Tensor.(word_e + pos_e + token_type_e)
    |> Tensor.dropout ~p:config.hidden_dropout_p ~is_training

let self_attention vs (config : Config.t) =
  let { Config.hidden_size; num_attention_heads; attention_probs_dropout_p; _ } =
    config
  in
  let linear n = Layer.linear Var_store.(vs / n) ~input_dim:hidden_size hidden_size in
  let query = linear "query" in
  let key = linear "key" in
  let value = linear "value" in
  let w =
    Var_store.new_var
      Var_store.(vs / "dense")
      ~shape:[ hidden_size; hidden_size ]
      ~init:Zeros
      ~name:"weight"
  in
  let bs =
    Var_store.new_var
      Var_store.(vs / "dense")
      ~shape:[ hidden_size ]
      ~init:Zeros
      ~name:"bias"
  in
  let layer_norm =
    Layer.layer_norm Var_store.(vs / "LayerNorm") config.hidden_size ~eps:1e-12
  in
  let dim_per_head = hidden_size / num_attention_heads in
  let split_heads xs ~batch_size =
    Tensor.view xs ~size:[ batch_size; -1; num_attention_heads; dim_per_head ]
    |> Tensor.transpose ~dim0:1 ~dim1:2
  in
  fun xs ~mask ~is_training ->
    let batch_size = Tensor.size xs |> List.hd_exn in
    let key =
      Layer.forward key xs
      |> split_heads ~batch_size
      |> Tensor.transpose ~dim0:(-1) ~dim1:(-2)
    in
    let value = Layer.forward value xs |> split_heads ~batch_size in
    let query = Layer.forward query xs |> split_heads ~batch_size in
    let scores = Tensor.matmul query key in
    let scores =
      match mask with
      | None -> scores
      | Some mask -> Tensor.(scores + mask)
    in
    let weights =
      Tensor.softmax scores ~dim:(-1) ~dtype:(T Float)
      |> Tensor.dropout ~p:attention_probs_dropout_p ~is_training
    in
    let context =
      Tensor.matmul weights value |> Tensor.transpose ~dim0:1 ~dim1:2 |> Tensor.contiguous
    in
    let context = Tensor.(einsum ~equation:"bfnd,ndh->bfh" [ context; w ] + bs) in
    let ys =
      Tensor.dropout context ~p:config.hidden_dropout_p ~is_training
      |> Layer.forward layer_norm
    in
    Tensor.(xs + ys)
