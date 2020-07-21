(** ALBERT: A Lite BERT
    http://ai.googleblog.com/2019/12/albert-lite-bert-for-self-supervised.html
    https://arxiv.org/abs/1909.11942

    Weights published under the Apache 2.0 license by Google and hosted by
    Huggingsface:
        https://cdn.huggingface.co/albert-base-v2/rust_model.ot
*)
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

let layer vs config =
  let { Config.hidden_size; hidden_act; intermediate_size; _ } = config in
  let attention = self_attention Var_store.(vs / "attention") config in
  let full_layer_norm =
    Layer.layer_norm Var_store.(vs / "full_layer_norm") hidden_size ~eps:1e-12
  in
  let ffn =
    Layer.linear Var_store.(vs / "ffn") ~input_dim:hidden_size intermediate_size
  in
  let ffn_output =
    Layer.linear Var_store.(vs / "ffn_output") ~input_dim:intermediate_size hidden_size
  in
  let activation =
    match hidden_act with
    | `gelu_new -> Activation.gelu_new
    | `gelu -> Activation.gelu
    | `relu -> Activation.relu
    | `mish -> Activation.mish
  in
  fun xs ~mask ~is_training ->
    let xs = attention xs ~mask ~is_training in
    let ys = Layer.forward ffn xs |> activation |> Layer.forward ffn_output in
    Tensor.(xs + ys) |> Layer.forward full_layer_norm

let layer_group vs (config : Config.t) =
  let vs = Var_store.(vs / "albert_layers") in
  let layers =
    List.init config.inner_group_num ~f:(fun i ->
        layer Var_store.(vs / Int.to_string i) config)
  in
  fun xs ~mask ~is_training ->
    List.fold layers ~init:xs ~f:(fun acc layer -> layer acc ~mask ~is_training)

let transformer vs (config : Config.t) =
  let vs = Var_store.(vs / "albert_layer_groups") in
  let embedding_hidden_mapping_in =
    Layer.linear
      Var_store.(vs / "embedding_hidden_mapping_in")
      ~input_dim:config.embedding_size
      config.hidden_size
  in
  let layers =
    List.init config.inner_group_num ~f:(fun i ->
        layer_group Var_store.(vs / Int.to_string i) config)
    |> Array.of_list
  in
  fun xs ~mask ~is_training ->
    let xs = Layer.forward embedding_hidden_mapping_in xs in
    List.init config.num_hidden_layers ~f:Fn.id
    |> List.fold ~init:xs ~f:(fun xs i ->
           let group_idx = i / (config.num_hidden_layers / config.num_hidden_groups) in
           layers.(group_idx) xs ~mask ~is_training)

let model vs (config : Config.t) =
  let embeddings = embeddings Var_store.(vs / "embeddings") config in
  let encoder = transformer Var_store.(vs / "encoder") config in
  let pooler =
    Layer.linear
      Var_store.(vs / "pooler")
      ~input_dim:config.hidden_size
      config.hidden_size
  in
  fun input_ids ~mask ~token_type_ids ~position_ids ~is_training ->
    let mask =
      match mask with
      | Some mask -> mask
      | None -> Tensor.ones_like input_ids
    in
    let mask = Tensor.unsqueeze mask ~dim:1 |> Tensor.unsqueeze ~dim:2 in
    let mask = Tensor.((f 1.0 - mask) * f (-10000.)) in
    let hidden_state =
      embeddings input_ids ~token_type_ids ~position_ids ~is_training
      |> encoder ~mask:(Some mask) ~is_training
    in
    let pooled_output =
      Tensor.select hidden_state ~dim:1 ~index:0 |> Layer.forward pooler |> Tensor.tanh
    in
    hidden_state, pooled_output

let masked_lm_head vs config =
  let { Config.embedding_size; hidden_size; vocab_size; hidden_act; _ } = config in
  let layer_norm =
    Layer.layer_norm Var_store.(vs / "LayerNorm") embedding_size ~eps:1e-12
  in
  let dense =
    Layer.linear Var_store.(vs / "dense") ~input_dim:hidden_size embedding_size
  in
  let decoder =
    Layer.linear Var_store.(vs / "decoder") ~input_dim:embedding_size vocab_size
  in
  let activation =
    match hidden_act with
    | `gelu_new -> Activation.gelu_new
    | `gelu -> Activation.gelu
    | `relu -> Activation.relu
    | `mish -> Activation.mish
  in
  Layer.of_fn (fun xs ->
      Layer.forward dense xs
      |> activation
      |> Layer.forward layer_norm
      |> Layer.forward decoder)

let masked_lm vs config =
  let model = model Var_store.(vs / "albert") config in
  let predictions = masked_lm_head Var_store.(vs / "predictions") config in
  fun input_ids ~mask ~token_type_ids ~position_ids ~is_training ->
    model input_ids ~mask ~token_type_ids ~position_ids ~is_training
    |> fst
    |> Layer.forward predictions
