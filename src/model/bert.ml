open Base
open Torch
module Config = Bert_config

let embeddings vs (config : Config.t) =
  let word_e =
    Layer.embeddings
      Var_store.(vs / "word_embeddings")
      ~num_embeddings:config.vocab_size
      ~embedding_dim:config.hidden_size
  in
  let pos_e =
    Layer.embeddings
      Var_store.(vs / "position_embeddings")
      ~num_embeddings:config.max_position_embeddings
      ~embedding_dim:config.hidden_size
  in
  let token_type_e =
    Layer.embeddings
      Var_store.(vs / "token_type_embeddings")
      ~num_embeddings:config.vocab_size
      ~embedding_dim:config.hidden_size
  in
  let layer_norm =
    Layer.layer_norm Var_store.(vs / "LayerNorm") config.hidden_size ~eps:1e-12
  in
  Layer.of_fn_ (fun xs ~is_training ->
      let seq_len = Tensor.shape xs |> List.last_exn in
      let pos_ids =
        Tensor.arange ~end_:(Scalar.i seq_len) ~options:(T Int64, Var_store.device vs)
      in
      let token_type_ids = Tensor.zeros_like xs in
      let word_e = Layer.forward word_e xs in
      let pos_e = Layer.forward pos_e pos_ids in
      let token_type_e = Layer.forward token_type_e token_type_ids in
      Layer.forward layer_norm Tensor.(word_e + pos_e + token_type_e)
      |> Tensor.dropout ~p:config.hidden_dropout_p ~is_training)

let self_attention vs (config : Config.t) =
  let linear name_ =
    Layer.linear Var_store.(vs / name_) ~input_dim:config.hidden_size config.hidden_size
  in
  let query = linear "query" in
  let key = linear "key" in
  let value = linear "value" in
  let attention_head_size = config.hidden_size / config.num_attention_heads in
  let sqrt_attention_head_size = Float.of_int attention_head_size |> Float.sqrt in
  let split_heads xs ~batch_size =
    Tensor.view
      xs
      ~size:[ batch_size; -1; config.num_attention_heads; attention_head_size ]
    |> Tensor.transpose ~dim0:1 ~dim1:2
  in
  fun ~hidden_states ~mask ~encoder_hidden_states ~encoder_mask ~is_training ->
    let batch_size = Tensor.size hidden_states |> List.hd_exn in
    let key, value, mask =
      match encoder_hidden_states with
      | Some xs -> Layer.forward key xs, Layer.forward value xs, encoder_mask
      | None -> Layer.forward key hidden_states, Layer.forward value hidden_states, mask
    in
    let query = Layer.forward query hidden_states |> split_heads ~batch_size in
    let query = Tensor.div1 query (Scalar.f sqrt_attention_head_size) in
    let key = split_heads key ~batch_size in
    let value = split_heads value ~batch_size in
    let scores = Tensor.matmul query (Tensor.transpose key ~dim0:(-1) ~dim1:(-2)) in
    let scores =
      match mask with
      | Some mask -> Tensor.( + ) scores mask
      | None -> scores
    in
    let weights =
      Tensor.softmax scores ~dim:(-1) ~dtype:(T Float)
      |> Tensor.dropout ~p:config.attention_dropout_p ~is_training
    in
    Tensor.matmul weights value
    |> Tensor.transpose ~dim0:1 ~dim1:2
    |> Tensor.contiguous
    |> Tensor.view
         ~size:[ batch_size; -1; config.num_attention_heads * attention_head_size ]

let output vs (config : Config.t) ~input_dim =
  let dense = Layer.linear Var_store.(vs / "dense") ~input_dim config.hidden_size in
  let layer_norm =
    Layer.layer_norm Var_store.(vs / "LayerNorm") config.hidden_size ~eps:1e-12
  in
  fun hidden_states input_tensor ~is_training ->
    Layer.forward dense hidden_states
    |> Tensor.dropout ~p:config.hidden_dropout_p ~is_training
    |> Tensor.add input_tensor
    |> Layer.forward layer_norm

let activation = function
  | `gelu -> Activation.gelu
  | `relu -> Activation.relu
  | `mish -> Activation.mish

let bert_intermediate vs (config : Config.t) =
  let dense =
    Layer.linear
      Var_store.(vs / "dense")
      ~input_dim:config.hidden_size
      config.intermediate_size
  in
  Layer.of_fn (fun xs -> Layer.forward dense xs |> activation config.hidden_act)

let self_output vs config = output vs config ~input_dim:config.Config.hidden_size
let bert_output vs config = output vs config ~input_dim:config.Config.intermediate_size

let bert_attention vs (config : Config.t) =
  let self_attention = self_attention Var_store.(vs / "self") config in
  let self_output = self_output Var_store.(vs / "output") config in
  fun ~hidden_states ~mask ~encoder_hidden_states ~encoder_mask ~is_training ->
    let self_attention =
      self_attention
        ~hidden_states
        ~mask
        ~encoder_hidden_states
        ~encoder_mask
        ~is_training
    in
    self_output self_attention hidden_states ~is_training

let bert_pooler vs (config : Config.t) =
  let dense =
    Layer.linear Var_store.(vs / "dense") ~input_dim:config.hidden_size config.hidden_size
  in
  Layer.of_fn (fun xs ->
      Tensor.select xs ~dim:1 ~index:0 |> Layer.forward dense |> Tensor.tanh)

let bert_layer vs (config : Config.t) =
  let cross_attention =
    if config.is_decoder
    then (
      let cross_attention = bert_attention Var_store.(vs / "cross_attention") config in
      Some cross_attention)
    else None
  in
  let bert_attention = bert_attention Var_store.(vs / "attention") config in
  let bert_intermediate = bert_intermediate Var_store.(vs / "intermediate") config in
  let bert_output = bert_output Var_store.(vs / "output") config in
  fun ~hidden_states ~mask ~encoder_hidden_states ~encoder_mask ~is_training ->
    let attention_output =
      bert_attention
        ~hidden_states
        ~mask
        ~encoder_hidden_states:None
        ~encoder_mask:None
        ~is_training
    in
    let attention_output =
      match cross_attention with
      | Some cross_attention ->
        cross_attention
          ~hidden_states:attention_output
          ~mask
          ~encoder_hidden_states
          ~encoder_mask
          ~is_training
      | None -> attention_output
    in
    let output = Layer.forward bert_intermediate attention_output in
    bert_output output attention_output ~is_training

let bert_encoder vs (config : Config.t) =
  let vs = Var_store.(vs / "layer") in
  let layers =
    List.init config.num_hidden_layers ~f:(fun i ->
        bert_layer Var_store.(vs / Int.to_string i) config)
  in
  fun ~hidden_states ~mask ~encoder_hidden_states ~encoder_mask ~is_training ->
    List.fold layers ~init:hidden_states ~f:(fun hidden_states layer ->
        layer ~hidden_states ~mask ~encoder_hidden_states ~encoder_mask ~is_training)

let model vs (config : Config.t) =
  let embeddings = embeddings Var_store.(vs / "embeddings") config in
  let encoder = bert_encoder Var_store.(vs / "encoder") config in
  let pooler = bert_pooler Var_store.(vs / "pooler") config in
  fun ~input_ids ~mask ~encoder_hidden_states ~encoder_mask ~is_training ->
    let input_shape = Tensor.size input_ids in
    let batch_size, seq_len =
      match input_shape with
      | batch_size :: seq_len :: _ -> batch_size, seq_len
      | _ -> assert false
    in
    let mask =
      match mask with
      | None -> Tensor.ones_like input_ids
      | Some mask -> mask
    in
    let mask =
      match Tensor.size mask |> List.length with
      | 3 -> Tensor.unsqueeze mask ~dim:1
      | 2 ->
        if config.is_decoder
        then (
          let seq_ids =
            Tensor.arange ~end_:(Scalar.i seq_len) ~options:(T Float, Var_store.device vs)
          in
          let causal_mask =
            Tensor.unsqueeze seq_ids ~dim:0
            |> Tensor.unsqueeze ~dim:0
            |> Tensor.repeat ~repeats:[ batch_size; seq_len; 1 ]
          in
          let causal_mask =
            Tensor.le1
              causal_mask
              (Tensor.unsqueeze seq_ids ~dim:0 |> Tensor.unsqueeze ~dim:(-1))
          in
          Tensor.mul causal_mask (Tensor.unsqueeze mask ~dim:1 |> Tensor.unsqueeze ~dim:1))
        else Tensor.unsqueeze mask ~dim:1 |> Tensor.unsqueeze ~dim:1
      | d -> Printf.sprintf "unexpected mask dimension %d" d |> failwith
    in
    let mask = Tensor.mul1 Tensor.(ones_like mask - mask) (Scalar.f (-10000.0)) in
    let encoder_mask =
      match config.is_decoder, encoder_hidden_states with
      | true, Some encoder_hidden_states ->
        let s0, s1 =
          match Tensor.size encoder_hidden_states with
          | s0 :: s1 :: _ -> s0, s1
          | _ -> assert false
        in
        let encoder_mask =
          match encoder_mask with
          | Some encoder_mask -> encoder_mask
          | None -> Tensor.ones [ s0; s1 ] ~kind:(T Int64)
        in
        (match Tensor.size encoder_mask |> List.length with
        | 2 -> Some (Tensor.unsqueeze encoder_mask ~dim:1 |> Tensor.unsqueeze ~dim:1)
        | 3 -> Some (Tensor.unsqueeze encoder_mask ~dim:1)
        | d -> Printf.sprintf "unexpected encoder mask dimension %d" d |> failwith)
      | _ -> None
    in
    let embedding_output = Layer.forward_ embeddings input_ids ~is_training in
    let hidden_states =
      encoder
        ~hidden_states:embedding_output
        ~mask:(Some mask)
        ~encoder_hidden_states
        ~encoder_mask
        ~is_training
    in
    hidden_states, Layer.forward pooler hidden_states

let prediction_head_transform vs (config : Config.t) =
  let dense =
    Layer.linear Var_store.(vs / "dense") ~input_dim:config.hidden_size config.hidden_size
  in
  let layer_norm =
    Layer.layer_norm Var_store.(vs / "LayerNorm") config.hidden_size ~eps:1e-12
  in
  Layer.of_fn (fun xs ->
      Layer.forward dense xs |> activation config.hidden_act |> Layer.forward layer_norm)

let lm_prediction_head vs (config : Config.t) =
  let vs = Var_store.(vs / "predictions") in
  let transform = prediction_head_transform Var_store.(vs / "transform") config in
  let decoder =
    Layer.linear
      Var_store.(vs / "decoder")
      ~input_dim:config.hidden_size
      config.vocab_size
      ~use_bias:false
  in
  let bias = Var_store.new_var vs ~name:"bias" ~shape:[ config.vocab_size ] ~init:Zeros in
  Layer.of_fn (fun xs ->
      Layer.forward transform xs |> Layer.forward decoder |> fun xs -> Tensor.(xs + bias))

let masked_lm vs (config : Config.t) =
  let model = model Var_store.(vs / "bert") config in
  let lm_prediction_head = lm_prediction_head Var_store.(vs / "cls") config in
  fun ~input_ids ~mask ~encoder_hidden_states ~encoder_mask ~is_training ->
    model ~input_ids ~mask ~encoder_hidden_states ~encoder_mask ~is_training
    |> fst
    |> Layer.forward lm_prediction_head
