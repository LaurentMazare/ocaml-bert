(* GPT2 model *)
open Base
open Torch
module Config = Gpt2_config

let conv1d vs ~nf ~nx =
  let weight =
    Var_store.new_var
      vs
      ~shape:[ nx; nf ]
      ~init:(Normal { mean = 0.; stdev = 0.02 })
      ~name:"weight"
  in
  let bias = Var_store.new_var vs ~shape:[ nf ] ~init:Zeros ~name:"bias" in
  Layer.of_fn (fun xs -> Tensor.(matmul xs weight + bias))

module Attention = struct
  let attn vs config ~scale =
    let { Config.n_ctx; n_embd; n_head; attn_p; resid_p; _ } = config in
    let bias =
      Tensor.ones [ n_ctx; n_ctx ] ~kind:(T Float)
      |> Tensor.tril ~diagonal:0
      |> Tensor.view ~size:[ 1; 1; n_ctx; n_ctx ]
    in
    let c_attn = conv1d Var_store.(vs / "c_attn") ~nf:(n_embd * 3) ~nx:n_embd in
    let c_proj = conv1d Var_store.(vs / "c_proj") ~nf:n_embd ~nx:n_embd in
    let dim_per_head = n_embd / n_head in
    let attention ~query ~key ~value ~attention_mask ~is_training =
      let v_shape = Tensor.size value in
      let w = Tensor.matmul query key in
      let w =
        if scale
        then
          Tensor.div1 w (List.last_exn v_shape |> Float.of_int |> Float.sqrt |> Scalar.f)
        else w
      in
      let nd, ns =
        match Tensor.size w with
        | _ :: _ :: nd :: ns :: _ -> nd, ns
        | _ -> failwith "unexpected number of dimensions"
      in
      let b =
        Tensor.narrow bias ~dim:2 ~start:(ns - nd) ~length:nd
        |> Tensor.narrow ~dim:3 ~start:0 ~length:nd
      in
      let w = Tensor.((w * b) + (f 1e4 * (b - f 1.))) in
      let w =
        match attention_mask with
        | None -> w
        | Some mask -> Tensor.(w + mask)
      in
      let w =
        Tensor.softmax w ~dim:(-1) ~dtype:(T Float)
        |> Tensor.dropout ~p:attn_p ~is_training
      in
      Tensor.matmul w value
    in
    let split_heads xs ~k =
      let batch_size = Tensor.size xs |> List.hd_exn in
      let dims = if k then [ 0; 2; 3; 1 ] else [ 0; 2; 1; 3 ] in
      Tensor.view xs ~size:[ batch_size; -1; n_head; dim_per_head ]
      |> Tensor.permute ~dims
    in
    fun xs ~layer_past ~attention_mask ~is_training ->
      let batch_size = Tensor.size xs |> List.hd_exn in
      let query, key, value =
        match Layer.forward c_attn xs |> Tensor.split ~split_size:n_embd ~dim:2 with
        | query :: key :: value :: _ ->
          split_heads query ~k:false, split_heads key ~k:true, split_heads value ~k:false
        | _ -> failwith "unexpected split size"
      in
      let key, value =
        match layer_past with
        | None -> key, value
        | Some p ->
          let key =
            Tensor.cat
              [ Tensor.get p 0 |> Tensor.transpose ~dim0:(-2) ~dim1:(-1); key ]
              ~dim:(-1)
          in
          let value = Tensor.cat [ Tensor.get p 1; value ] ~dim:(-2) in
          key, value
      in
      let present =
        Tensor.stack [ Tensor.transpose key ~dim0:(-2) ~dim1:(-1); value ] ~dim:0
      in
      let a =
        attention ~query ~key ~value ~attention_mask ~is_training
        |> Tensor.transpose ~dim0:1 ~dim1:2
        |> Tensor.contiguous
        |> Tensor.view ~size:[ batch_size; -1; n_head * dim_per_head ]
        |> Layer.forward c_proj
        |> Tensor.dropout ~p:resid_p ~is_training
      in
      a, present
end

let mlp vs config =
  let { Config.resid_p; n_embd; afn; _ } = config in
  let c_fc = conv1d Var_store.(vs / "c_fc") ~nf:(n_embd * 4) ~nx:n_embd in
  let c_proj = conv1d Var_store.(vs / "c_proj") ~nf:n_embd ~nx:(n_embd * 4) in
  let activation =
    match afn with
    | `swish -> Activation.swish
    | `relu -> Activation.relu
    | `gelu -> Activation.gelu_new
  in
  Layer.of_fn_ (fun xs ~is_training ->
      Layer.forward c_fc xs
      |> activation
      |> Layer.forward c_proj
      |> Tensor.dropout ~p:resid_p ~is_training)

let block vs config ~scale =
  let { Config.n_embd; layer_norm_eps; _ } = config in
  let ln1 = Layer.layer_norm Var_store.(vs / "ln_1") n_embd ~eps:layer_norm_eps in
  let ln2 = Layer.layer_norm Var_store.(vs / "ln_2") n_embd ~eps:layer_norm_eps in
  let attn = Attention.attn Var_store.(vs / "attn") config ~scale in
  let mlp = mlp Var_store.(vs / "mlp") config in
  fun xs ~layer_past ~attention_mask ~is_training ->
    let output, present =
      Layer.forward ln1 xs |> attn ~layer_past ~attention_mask ~is_training
    in
    let xs = Tensor.(xs + output) in
    let m = Layer.forward ln2 xs |> Layer.forward_ mlp ~is_training in
    Tensor.(xs + m), present

let model vs config =
  let vs = Var_store.(vs / "transformer") in
  let { Config.vocab_size; n_layer; n_positions; n_embd; layer_norm_eps; embd_p; _ } =
    config
  in
  let wte =
    Layer.embeddings
      Var_store.(vs / "wte")
      ~num_embeddings:vocab_size
      ~embedding_dim:n_embd
  in
  let wpe =
    Layer.embeddings
      Var_store.(vs / "wpe")
      ~num_embeddings:n_positions
      ~embedding_dim:n_embd
  in
  let ln_f = Layer.layer_norm Var_store.(vs / "ln_f") n_embd ~eps:layer_norm_eps in
  let vs_h = Var_store.(vs / "h") in
  let layers =
    List.init n_layer ~f:(fun layer_idx ->
        block Var_store.(vs_h / Int.to_string layer_idx) config ~scale:true)
  in
  fun input_ids ~layer_past ~attention_mask ~token_type_ids ~position_ids ~is_training ->
    let input_shape = Tensor.size input_ids in
    let seq_len = List.last_exn input_shape in
    let input_embeddings = Layer.forward wte input_ids in
    let layer_past_len =
      Option.value_map
        layer_past
        ~f:(fun (`layer layer_past) -> List.nth_exn (Tensor.size layer_past.(0)) 3)
        ~default:0
    in
    let position_ids =
      match position_ids with
      | Some position_ids -> position_ids
      | None ->
        Tensor.arange1
          ~start:(Scalar.i layer_past_len)
          ~end_:(Scalar.i (seq_len + layer_past_len))
          ~options:(T Int64, Var_store.device vs)
    in
    let attention_mask =
      Option.map attention_mask ~f:(fun attention_mask ->
          let attention_mask =
            Tensor.flatten attention_mask
            |> Tensor.unsqueeze ~dim:1
            |> Tensor.unsqueeze ~dim:2
          in
          Tensor.((attention_mask - f 1.) * f 1e4))
    in
    let position_embeddings = Layer.forward wpe position_ids in
    let token_type_embeddings =
      match token_type_ids with
      | None -> Tensor.zeros_like position_embeddings
      | Some token_type_ids -> Layer.forward wte token_type_ids
    in
    let hidden_state =
      Tensor.(input_embeddings + position_embeddings + token_type_embeddings)
      |> Tensor.dropout ~p:embd_p ~is_training
    in
    let hidden_state, presents =
      List.fold_mapi layers ~init:hidden_state ~f:(fun idx acc layer ->
          let layer_past =
            Option.map layer_past ~f:(fun (`layer layer_past) -> layer_past.(idx))
          in
          layer acc ~layer_past ~attention_mask ~is_training)
    in
    let output = Layer.forward ln_f hidden_state in
    output, `layer (Array.of_list presents)

let lm_model vs config =
  let { Config.n_embd; vocab_size; _ } = config in
  let transformer = model vs config in
  let lm_head =
    Layer.linear Var_store.(vs / "lm_head") ~use_bias:false ~input_dim:n_embd vocab_size
  in
  fun input_ids ~layer_past ~attention_mask ~token_type_ids ~position_ids ~is_training ->
    let output, presents =
      transformer
        input_ids
        ~layer_past
        ~attention_mask
        ~token_type_ids
        ~position_ids
        ~is_training
    in
    Layer.forward lm_head output, presents
