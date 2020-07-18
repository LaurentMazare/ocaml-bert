(* DistilBERT model
   https://medium.com/huggingface/distilbert-8cf3380435b5
*)
open Base
open Torch

let gelu xs =
  let erf = Tensor.erf (Tensor.div1 xs (Scalar.f (Float.sqrt 2.))) in
  Tensor.mul1 (Tensor.add1 erf (Scalar.f 1.)) (Scalar.f 0.5) |> Tensor.mul xs

module Attention = struct
  let multihead vs (config : Config.t) =
    let dim_per_head = config.dim / config.n_heads in
    let split_heads xs ~batch_size =
      Tensor.view xs ~size:[ batch_size; -1; config.n_heads; dim_per_head ]
      |> Tensor.transpose ~dim0:1 ~dim1:2
    in
    let linear name_ =
      Layer.linear Var_store.(vs / name_) ~input_dim:config.dim config.dim
    in
    let sqrt_dim_per_head = Float.of_int dim_per_head |> Float.sqrt in
    let q_lin = linear "q_lin" in
    let k_lin = linear "k_lin" in
    let v_lin = linear "v_lin" in
    let out_lin = linear "out_lin" in
    fun ~query ~key ~value ~mask ~is_training ->
      let batch_size, k_len =
        match Tensor.shape query with
        | batch_size :: k_len :: _ -> batch_size, k_len
        | shape ->
          let shape = String.concat ~sep:", " (List.map ~f:Int.to_string shape) in
          Printf.sprintf "unexpected shape <%s>" shape |> failwith
      in
      let q = Layer.forward q_lin query |> split_heads ~batch_size in
      let k = Layer.forward k_lin key |> split_heads ~batch_size in
      let v = Layer.forward v_lin value |> split_heads ~batch_size in
      let q = Tensor.div1 q (Scalar.float sqrt_dim_per_head) in
      let scores = Tensor.matmul q (Tensor.transpose k ~dim0:2 ~dim1:3) in
      let scores =
        match mask with
        | None -> scores
        | Some mask ->
          let mask =
            Tensor.le1 mask (Tensor.add1 (Tensor.zeros_like mask) (Scalar.f 0.1))
            |> Tensor.view ~size:[ batch_size; 1; 1; k_len ]
          in
          Tensor.masked_fill scores ~mask ~value:(Scalar.f Float.neg_infinity)
      in
      let weights =
        Tensor.softmax scores ~dim:(-1) ~dtype:(T Float)
        |> Tensor.dropout ~p:config.attention_dropout_p ~is_training
      in
      let context =
        Tensor.matmul weights v
        |> Tensor.transpose ~dim0:1 ~dim1:2
        |> Tensor.contiguous
        |> Tensor.view ~size:[ batch_size; -1; config.n_heads * dim_per_head ]
        |> Layer.forward out_lin
      in
      context, weights
end

let sinusoidal_position_embeddings ~device ~dim ~max_len =
  let position = Tensor.arange ~end_:(Scalar.i max_len) ~options:(T Float, Cpu) in
  let div_term i =
    Float.exp (-2. *. Float.of_int i *. Float.log 10000.0 /. Float.of_int dim)
  in
  let sin i = Tensor.(sin (position * f (div_term i))) in
  let cos i = Tensor.(cos (position * f (div_term i))) in
  let weight =
    List.init dim ~f:(fun i -> if i % 2 = 0 then sin (i / 2) else cos (i / 2))
    |> Tensor.stack ~dim:1
    |> Tensor.unsqueeze ~dim:0
    |> Tensor.to_device ~device
    |> Tensor.detach
  in
  Layer.of_fn (fun indices ->
      Tensor.embedding
        ~weight
        ~indices
        ~padding_idx:(-1)
        ~sparse:false
        ~scale_grad_by_freq:false)

let layer_norm vs dim =
  let weight = Var_store.new_var vs ~name:"weight" ~shape:[ dim ] ~init:Ones in
  let bias = Var_store.new_var vs ~name:"bias" ~shape:[ dim ] ~init:Zeros in
  Layer.of_fn (fun xs ->
      Tensor.layer_norm
        xs
        ~normalized_shape:[ dim ]
        ~weight:(Some weight)
        ~bias:(Some bias)
        ~eps:1e-12
        ~cudnn_enable:false)

let embeddings vs (config : Config.t) =
  let word_e =
    Layer.embeddings
      Var_store.(vs / "word_embeddings")
      ~num_embeddings:config.vocab_size
      ~embedding_dim:config.dim
  in
  let pos_e =
    if config.use_sinusoidal_position_embeddings
    then
      sinusoidal_position_embeddings
        ~dim:config.dim
        ~device:(Var_store.device vs)
        ~max_len:config.max_position_embeddings
    else
      Layer.embeddings
        Var_store.(vs / "position_embeddings")
        ~num_embeddings:config.max_position_embeddings
        ~embedding_dim:config.dim
  in
  let layer_norm_ = layer_norm Var_store.(vs / "LayerNorm") config.dim in
  Layer.of_fn_
    Tensor.(
      fun xs ~is_training ->
        let seq_len = Tensor.shape xs |> List.last_exn in
        let pos_ids =
          Tensor.arange ~end_:(Scalar.i seq_len) ~options:(T Int64, Var_store.device vs)
        in
        let word_e = Layer.forward word_e xs in
        let pos_e = Layer.forward pos_e pos_ids in
        Layer.forward layer_norm_ (word_e + pos_e)
        |> Tensor.dropout ~p:config.dropout_p ~is_training)

let feed_forward vs (config : Config.t) =
  let lin1 =
    Layer.linear Var_store.(vs / "lin1") ~input_dim:config.dim config.hidden_dim
  in
  let lin2 =
    Layer.linear Var_store.(vs / "lin2") ~input_dim:config.dim config.hidden_dim
  in
  let activation =
    match config.activation with
    | `relu -> Tensor.relu
    | `gelu -> gelu
  in
  Layer.of_fn_ (fun xs ~is_training ->
      Layer.forward lin1 xs
      |> activation
      |> Layer.forward lin2
      |> Tensor.dropout ~p:config.dropout_p ~is_training)

let transformer_block vs (config : Config.t) =
  let attention = Attention.multihead Var_store.(vs / "attention") config in
  let sa_layer_norm = layer_norm Var_store.(vs / "sa_layer_norm") config.dim in
  let output_layer_norm = layer_norm Var_store.(vs / "output_layer_norm") config.dim in
  let ffn = feed_forward Var_store.(vs / "ffn") config in
  fun xs ~mask ~is_training ->
    let output, sa_weights = attention ~query:xs ~key:xs ~value:xs ~mask ~is_training in
    let output = Tensor.(xs + output) |> Layer.forward sa_layer_norm in
    let output =
      Tensor.(output + Layer.forward_ ffn output ~is_training)
      |> Layer.forward output_layer_norm
    in
    output, sa_weights

let transformer vs (config : Config.t) =
  let vs = Var_store.(vs / "layer") in
  let blocks =
    List.init config.n_layers ~f:(fun i ->
        transformer_block Var_store.(vs / Int.to_string i) config)
  in
  fun xs ~mask ~is_training ->
    List.fold_map blocks ~init:xs ~f:(fun xs block ->
        let next_xs, attns = block xs ~mask ~is_training in
        next_xs, (attns, xs))

module With_mask = struct
  let model vs (config : Config.t) =
    let vs = Var_store.(vs / "distilbert") in
    let embeddings = embeddings Var_store.(vs / "embeddings") config in
    let transformer = transformer Var_store.(vs / "transformer") config in
    fun xs ~mask ~is_training ->
      Layer.forward_ embeddings xs ~is_training |> transformer ~mask ~is_training

  let masked_lm vs (config : Config.t) =
    let model = model vs config in
    let vocab_transform =
      Layer.linear Var_store.(vs / "vocab_transform") ~input_dim:config.dim config.dim
    in
    let vocab_layer_norm =
      Layer.layer_norm Var_store.(vs / "vocab_layer_norm") ~eps:1e-12 config.dim
    in
    let vocab_projector =
      Layer.linear
        Var_store.(vs / "vocab_projector")
        ~input_dim:config.dim
        config.vocab_size
    in
    fun xs ~mask ~is_training ->
      model xs ~mask ~is_training
      |> fst
      |> Layer.forward vocab_transform
      |> gelu
      |> Layer.forward vocab_layer_norm
      |> Layer.forward vocab_projector

  let classifier vs (config : Config.t) ~num_labels ~classifier_dropout_p =
    let model = model vs config in
    let pre_classifier =
      Layer.linear Var_store.(vs / "pre_classifier") ~input_dim:config.dim config.dim
    in
    let classifier =
      Layer.linear Var_store.(vs / "classifier") ~input_dim:config.dim num_labels
    in
    fun xs ~mask ~is_training ->
      model xs ~mask ~is_training
      |> fst
      |> Tensor.select ~dim:1 ~index:0
      |> Layer.forward pre_classifier
      |> Tensor.relu
      |> Tensor.dropout ~p:classifier_dropout_p ~is_training
      |> Layer.forward classifier

  let question_answering vs (config : Config.t) ~qa_dropout_p =
    let model = model vs config in
    let qa_outputs =
      Layer.linear Var_store.(vs / "qua_outputs") ~input_dim:config.dim 2
    in
    fun xs ~mask ~is_training ->
      let logits =
        model xs ~mask ~is_training
        |> fst
        |> Tensor.dropout ~p:qa_dropout_p ~is_training
        |> Tensor.select ~dim:1 ~index:0
        |> Layer.forward qa_outputs
        |> Tensor.split ~split_size:1 ~dim:(-1)
      in
      match logits with
      | [ start_logits; end_logits ] ->
        let start_logits = Tensor.squeeze1 start_logits ~dim:(-1) in
        let end_logits = Tensor.squeeze1 end_logits ~dim:(-1) in
        start_logits, end_logits
      | _ -> assert false
end

let model vs config =
  let model = With_mask.model vs config in
  Layer.of_fn_ (fun xs ~is_training -> model xs ~mask:None ~is_training |> fst)

let masked_lm vs config =
  let masked_lm = With_mask.masked_lm vs config in
  Layer.of_fn_ (fun xs ~is_training -> masked_lm xs ~mask:None ~is_training)

let classifier vs config ~num_labels ~classifier_dropout_p =
  let classifier = With_mask.classifier vs config ~num_labels ~classifier_dropout_p in
  Layer.of_fn_ (fun xs ~is_training -> classifier xs ~mask:None ~is_training)

let question_answering vs config ~qa_dropout_p =
  let qa = With_mask.question_answering vs config ~qa_dropout_p in
  fun xs ~is_training -> qa xs ~mask:None ~is_training

let%expect_test "tensor" =
  let t = Tensor.of_int0 42 in
  Stdio.printf "> %d\n%!" (Tensor.to_int0_exn t);
  [%expect {| > 42 |}]
