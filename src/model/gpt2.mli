open! Base
open! Torch
module Config = Gpt2_config

val model
  :  Var_store.t
  -> Config.t
  -> Tensor.t
  -> layer_past:[ `layer of Tensor.t array ] option
  -> attention_mask:Tensor.t option
  -> token_type_ids:Tensor.t option
  -> position_ids:Tensor.t option
  -> is_training:bool
  -> Tensor.t * [ `layer of Tensor.t array ]

val lm_model
  :  Var_store.t
  -> Config.t
  -> Tensor.t
  -> layer_past:[ `layer of Tensor.t array ] option
  -> attention_mask:Tensor.t option
  -> token_type_ids:Tensor.t option
  -> position_ids:Tensor.t option
  -> is_training:bool
  -> Tensor.t * [ `layer of Tensor.t array ]
