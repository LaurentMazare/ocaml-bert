open! Base
open Torch
module Config = Bert_config

val model
  :  Var_store.t
  -> Config.t
  -> input_ids:Tensor.t
  -> mask:Tensor.t option
  -> encoder_hidden_states:Tensor.t option
  -> encoder_mask:Tensor.t option
  -> is_training:bool
  -> Tensor.t

val masked_lm
  :  Var_store.t
  -> Config.t
  -> input_ids:Tensor.t
  -> mask:Tensor.t option
  -> encoder_hidden_states:Tensor.t option
  -> encoder_mask:Tensor.t option
  -> is_training:bool
  -> Tensor.t
