open! Base
open! Torch
module Config = Albert_config

val model
  :  Var_store.t
  -> Config.t
  -> Tensor.t
  -> mask:Tensor.t option
  -> token_type_ids:Tensor.t option
  -> position_ids:Tensor.t option
  -> is_training:bool
  -> Tensor.t * Tensor.t

val masked_lm
  :  Var_store.t
  -> Config.t
  -> Tensor.t
  -> mask:Tensor.t option
  -> token_type_ids:Tensor.t option
  -> position_ids:Tensor.t option
  -> is_training:bool
  -> Tensor.t
