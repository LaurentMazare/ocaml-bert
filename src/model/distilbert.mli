open Torch
module Config = Distilbert_config

module With_mask : sig
  val model
    :  Var_store.t
    -> Config.t
    -> Tensor.t
    -> mask:Tensor.t option
    -> is_training:bool
    -> Tensor.t * (Tensor.t * Tensor.t) list

  val masked_lm
    :  Var_store.t
    -> Config.t
    -> Tensor.t
    -> mask:Tensor.t option
    -> is_training:bool
    -> Tensor.t

  val classifier
    :  Var_store.t
    -> Config.t
    -> num_labels:int
    -> classifier_dropout_p:float
    -> Tensor.t
    -> mask:Tensor.t option
    -> is_training:bool
    -> Tensor.t
end

val model : Var_store.t -> Config.t -> Layer.t_with_training
val masked_lm : Var_store.t -> Config.t -> Layer.t_with_training

val classifier
  :  Var_store.t
  -> Config.t
  -> num_labels:int
  -> classifier_dropout_p:float
  -> Layer.t_with_training

val question_answering
  :  Var_store.t
  -> Config.t
  -> qa_dropout_p:float
  -> Tensor.t
  -> is_training:bool
  -> Tensor.t * Tensor.t
