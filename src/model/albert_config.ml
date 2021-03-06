open! Base

type t =
  { vocab_size : int
  ; embedding_size : int
  ; hidden_size : int
  ; intermediate_size : int
  ; max_position_embeddings : int
  ; num_attention_heads : int
  ; inner_group_num : int
  ; num_hidden_layers : int
  ; num_hidden_groups : int
  ; hidden_dropout_p : float
  ; attention_probs_dropout_p : float
  ; classifier_dropout_p : float
  ; hidden_act : [ `gelu_new | `gelu | `relu | `mish ]
  }
[@@deriving sexp]

let base =
  { vocab_size = 30000
  ; embedding_size = 128
  ; hidden_size = 768
  ; intermediate_size = 3072
  ; max_position_embeddings = 512
  ; num_attention_heads = 12
  ; inner_group_num = 1
  ; num_hidden_layers = 12
  ; num_hidden_groups = 1
  ; hidden_dropout_p = 0.
  ; attention_probs_dropout_p = 0.1
  ; classifier_dropout_p = 0.1
  ; hidden_act = `gelu_new
  }
  [@@deriving sexp]
