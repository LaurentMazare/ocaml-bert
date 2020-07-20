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
  ; hidden_act : [ `gelu_new | `gelu | `relu | `mish ]
  }
[@@deriving sexp]
