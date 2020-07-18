open! Base

type t =
  { vocab_size : int
  ; hidden_size : int
  ; intermediate_size : int
  ; num_hidden_layers : int
  ; max_position_embeddings : int
  ; hidden_dropout_p : float
  ; attention_dropout_p : float
  ; num_attention_heads : int
  ; is_decoder : bool
  ; hidden_act : [ `gelu | `relu | `mish ]
  }
[@@deriving sexp]
