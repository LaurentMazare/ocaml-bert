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

let base =
  { vocab_size = 30522
  ; hidden_size = 768
  ; intermediate_size = 3072
  ; num_hidden_layers = 12
  ; max_position_embeddings = 512
  ; hidden_dropout_p = 0.1
  ; attention_dropout_p = 0.1
  ; num_attention_heads = 12
  ; is_decoder = false
  ; hidden_act = `gelu
  }
