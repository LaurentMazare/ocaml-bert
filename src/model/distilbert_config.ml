open! Base

type t =
  { vocab_size : int
  ; dim : int
  ; hidden_dim : int
  ; max_position_embeddings : int
  ; use_sinusoidal_position_embeddings : bool
  ; dropout_p : float
  ; n_heads : int
  ; n_layers : int
  ; attention_dropout_p : float
  ; activation : [ `relu | `gelu ]
  }
[@@deriving sexp]

let base =
  { activation = `gelu
  ; attention_dropout_p = 0.1
  ; dim = 768
  ; dropout_p = 0.1
  ; hidden_dim = 3072
  ; max_position_embeddings = 512
  ; n_heads = 12
  ; n_layers = 6
  ; use_sinusoidal_position_embeddings = false
  ; vocab_size = 30522
  }
