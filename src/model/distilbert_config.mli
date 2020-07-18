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

val base : t
