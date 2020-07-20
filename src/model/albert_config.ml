open! Base

type t =
  { vocab_size : int
  ; embedding_size : int
  ; hidden_size : int
  ; max_position_embeddings : int
  ; num_attention_heads : int
  ; hidden_dropout_p : float
  ; attention_probs_dropout_p : float
  }
[@@deriving sexp]
