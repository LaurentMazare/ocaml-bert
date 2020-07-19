open! Base

type t =
  { vocab_size : int
  ; n_positions : int
  ; n_ctx : int
  ; n_embd : int
  ; n_head : int
  ; n_layer : int
  ; attn_p : float
  ; embd_p : float
  ; resid_p : float
  ; afn : [ `relu | `gelu | `swish ]
  ; layer_norm_eps : float
  }
[@@deriving sexp]

let distilgpt2 =
  { vocab_size = 50257
  ; n_positions = 1024
  ; n_ctx = 1024
  ; n_embd = 768
  ; n_head = 12
  ; n_layer = 6
  ; attn_p = 0.1
  ; embd_p = 0.1
  ; resid_p = 0.1
  ; afn = `gelu
  ; layer_norm_eps = 1e-5
  }
