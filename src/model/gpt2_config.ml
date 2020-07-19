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
