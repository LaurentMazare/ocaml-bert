open! Base

module Kind : sig
  type t =
    | Normal
    | Punctuation
    | Special
  [@@deriving sexp]
end

module With_id : sig
  type t =
    { token_id : int
    ; is_continuation : bool
    }
  [@@deriving sexp]
end

(* [start] and [stop] are offset in bytes and not in utf-8 chars.
   [start] is included but [stop] is not.  *)
type t =
  { text : string
  ; start : int
  ; stop : int
  ; with_id : With_id.t option
  ; kind : Kind.t
  }
[@@deriving sexp]

val add_offset : t list -> offset:int -> t list
