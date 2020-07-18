open! Base

module Kind = struct
  type t =
    | Normal
    | Punctuation
    | Special
  [@@deriving sexp]
end

module With_id = struct
  type t =
    { token_id : int
    ; is_continuation : bool
    }
  [@@deriving sexp]
end

type t =
  { text : string
  ; start : int
  ; stop : int
  ; with_id : With_id.t option
  ; kind : Kind.t
  }
[@@deriving sexp]

let add_offset ts ~offset =
  List.map ts ~f:(fun t -> { t with start = t.start + offset; stop = t.stop + offset })
