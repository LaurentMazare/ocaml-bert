type t

val create : tokens:string list -> bpe_ranks:(string * string) list -> t
val load : vocab_filename:string -> merge_filename:string -> t
val token_id : t -> string -> int option
val token : t -> int -> string
val bpe : t -> string -> int list
