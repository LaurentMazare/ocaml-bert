type t

val create : Gpt2_vocab.t -> lower_case:bool -> t
val tokenize : t -> string -> int list
