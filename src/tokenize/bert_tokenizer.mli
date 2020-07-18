open! Base

type t

val create : Bert_vocab.t -> lower_case:bool -> t
val tokenize : t -> ?include_special_characters:bool -> string -> Token.t list
