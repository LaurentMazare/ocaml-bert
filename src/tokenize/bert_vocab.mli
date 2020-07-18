type t

val create : tokens:string list -> t
val load : filename:string -> t
val token_id : t -> string -> int option
val token : t -> int -> string
val wordpiece : t -> string -> kind:Token.Kind.t -> Token.t list
val cls_token : t -> pos:int -> Token.t option
val sep_token : t -> pos:int -> Token.t option

module Test : sig
  val create : unit -> t
end
