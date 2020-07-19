(* https://cdn.huggingface.co/bert-base-uncased-vocab.txt *)
open! Base

type t =
  { tokens : string array
  ; token_indices : (string, int, String.comparator_witness) Map.t
  ; unknown_token : string
  ; cls_token_id : int option
  ; sep_token_id : int option
  }

let create ~tokens =
  let token_indices =
    List.mapi tokens ~f:(fun idx token -> String.strip token, idx)
    |> Map.of_alist_exn (module String)
  in
  { tokens = Array.of_list tokens
  ; token_indices
  ; unknown_token = "[UNK]"
  ; cls_token_id = Map.find token_indices "[CLS]"
  ; sep_token_id = Map.find token_indices "[SEP]"
  }

let load ~filename =
  let tokens = Stdio.In_channel.with_file filename ~f:Stdio.In_channel.input_lines in
  create ~tokens

let token_id t token = Map.find t.token_indices token
let token t id = t.tokens.(id)

let special_token t ~token_id_option ~pos =
  Option.map token_id_option ~f:(fun token_id ->
      { Token.text = token t token_id
      ; start = pos
      ; stop = pos
      ; with_id = Some { token_id; is_continuation = false }
      ; kind = Special
      })

let cls_token t ~pos = special_token t ~token_id_option:t.cls_token_id ~pos
let sep_token t ~pos = special_token t ~token_id_option:t.sep_token_id ~pos

(* We assume that [t] only contains 'complete' utf-8 words. In this
   case we can do lookups for 'incomplete' words without it being an
   issue as these will end up with no match. *)
let wordpiece t str ~kind =
  if String.is_empty str
  then []
  else (
    let create_token ~with_id ~start ~stop =
      { Token.text = String.sub str ~pos:start ~len:(stop - start)
      ; start
      ; stop
      ; with_id
      ; kind
      }
    in
    let str_len = String.length str in
    let rec loop_start ~start =
      let rec loop_end ~stop =
        if start = stop
        then str_len, create_token ~with_id:None ~start ~stop:str_len
        else (
          let candidate = String.sub str ~pos:start ~len:(stop - start) in
          let candidate = if start = 0 then candidate else "##" ^ candidate in
          match Map.find t.token_indices candidate with
          | None -> loop_end ~stop:(stop - 1)
          | Some token_id ->
            let with_id = { Token.With_id.token_id; is_continuation = start <> 0 } in
            stop, create_token ~with_id:(Some with_id) ~start ~stop)
      in
      let start, token = loop_end ~stop:str_len in
      let tail = if start >= str_len then [] else loop_start ~start in
      token :: tail
    in
    loop_start ~start:0)

module Test = struct
  let create () =
    create
      ~tokens:[ "hello"; "world"; "una"; "##ffa"; "##ble"; "##bl"; "##e"; "!"; "##!" ]
end

let%expect_test "wordpiece" =
  let module P = Caml.Printf in
  let t = Test.create () in
  List.iter
    ~f:(fun str ->
      let tokens = wordpiece t str ~kind:Normal in
      P.printf "\"%s\"\n%s\n%!" str ([%sexp_of: Token.t list] tokens |> Sexp.to_string_hum))
    [ ""; "hello"; "hello1"; "helloworld"; "helloble"; "helloeee"; "unaffable" ];
  [%expect
    {|
    ""
    ()
    "hello"
    (((text hello) (start 0) (stop 5)
      (with_id (((token_id 0) (is_continuation false)))) (kind Normal)))
    "hello1"
    (((text hello) (start 0) (stop 5)
      (with_id (((token_id 0) (is_continuation false)))) (kind Normal))
     ((text 1) (start 5) (stop 6) (with_id ()) (kind Normal)))
    "helloworld"
    (((text hello) (start 0) (stop 5)
      (with_id (((token_id 0) (is_continuation false)))) (kind Normal))
     ((text world) (start 5) (stop 10) (with_id ()) (kind Normal)))
    "helloble"
    (((text hello) (start 0) (stop 5)
      (with_id (((token_id 0) (is_continuation false)))) (kind Normal))
     ((text ble) (start 5) (stop 8)
      (with_id (((token_id 4) (is_continuation true)))) (kind Normal)))
    "helloeee"
    (((text hello) (start 0) (stop 5)
      (with_id (((token_id 0) (is_continuation false)))) (kind Normal))
     ((text e) (start 5) (stop 6)
      (with_id (((token_id 6) (is_continuation true)))) (kind Normal))
     ((text e) (start 6) (stop 7)
      (with_id (((token_id 6) (is_continuation true)))) (kind Normal))
     ((text e) (start 7) (stop 8)
      (with_id (((token_id 6) (is_continuation true)))) (kind Normal)))
    "unaffable"
    (((text una) (start 0) (stop 3)
      (with_id (((token_id 2) (is_continuation false)))) (kind Normal))
     ((text ffa) (start 3) (stop 6)
      (with_id (((token_id 3) (is_continuation true)))) (kind Normal))
     ((text ble) (start 6) (stop 9)
      (with_id (((token_id 4) (is_continuation true)))) (kind Normal))) |}]
