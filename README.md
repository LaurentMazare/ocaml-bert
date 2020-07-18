# ocaml-bert
BERT-like NLP models implemented in OCaml using the
[ocaml-torch](https://github.com/LaurentMazare/ocaml-torch) PyTorch bindings.

These are based on the [rust-bert](https://github.com/guillaume-be/rust-bert)
implementation which itself is a port of Huggingface's [Transformers](https://github.com/huggingface/transformers)
to Rust.

The models implemented so far are:

- DistillBERT:
  pre-trained weights are available under the Apache 2.0 license from Huggingface.
  [weights file](https://cdn.huggingface.co/distilbert-base-uncased-rust_model.ot),
  [vocabulary file](https://cdn.huggingface.co/bert-base-uncased-vocab.txt).
