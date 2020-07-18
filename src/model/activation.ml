open Torch

let gelu xs =
  let erf = Tensor.erf (Tensor.div1 xs (Scalar.f (Float.sqrt 2.))) in
  Tensor.mul1 (Tensor.add1 erf (Scalar.f 1.)) (Scalar.f 0.5) |> Tensor.mul xs

let relu = Tensor.relu
let mish xs = Tensor.(xs * tanh (softplus xs))
