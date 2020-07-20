open Torch

let gelu_new =
  let sqrt_two_over_pi = 2. /. Float.pi |> Float.sqrt in
  fun xs ->
    let ys =
      Tensor.(
        tanh (((pow xs ~exponent:(Scalar.f 3.) * f 0.044715) + xs) * f sqrt_two_over_pi))
    in
    Tensor.(xs * f 0.5 * (ys + f 1.))

let gelu xs =
  let erf = Tensor.erf (Tensor.div1 xs (Scalar.f (Float.sqrt 2.))) in
  Tensor.mul1 (Tensor.add1 erf (Scalar.f 1.)) (Scalar.f 0.5) |> Tensor.mul xs

let relu = Tensor.relu
let mish xs = Tensor.(xs * tanh (softplus xs))
let swish xs = Tensor.(xs * sigmoid xs)
