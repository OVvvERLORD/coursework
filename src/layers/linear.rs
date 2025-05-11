use std::sync::atomic;

use ndarray::Zip;

use crate::{
    layers::layer::Layer,
    func::functions::{Tensor_Mul, input, output}
};

pub struct Linear{
    pub weights: ndarray::Array4<f32>,
    pub bias: ndarray::Array4<f32>,
    pub is_bias : bool,
}

impl Layer for Linear {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let _ = Tensor_Mul(args, &self.weights).unwrap();
        if self.is_bias {
            let bias = self.bias
            .broadcast(args.shape())
            .unwrap();
            *args += &bias;
        }
        Ok(())
    }
}

// impl Linear {
//     pub fn new(in_features : usize, out_features : usize, bias : bool) -> Self {
//             let mut weights_shape : Vec<usize> = Vec::new();
//             weights_shape.push(out_features);
//             weights_shape.push(in_features);
//             let mut weights_vec : Vec<f32> = Vec::new();
//             for _ in 0..in_features*out_features {
//                 weights_vec.push(rand::random::<f32>());
//             }
//             let mut bias_shape : Vec<usize> = Vec::new();
//             let mut bias_vec : Vec<f32> = Vec::new();
//             if bias {
//                 bias_shape.push(out_features);
//                 for _ in 0..out_features {
//                     bias_vec.push(rand::random::<f32>());
//                 }
//             }
//             Self { weigths: weights_vec, weights_shape: weights_shape, bias: bias_vec, bias_shape: bias_shape, is_bias: bias }
//     }
// }

// #[test]
// fn test_linear_big_unbiased_sym(){
//     let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_inp_linear.safetensors".to_string()).unwrap();
//     let (lin_w, lin_w_shape) = input(r"C:\study\coursework\src\trash\test_weight_linear.safetensors".to_string()).unwrap();
//     let lin = Linear{weigths: lin_w.to_vec(), weights_shape: lin_w_shape.to_vec(), bias : lin_w.to_vec(), bias_shape: lin_w_shape.to_vec(), is_bias: false};
//     let (res_vec, res_vec_shape) = lin.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
//     let _ = output(r"C:\study\coursework\src\trash\test_linear_rust.safetensors".to_string(), res_vec.clone(), res_vec_shape.clone()).unwrap();
//     let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_linear_python.safetensors".to_string()).unwrap();
//     assert!(res_vec_shape == py_vec_shape.to_vec());
//     for i in 0..res_vec.len() {
//         assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-67 );
//     }
// }

#[test]
fn test_linear_big_biased_sym() {
    let mut tensor = input(r"C:\study\coursework\src\trash\test_inp_bias_linear.safetensors".to_string()).unwrap();
    let weight = input(r"C:\study\coursework\src\trash\test_weight_bias_linear.safetensors".to_string()).unwrap();
    let bias = input(r"C:\study\coursework\src\trash\test_bias_linear.safetensors".to_string()).unwrap();
    let lin = Linear{is_bias: true, weights: weight, bias: bias};
    let _ = lin.operation(&mut tensor);
    let py_tensor = input(r"C:\study\coursework\src\trash\test_linear_bias_python.safetensors".to_string()).unwrap();
    let shape = tensor.shape();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-03);
                }
            }
        }
    }
}

#[test]
fn test_linear_big_unbiased_unsym(){
    let mut tensor = input(r"C:\study\coursework\src\trash\test_inp_unsym_linear.safetensors".to_string()).unwrap();
    let weight = input(r"C:\study\coursework\src\trash\test_weight_unsym_linear.safetensors".to_string()).unwrap();
    let bias = input(r"C:\study\coursework\src\trash\test_unsym_bias_linear.safetensors".to_string()).unwrap();
    print!("{:?} {:?} {:?}", tensor.shape(), weight.shape(), bias.shape());
    let lin = Linear{is_bias: false, weights: weight, bias: bias};
    let _ = lin.operation(&mut tensor);
    let py_tensor =  input(r"C:\study\coursework\src\trash\test_linear_unsym_python.safetensors".to_string()).unwrap();
    let shape = tensor.shape();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-09993223232232329);
                }
            }
        }
    }

}

#[test]
fn test_linear_big_biased_unsym() {
    let mut tensor = input(r"C:\study\coursework\src\trash\test_inp_unsym_bias_linear.safetensors".to_string()).unwrap();
    let weight = input(r"C:\study\coursework\src\trash\test_weight_unsym_bias_linear.safetensors".to_string()).unwrap();
    let bias = input(r"C:\study\coursework\src\trash\test_unsym_bias_linear.safetensors".to_string()).unwrap();
    let lin = Linear{is_bias: true, weights: weight, bias: bias};
    let _ = lin.operation(&mut tensor);
    let py_tensor =  input(r"C:\study\coursework\src\trash\test_linear_unsym_bias_python.safetensors".to_string()).unwrap();
    let shape = tensor.shape();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-09993223232232329);
                }
            }
        }
    }

}