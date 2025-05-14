use ndarray::{Axis, Zip};
use ndarray::{ArrayBase, DataMut, Ix4};
use crate::layers::layer::Layer;
use crate::func::functions::{input, output};
use rayon::prelude::*;
use ndarray::parallel::prelude::*;
pub struct GroupNorm{
    pub number_of_groups: usize,
    pub eps: f32,
    pub gamma: ndarray::Array4<f32>,
    pub beta: ndarray::Array4<f32>
}

impl Layer for GroupNorm {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let shape = args.dim();
        let ch_per_group = shape.1 / self.number_of_groups;
        let for_index_opt =  ch_per_group * shape.2 * shape.3;
        for batch_idx in 0..shape.0{
            for gr in 0..self.number_of_groups {
                let start_index = gr * ch_per_group;
                let end_index = start_index + ch_per_group;

                let mut sum = 0_f32;
                let mut sum_var = 0_f32;
                for ch in start_index..end_index {
                    for y in 0..shape.2 {
                        for x in 0..shape.3 {
                            sum += args[[batch_idx, ch, y, x]];
                            sum_var += args[[batch_idx, ch, y, x]] * args[[batch_idx, ch, y, x]];
                        }
                    }
                }
                let mean = sum / for_index_opt as f32;
                let var = (sum_var / for_index_opt as f32) - (mean * mean);
                let inv_std = 1. / (var + self.eps).sqrt();
                for ch in start_index..end_index {
                    let gamma = self.gamma[[0, 0, 0, ch]];
                    let beta = self.beta[[0, 0, 0, ch]];
                    for y in 0..shape.2{
                        for x in 0..shape.3 {
                            let sl = &mut args[[batch_idx, ch, y, x]];
                            *sl = (*sl - mean) * inv_std * gamma + beta;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}


pub struct LayerNorm {
    pub eps : f32,
    pub gamma: ndarray::Array4<f32>,
    pub beta: ndarray::Array4<f32>,
    pub number : usize,
}

impl Layer for LayerNorm {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let means = args.map_axis(ndarray::Axis(3), |sl| sl.mean().unwrap())
        .insert_axis(ndarray::Axis(3));
        let vars = args.map_axis(ndarray::Axis(3), |sl| {
            let mean = sl.mean().unwrap();
            sl.into_par_iter().map(|x| (x - mean).powi(2)).sum::<f32>() / sl.len() as f32
        }).insert_axis(ndarray::Axis(3));
        let means = means
        .broadcast(args.shape())
        .unwrap()
        .into_dimensionality()
        .unwrap();
        let vars = vars
        .broadcast(args.shape())
        .unwrap()
        .into_dimensionality()
        .unwrap();
        let gamma = self.gamma
        .broadcast(args.shape())
        .unwrap()
        .into_dimensionality()
        .unwrap();
        let beta = self.beta.broadcast(args.shape()).unwrap().into_dimensionality().unwrap();
        Zip::from(args)
        .and(&means)
        .and(&vars)
        .and(&gamma)
        .and(&beta)
        .for_each(|x, &m, &v, &g, &b| {
            let std = (v + self.eps).sqrt();
            *x = (*x - m) / std *g + b;
        }
        );
        Ok(())
    }
}

// #[test]
// fn test_layer_norm_small() {
//     let input_vec = vec![2., 7., 161., 45124., 323., 21., 1., 1515., 321., 32., 323., 5252., 12., 3., 4., 16.];
//     let input_vec_shape: Vec<usize> = vec![2, 2, 2, 2];
//     let layernorm = LayerNorm {eps : 0., gamma: 1., beta: 0., number: 2};
//     let (res_vec, res_vec_shape) = layernorm.operation((input_vec.to_vec(), input_vec_shape.to_vec())).unwrap();
//     assert!(input_vec_shape == res_vec_shape);
//     for i in (0..res_vec_shape.len()).step_by(2) {
//         assert!((-1. - res_vec[i]).abs() <= 1e-67);
//         assert!((1. - res_vec[i + 1]).abs() <= 1e-67);
//     }
// }

// #[test]
// fn test_layer_norm_big() {
//     let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_inp.safetensors".to_string()).unwrap();
//     let layernorm = LayerNorm {eps : 1e-05, gamma: 1., beta : 0., number: 1280};
//     let (res_vec, res_vec_shape ) = layernorm.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
//     assert!(test_vec_shape == res_vec_shape.clone().into());
//     let _ = output(r"C:\study\coursework\src\trash\test_layernorm_rust.safetensors".to_string(), res_vec.to_vec(), res_vec_shape.to_vec());
//     let (layernorm_vec, _) = input(r"C:\study\coursework\src\trash\test_layernorm_python.safetensors".to_string()).unwrap();
//     for i in 0..res_vec.len() {
//         assert!((res_vec[i] - layernorm_vec[i]).abs() <= 1e-5);
//     }
// }

// #[test]
// fn test_group_norm_big() {
//     let (input_vec, input_vec_shape) = input(r"C:\study\coursework\src\trash\test_inp.safetensors".to_string()).unwrap();
//     let grnorm = GroupNorm { eps: 1e-05, number_of_groups: 32, gamma: 1., beta: 0.};
//     let (res_vec, res_vec_shape) = grnorm.operation((input_vec.to_vec(), input_vec_shape.to_vec())).unwrap();
//     assert!(input_vec_shape == res_vec_shape.clone().into());
//     let (grnorm_vec, _) = input(r"C:\study\coursework\src\trash\test_grnorm_python.safetensors".to_string()).unwrap();
//     for i in 0..res_vec.len() {
//         assert!( (res_vec[i] - grnorm_vec[i]).abs() <= 1e-03);
//     }
// }

#[test]
fn test_group_norm_biased() {
    let mut input_tensor = input(format!( r"C:\study\coursework\src\trash\test_grnorm_bias_i.safetensors")).unwrap();
    let gamma = input(format!(r"C:\study\coursework\src\trash\test_grnorm_bias_w.safetensors")).unwrap();
    let beta = input(format!(r"C:\study\coursework\src\trash\test_grnorm_bias_b.safetensors")).unwrap();
    let grnorm = GroupNorm{number_of_groups: 32, eps: 1e-05, gamma: gamma, beta: beta};
    let _ = grnorm.operation(&mut input_tensor);
    let py_tensor = input(format!(r"C:\study\coursework\src\trash\test_grnorm_bias_r.safetensors")).unwrap();
    let shape = input_tensor.shape();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((input_tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-03);
                }
            }
        }
    }
}
#[test]
fn test_layer_norm_bias() {
    let mut input_tensor = input(format!(r"C:\study\coursework\src\trash\test_lnorm_bias_i.safetensors")).unwrap();
    let gamma = input(format!(r"C:\study\coursework\src\trash\test_lnorm_bias_w.safetensors")).unwrap();
    let beta = input(format!(r"C:\study\coursework\src\trash\test_lnorm_bias_b.safetensors")).unwrap();
    let lnorm = LayerNorm{ gamma: gamma, beta: beta, eps: 1e-05, number: 1280};
    let _ = lnorm.operation((&mut input_tensor));
    let py_tensor = input(format!(r"C:\study\coursework\src\trash\test_lnorm_bias_r.safetensors")).unwrap();
    let shape = input_tensor.shape();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((input_tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-05);
                }
            }
        }
    }
}