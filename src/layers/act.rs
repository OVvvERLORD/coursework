use crate::layers::layer::Layer;
use crate::func::functions::input;
use statrs::function::erf::erf;
use crate::f32::consts::E;

pub struct SiLU;
impl Layer for SiLU {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let mut vec = args.0;
        for i in 0..vec.len() {
            vec[i] = vec[i] * (1.0 / (1.0 + E.powf(-vec[i])));
        }
        Ok((vec, args.1))
    }
}

pub struct GeLU;
impl Layer for GeLU {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let mut vec = args.0;
        for i in 0..vec.len() {
            vec[i] = vec[i] * (1. / 2.) * ((1. + erf((vec[i] as f64) / (2_f64).powf(1. / 2.))) as f32);
        }
        Ok((vec, args.1))
    }
}

#[test]
fn test_silu_big(){
    let (input_vec, input_vec_shape) = input(r"C:\study\coursework\src\trash\test_act.safetensors".to_string()).unwrap();
    let silu = SiLU;
    let (res_vec, res_vec_shape) = silu.operation((input_vec.to_vec(), input_vec_shape.to_vec())).unwrap();
    assert!(res_vec_shape == input_vec_shape.to_vec());
    let (silu_vec, _) = input(r"C:\study\coursework\src\trash\test_silu_python.safetensors".to_string()).unwrap();
    for i in 0..res_vec.len() {
        assert!((res_vec[i] - silu_vec[i]).abs() <= 1e-06);
    }
}

#[test]
fn test_gelu_big() {
    let (input_vec, input_vec_shape) = input(r"C:\study\coursework\src\trash\test_act.safetensors".to_string()).unwrap();
    let gelu = GeLU;
    let (res_vec, res_vec_shape) = gelu.operation((input_vec.to_vec(), input_vec_shape.to_vec())).unwrap();
    assert!(res_vec_shape == input_vec_shape.to_vec());
    let (gelu_vec, _) = input(r"C:\study\coursework\src\trash\test_gelu_python.safetensors".to_string()).unwrap();
    for i in 0..res_vec.len() {
        assert!((res_vec[i] - gelu_vec[i]).abs() <= 1e-06);
    }
}