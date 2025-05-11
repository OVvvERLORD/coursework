use crate::layers::layer::Layer;
use crate::func::functions::input;
use statrs::function::erf::erf;
use crate::f32::consts::E;
use ndarray;

pub struct SiLU;
impl Layer for SiLU {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        args.mapv_inplace(|x| x / (1. + (-x).exp()));
        Ok(())
    }
}

pub struct GeLU;
impl Layer for GeLU {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let cnst = (2_f64).powf(1. / 2.);
        args.mapv_inplace(|x| x * (1. / 2.) * ((1. + erf((x as f64) / cnst)) as f32));
        Ok(())
    }
}

// // #[test]
// // fn test_silu_big(){
// //     let (input_vec, input_vec_shape) = input(r"C:\study\coursework\src\trash\test_act.safetensors".to_string()).unwrap();
// //     let silu = SiLU;
// //     let (res_vec, res_vec_shape) = silu.operation((input_vec.to_vec(), input_vec_shape.to_vec())).unwrap();
// //     assert!(res_vec_shape == input_vec_shape.to_vec());
// //     let (silu_vec, _) = input(r"C:\study\coursework\src\trash\test_silu_python.safetensors".to_string()).unwrap();
// //     for i in 0..res_vec.len() {
// //         assert!((res_vec[i] - silu_vec[i]).abs() <= 1e-06);
// //     }
// // }

// // #[test]
// // fn test_gelu_big() {
// //     let (input_vec, input_vec_shape) = input(r"C:\study\coursework\src\trash\test_act.safetensors".to_string()).unwrap();
// //     let gelu = GeLU;
// //     let (res_vec, res_vec_shape) = gelu.operation((input_vec.to_vec(), input_vec_shape.to_vec())).unwrap();
// //     assert!(res_vec_shape == input_vec_shape.to_vec());
// //     let (gelu_vec, _) = input(r"C:\study\coursework\src\trash\test_gelu_python.safetensors".to_string()).unwrap();
// //     for i in 0..res_vec.len() {
// //         assert!((res_vec[i] - gelu_vec[i]).abs() <= 1e-06);
// //     }
// // }