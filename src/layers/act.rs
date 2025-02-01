use crate::layers::layer::Layer;
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