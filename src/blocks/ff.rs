use crate::layers::{
        layer::Layer,
        linear::Linear,
        act::GeLU
};
use crate::func::functions::input;

pub struct FeedForward {
    pub operations: Vec<Box<dyn Layer>>,
}
impl FeedForward {
    pub fn FeedForward_constr (
        weigths_1: Vec<f32>,
        weights_shape_1 : Vec<usize>,
        bias_1: Vec<f32>,
        bias_shape_1 : Vec<usize>,
        is_bias_1 : bool,
        weigths_2: Vec<f32>,
        weights_shape_2 : Vec<usize>,
        bias_2: Vec<f32>,
        bias_shape_2 : Vec<usize>,
        is_bias_2 : bool,
    ) -> Self {
        let mut vec: Vec<Box<dyn Layer>> = Vec::new();
        vec.push(Box::new(Linear {weigths : weigths_1, weights_shape : weights_shape_1, bias : bias_1, bias_shape : bias_shape_1, is_bias : is_bias_1}));
        vec.push(Box::new(GeLU));
        vec.push(Box::new(Linear { weigths : weigths_2, weights_shape : weights_shape_2, bias : bias_2, bias_shape : bias_shape_2, is_bias : is_bias_2}));
        Self { operations: vec }
    }
}
impl Layer for FeedForward {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let res = &self.operations[0].operation(args)?;
        let mut res_vec = res.0.clone(); // результаты линейного
        let mut res_vec_shape = res.1.clone();
        let limit = res_vec_shape[3] / 2;
        let lin_tensor = ndarray::Array4::from_shape_vec((res.1[0], res.1[1], res.1[2], res.1[3]), res_vec.to_vec())?;
        let part_1= lin_tensor.slice(ndarray::s![.., .., .., ..limit]);
        let part_2= lin_tensor.slice(ndarray::s![.., .., .., limit..]);
        let part_vec_1 = part_1.to_owned().into_raw_vec_and_offset().0;
        let part_vec_2 = part_2.to_owned().into_raw_vec_and_offset().0;
        let mut part_vec_shape = res_vec_shape;
        part_vec_shape[3] = limit;
        let act_part_vec_2 = &self.operations[1].operation((part_vec_2, part_vec_shape.clone()))?;
        let part_vec_2 = act_part_vec_2.0.clone();
        let act_vec = ndarray::Array1::from_shape_vec(part_vec_2.len(), part_vec_2)?;
        let another_vec = ndarray::Array1::from_shape_vec(part_vec_1.len(), part_vec_1)?;
        res_vec = (act_vec * another_vec).to_vec();
        res_vec_shape = part_vec_shape;
        let res = &self.operations[2].operation((res_vec, res_vec_shape))?;
        Ok((res.0.clone(), res.1.clone()))
    }
}

#[test]
fn ff_test_2x2x1280x1280() {
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\ff_input.safetensors".to_string()).unwrap();
    let (weigths_1, weights_shape_1) = input(r"C:\study\coursework\src\trash\ff_lin1.safetensors".to_string()).unwrap();
    let (bias_1, bias_shape_1) = input(r"C:\study\coursework\src\trash\ff_lin1_bias.safetensors".to_string()).unwrap();
    let (bias_2, bias_shape_2) = input(r"C:\study\coursework\src\trash\ff_lin2_bias.safetensors".to_string()).unwrap();
    let (weigths_2, weights_shape_2) = input(r"C:\study\coursework\src\trash\ff_lin2.safetensors".to_string()).unwrap();
    let ff = FeedForward::FeedForward_constr(weigths_1.to_vec(), weights_shape_1.to_vec(), bias_1.to_vec(), bias_shape_1.to_vec(), true, weigths_2.to_vec(), weights_shape_2.to_vec(), bias_2.to_vec(), bias_shape_2.to_vec(), true);
    print!("{:?}", test_vec_shape);
    let (res_vec, res_vec_shape) = ff.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (ff_vec, ff_vec_shape) = input(r"C:\study\coursework\src\trash\ff_output.safetensors".to_string()).unwrap();
    assert!(res_vec_shape == ff_vec_shape.to_vec());
    for i in 0..res_vec.len() {
        assert!((res_vec[i] - ff_vec[i]).abs() <= 1e-05);
    }
}