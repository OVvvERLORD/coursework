use crate::layers::{
        layer::Layer,
        linear::Linear,
        act::GeLU
};
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
        let mut res_vec = res.0.clone();
        let mut res_vec_shape = res.1.clone();
        let limit = (res_vec_shape[0] *res_vec_shape[1] *res_vec_shape[2] *res_vec_shape[3]) / 2;
        let (part_vec_1, part_vec_2) = res_vec.split_at(limit);
        let part_vec_1 = part_vec_1.to_vec();
        let part_vec_2 = part_vec_2.to_vec();
        let mut part_vec_shape = res_vec_shape;
        part_vec_shape[3] = part_vec_shape[3] / 2;
        let act_part_vec_1 = &self.operations[1].operation((part_vec_1, part_vec_shape.clone()))?;
        let part_vec_1 = act_part_vec_1.0.clone();
        let act_vec = ndarray::Array1::from_shape_vec(limit, part_vec_1)?;
        let another_vec = ndarray::Array1::from_shape_vec(limit, part_vec_2)?;
        res_vec = (act_vec * another_vec).to_vec();
        res_vec_shape = part_vec_shape;
        let res = &self.operations[2].operation((res_vec, res_vec_shape))?;
        Ok((res.0.clone(), res.1.clone()))
    }
}