use crate::layers::layer::Layer;

pub struct GroupNorm {
    pub number_of_groups: usize,
    pub eps: f32,
    pub gamma: f32,
    pub beta: f32,
}

impl Layer for GroupNorm {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let number_of_groups = self.number_of_groups;
        let eps = self.eps;
        let gamma = self.gamma;
        let beta = self.beta;
        let mut vec = args.0; 
        let ch_per_group = args.1[1] / number_of_groups;
        let for_index_opt =  ch_per_group * args.1[2] * args.1[3];
        for batch_idx in 0..args.1[0]{
            for ch in 0..number_of_groups {
                let start_index = batch_idx * args.1[1] * args.1[2] * args.1[3] + ch * for_index_opt;
                let end_index = start_index + for_index_opt;
                let mut mean: f32 = 0.;
                let cnt: f32 = (end_index - start_index) as f32;
                for x in start_index..end_index {
                    mean += vec[x];
                }
                mean = mean / cnt;
                let mut var: f32 = 0.;
        
                for x in start_index..end_index {
                    var += (vec[x] - mean).powf(2.);
                }
                var = var / (cnt);
                let std = (var+eps).sqrt();
                for x in start_index..end_index {
                    vec[x] = ((vec[x] - mean) * gamma) / (std);
                    vec[x] += beta;
                }
            }
        }
        Ok((vec, args.1))
    }
}

pub struct LayerNorm {
    pub eps : f32,
    pub gamma : f32,
    pub beta : f32,
    pub number : usize,
}
impl Layer for LayerNorm {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let mut vec = args.0.clone();
        let limit = self.number;
        for i in (0..args.0.len()).step_by(limit) {
            let mut mean: f32 = 0.;
            let mut var: f32 = 0.;
            for j in 0..limit {
                mean = mean + vec[i + j];
            }
            mean = mean / (limit as f32);
            for j in 0..limit {
                var = var + (vec[i + j] - mean).powf(2.);
            }
            var = var / (limit as f32);
            let std = (var + self.eps).sqrt();
            for j in 0..limit {
                vec[i + j] = ((vec[i + j] - mean) * self.gamma) / (std);
                vec[i + j] = vec[i + j] + self.beta;
            }
        }
        Ok((vec, args.1))
    }
}