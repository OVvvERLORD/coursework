use crate::layers::layer::Layer;
use rand_distr::Distribution;
use rand;
pub struct Dropout {
    probability: f32,
}

impl Layer for Dropout {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let probability = self.probability;
        let mut vec = args.0.clone();
        let bern = rand_distr::Bernoulli::new(probability.into()).unwrap();
        let scale = 1.0 / (1.0 - probability);
        let mut rng = rand::thread_rng();
    
        for i in 0..vec.len() {
            if bern.sample(&mut rng) == true {
                vec[i] = 0.0;
            } else {
                vec[i] *= scale as f32;
            }
        }
        Ok((vec, args.1))
    }
}
