use ndarray;
pub trait Layer {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
} 