use std::{fs::File, io::{Read, Write}, sync::Arc};
use ndarray;
use safetensors::{serialize, SafeTensors};

pub fn Tensor_Mul(args:(Vec<f32>, Vec<usize>, Vec<f32>, Vec<usize>)) ->  Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let A = ndarray::Array4::from_shape_vec((args.1[0], args.1[1], args.1[2], args.1[3]), args.0)?;
    let B = ndarray::Array4::from_shape_vec((args.3[0], args.3[1], args.3[2], args.3[3]), args.2)?;
    let B = B.broadcast([args.1[0], args.1[1], args.3[2], args.3[3]]).unwrap();
    let mut result_shape = args.1;
    result_shape[3] = args.3[2];
    let mut result = ndarray::Array4::<f32>::zeros((result_shape[0], result_shape[1], result_shape[2], result_shape[3]));
    for ((mut res, a_batch), b_batch) in result
        .axis_iter_mut(ndarray::Axis(0))
        .zip(A.axis_iter(ndarray::Axis(0)))
        .zip(B.axis_iter(ndarray::Axis(0)))
    {
        for ((mut res_slice, a_slice), b_slice) in res
            .axis_iter_mut(ndarray::Axis(0))
            .zip(a_batch.axis_iter(ndarray::Axis(0)))
            .zip(b_batch.axis_iter(ndarray::Axis(0)))
        {
            let b_transposed = b_slice.clone().reversed_axes();
            let product = a_slice.dot(&b_transposed);
            res_slice.assign(&product);
        }
    }
    let result = result.into_raw_vec_and_offset().0;
    Ok((result, result_shape))
}

pub fn input(input_name: String) -> Result<(Arc<Vec<f32>>, Arc<Vec<usize>>), Box<dyn std::error::Error>> { 
    let mut file = File::open(input_name.to_string())?;
    let mut buffer = Vec::<u8>::new();
    let _ = file.read_to_end(&mut buffer);

    let tensors = SafeTensors::deserialize(&buffer)?;
    let mut input_vec = Vec::<f32>::new();
    let mut input_vec_shape = Vec::<usize>::new();
    for (_, tensor) in tensors.tensors() {
        input_vec = bytemuck::cast_slice(tensor.data()).to_vec();
        input_vec_shape = tensor.shape().to_vec();
    }
    if input_vec_shape.len() == 2 {
        input_vec_shape.insert(0, 1);
        input_vec_shape.insert(0, 1);
    }

    let input_vec: Arc<Vec<f32>> = Arc::from(input_vec);
    let input_vec_shape: Arc<Vec<usize>> =  Arc::from(input_vec_shape);

    Ok((input_vec, input_vec_shape))
}

pub fn output(output_name: String, tensor_vec:Vec<f32>, shape_vec:Vec<usize>) -> Result<(), Box<dyn std::error::Error>> {
    let binding = tensor_vec.to_vec();
    let prep_output:&[u8] = bytemuck::cast_slice(&binding);
    let mut tensors = std::collections::HashMap::new();
    tensors.insert("output_tensor".to_string(), safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape_vec.clone(), prep_output)?);
    let serialized_data = serialize(&tensors, &None)?;
    let mut file = File::create(output_name.to_string())?;
    file.write_all(&serialized_data)?;
    Ok(())
}

pub fn nearest_neighbour_interpolation (args:(Vec<f32>, Vec<usize>)) ->  Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let (input_vec, input_vec_shape) = args;
    let input_tensor = ndarray::Array4::from_shape_vec((input_vec_shape[0], input_vec_shape[1], input_vec_shape[2], input_vec_shape[3]), input_vec)?;
    let mut output_tensor = ndarray::Array4::<f32>::zeros((input_vec_shape[0], input_vec_shape[1], input_vec_shape[2] * 2, input_vec_shape[3] * 2));
    for batches in 0..input_vec_shape[0] {
        for channels in 0..input_vec_shape[1] {
            for height in 0..input_vec_shape[2] * 2 {
                for width in 0..input_vec_shape[3] * 2 {
                    let round_height = height / 2;
                    let round_width = width / 2;
                    output_tensor[[batches, channels, height, width]] = input_tensor[[batches, channels, round_height, round_width]];
                }
            }
        }
    }
    let res_vec = output_tensor.into_raw_vec_and_offset().0;
    let mut res_vec_shape = input_vec_shape;
    res_vec_shape[2] *= 2;
    res_vec_shape[3] *= 2;
    Ok((res_vec, res_vec_shape))
}