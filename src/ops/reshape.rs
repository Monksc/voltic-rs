use std::collections::HashMap;

use crate::{GpuContext, ID, Result, VolticError, buffer_kind};

#[derive(Debug)]
pub struct ReshapeOp {
    input: ID,
    output: ID,
    new_shape: Vec<u32>,
    n: u32,
}

impl ReshapeOp {
    pub fn new(input: ID, output: ID, new_shape: Vec<u32>) -> Self {
        let n: u32 = new_shape.iter().product();
        Self {
            input,
            output,
            new_shape,
            n,
        }
    }
}

impl super::Op for ReshapeOp {
    fn inputs(&self) -> &[ID] {
        std::slice::from_ref(&self.input)
    }
    fn outputs(&self) -> &[ID] {
        std::slice::from_ref(&self.output)
    }

    fn infer_shape(&self, _input_shapes: &[&Vec<u32>]) -> Result<Vec<u32>> {
        Ok(self.new_shape.clone())
    }

    fn pipeline_keys(&self) -> Vec<&'static str> {
        vec![] // no shaders needed
    }

    fn create_pipelines(
        &self,
        _device: &wgpu::Device,
    ) -> Vec<(&'static str, wgpu::ComputePipeline)> {
        vec![]
    }

    fn buffers_needed(&self, _shapes: &HashMap<ID, Vec<u32>>) -> Vec<(ID, &'static str, u32)> {
        vec![
            (self.input, buffer_kind::GRAD, self.n),
            (self.output, buffer_kind::GRAD, self.n),
        ]
    }

    fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
        let input_buf = ctx.buffers.get(&self.input).ok_or_else(|| {
            VolticError::Internal(format!("reshape input buffer not found: {:?}", self.input))
        })?;
        let output_buf = ctx.buffers.get(&self.output).ok_or_else(|| {
            VolticError::Internal(format!(
                "reshape output buffer not found: {:?}",
                self.output
            ))
        })?;

        let size = (self.n as u64) * 4; // f32 = 4 bytes
        ctx.encoder
            .copy_buffer_to_buffer(input_buf, 0, output_buf, 0, size);
        Ok(())
    }

    fn backward(&self, ctx: &mut GpuContext) -> Result<()> {
        let grad_out_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("reshape grad_out not found".into()))?;
        let grad_in_buf = ctx
            .training_buffers
            .get(&(self.input, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("reshape grad_input not found".into()))?;

        let size = (self.n as u64) * 4;
        ctx.encoder
            .copy_buffer_to_buffer(grad_out_buf, 0, grad_in_buf, 0, size);
        Ok(())
    }
}
