use std::collections::HashMap;

use crate::{GpuContext, ID, Result};

pub trait Op: std::fmt::Debug + Send + Sync {
    // Optional — defaults to type name
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    // Shape inference — happens at graph build time, no GPU needed
    fn infer_shape(&self, input_shapes: &[&Vec<u32>]) -> Result<Vec<u32>>;

    // IDs
    fn inputs(&self) -> &[ID];
    fn outputs(&self) -> &[ID];

    // Pipeline registration — called once, cached forever
    fn pipeline_keys(&self) -> Vec<&'static str>;
    fn create_pipelines(&self, device: &wgpu::Device)
    -> Vec<(&'static str, wgpu::ComputePipeline)>;

    fn buffers_needed(&self, shapes: &HashMap<ID, Vec<u32>>) -> Vec<(ID, &'static str, u32)> {
        vec![]
    }

    // Execution — gets everything it needs via GpuContext
    fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()>;

    // Gradients
    fn backward(&self, ctx: &mut GpuContext) -> Result<()> {
        Ok(())
    }
}
