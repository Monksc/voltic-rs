use std::collections::HashMap;

use crate::{GpuContext, ID, Op, Result};

#[derive(Debug)]
pub struct ConstantOp {
    id: ID,
}

impl ConstantOp {
    pub fn new(id: ID) -> Self {
        Self { id }
    }
}

impl Op for ConstantOp {
    fn inputs(&self) -> &[ID] {
        &[]
    }
    fn outputs(&self) -> &[ID] {
        std::slice::from_ref(&self.id)
    }

    fn pipeline_keys(&self) -> Vec<&'static str> {
        vec![]
    }
    fn create_pipelines(&self, _: &wgpu::Device) -> Vec<(&'static str, wgpu::ComputePipeline)> {
        vec![]
    }
    fn buffers_needed(&self, _: &HashMap<ID, Vec<u32>>) -> Vec<(ID, &'static str, u32)> {
        vec![]
    } // no grad buffer

    fn infer_shape(&self, _: &[&Vec<u32>]) -> Result<Vec<u32>> {
        Ok(vec![1])
    }

    fn forward_gpu(&self, _: &mut GpuContext) -> Result<()> {
        Ok(())
    } // no-op
    fn backward(&self, _: &mut GpuContext) -> Result<()> {
        Ok(())
    } // no-op
}
