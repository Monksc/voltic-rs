use crate::{GpuContext, ID, Result, VolticError, buffer_kind};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("shaders/sgd.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SgdDims {
    lr: f32,
    n: u32,
}

#[derive(Debug)]
pub struct SgdOp {
    weight_id: ID,
    n: u32,
    lr: f32,
}

impl SgdOp {
    pub fn new(weight_id: ID, n: u32, lr: f32) -> Self {
        Self { weight_id, n, lr }
    }
}

impl super::Op for SgdOp {
    fn inputs(&self) -> &[ID] {
        std::slice::from_ref(&self.weight_id)
    }
    fn outputs(&self) -> &[ID] {
        std::slice::from_ref(&self.weight_id)
    }

    fn infer_shape(&self, input_shapes: &[&Vec<u32>]) -> Result<Vec<u32>> {
        Ok(input_shapes[0].clone())
    }

    fn pipeline_keys(&self) -> Vec<&'static str> {
        vec!["sgd"]
    }

    fn create_pipelines(
        &self,
        device: &wgpu::Device,
    ) -> Vec<(&'static str, wgpu::ComputePipeline)> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sgd_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sgd_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        vec![("sgd", pipeline)]
    }

    fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
        let pipeline = ctx
            .pipelines
            .get("sgd")
            .ok_or_else(|| VolticError::Internal("sgd pipeline not found".into()))?;

        let weight_buf = ctx.buffers.get(&self.weight_id).ok_or_else(|| {
            VolticError::Internal(format!("weight buffer not found: {:?}", self.weight_id))
        })?;
        let grad_buf = ctx
            .training_buffers
            .get(&(self.weight_id, buffer_kind::GRAD))
            .ok_or_else(|| {
                VolticError::Internal(format!("grad buffer not found: {:?}", self.weight_id))
            })?;

        let dims = SgdDims {
            lr: self.lr,
            n: self.n,
        };
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sgd_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sgd_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sgd_pass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(self.n.div_ceil(256), 1, 1);

        Ok(())
    }

    fn backward(&self, _ctx: &mut GpuContext) -> Result<()> {
        Ok(()) // optimizer has no backward
    }
}
