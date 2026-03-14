use std::collections::HashMap;

use crate::{buffer_kind, GpuContext, Result, VolticError, ID};
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("shaders/mse.wgsl");
const BACKWARD_SHADER: &str = include_str!("shaders/mse_backward.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MseDims {
    n: u32,
}

#[derive(Debug)]
pub struct MseOp {
    inputs: [ID; 2], // [y_pred, y_true]
    output: ID,
    n: u32,
}

impl MseOp {
    pub fn new(y_pred: ID, y_true: ID, output: ID, n: u32) -> Self {
        Self {
            inputs: [y_pred, y_true],
            output,
            n,
        }
    }
}

impl super::Op for MseOp {
    fn inputs(&self) -> &[ID] {
        &self.inputs
    }
    fn outputs(&self) -> &[ID] {
        std::slice::from_ref(&self.output)
    }

    fn infer_shape(&self, _input_shapes: &[&Vec<u32>]) -> Result<Vec<u32>> {
        Ok(vec![self.n]) // one error value per element
    }

    fn pipeline_keys(&self) -> Vec<&'static str> {
        vec!["mse", "mse_backward"]
    }

    fn create_pipelines(
        &self,
        device: &wgpu::Device,
    ) -> Vec<(&'static str, wgpu::ComputePipeline)> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mse_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mse_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mse_backward_shader"),
            source: wgpu::ShaderSource::Wgsl(BACKWARD_SHADER.into()),
        });
        let backward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mse_backward_pipeline"),
            layout: None,
            module: &backward_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        vec![("mse", pipeline), ("mse_backward", backward_pipeline)]
    }

    fn buffers_needed(&self, shapes: &HashMap<ID, Vec<u32>>) -> Vec<(ID, &'static str, u32)> {
        let n: u32 = shapes[&self.inputs[0]].iter().product();
        vec![
            (self.inputs[0], buffer_kind::GRAD, n), // grad for y_pred
        ]
    }

    fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
        let pipeline = ctx
            .pipelines
            .get("mse")
            .ok_or_else(|| VolticError::Internal("mse pipeline not found".into()))?;

        let y_pred_buf = ctx.buffers.get(&self.inputs[0]).ok_or_else(|| {
            VolticError::Internal(format!("buffer not found: {:?}", self.inputs[0]))
        })?;
        let y_true_buf = ctx.buffers.get(&self.inputs[1]).ok_or_else(|| {
            VolticError::Internal(format!("buffer not found: {:?}", self.inputs[1]))
        })?;
        let out_buf = ctx
            .buffers
            .get(&self.output)
            .ok_or_else(|| VolticError::Internal(format!("buffer not found: {:?}", self.output)))?;

        let dims = MseDims { n: self.n };
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mse_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mse_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: y_pred_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y_true_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = self.n.div_ceil(256);

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mse_pass"),
                timestamp_writes: None,
            });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }

    fn backward(&self, ctx: &mut GpuContext) -> Result<()> {
        let pipeline = ctx
            .pipelines
            .get("mse_backward")
            .ok_or_else(|| VolticError::Internal("mse_backward pipeline not found".into()))?;

        let y_pred_buf = ctx.buffers.get(&self.inputs[0]).ok_or_else(|| {
            VolticError::Internal(format!("buffer not found: {:?}", self.inputs[0]))
        })?;
        let y_true_buf = ctx.buffers.get(&self.inputs[1]).ok_or_else(|| {
            VolticError::Internal(format!("buffer not found: {:?}", self.inputs[1]))
        })?;
        let grad_buf = ctx
            .training_buffers
            .get(&(self.inputs[0], buffer_kind::GRAD))
            .ok_or_else(|| {
                VolticError::Internal(format!("grad buffer not found: {:?}", self.inputs[0]))
            })?;

        let dims = MseDims { n: self.n };
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mse_backward_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mse_backward_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: y_pred_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y_true_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = self.n.div_ceil(256);
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mse_backward_pass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }
}
