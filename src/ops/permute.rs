use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::{buffer_kind, GpuContext, Result, VolticError, ID};

const SHADER: &str = include_str!("shaders/permute.wgsl");
const BACKWARD_SHADER: &str = include_str!("shaders/permute_backward.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PermuteDims {
    rank: u32,
    total: u32,
    _pad0: u32,
    _pad1: u32,
    shape: [[u32; 4]; 2],
    perm: [[u32; 4]; 2],
    out_shape: [[u32; 4]; 2],
}

#[derive(Debug)]
pub struct PermuteOp {
    input: ID,
    output: ID,
    shape: Vec<u32>,
    perm: Vec<usize>,
    out_shape: Vec<u32>,
    total: u32,
}

impl PermuteOp {
    pub fn new(input: ID, output: ID, shape: Vec<u32>, perm: Vec<usize>) -> Self {
        let out_shape: Vec<u32> = perm.iter().map(|&p| shape[p]).collect();
        let total: u32 = out_shape.iter().product();
        Self {
            input,
            output,
            shape,
            perm,
            out_shape,
            total,
        }
    }

    fn make_dims(&self) -> PermuteDims {
        let rank = self.shape.len();
        let mut shape = [[0u32; 4]; 2];
        let mut perm = [[0u32; 4]; 2];
        let mut out_shape = [[0u32; 4]; 2];

        for i in 0..rank.min(8) {
            shape[i / 4][i % 4] = self.shape[i];
            perm[i / 4][i % 4] = self.perm[i] as u32;
            out_shape[i / 4][i % 4] = self.out_shape[i];
        }

        PermuteDims {
            rank: rank as u32,
            total: self.total,
            _pad0: 0,
            _pad1: 0,
            shape,
            perm,
            out_shape,
        }
    }
}

impl super::Op for PermuteOp {
    fn inputs(&self) -> &[ID] {
        std::slice::from_ref(&self.input)
    }
    fn outputs(&self) -> &[ID] {
        std::slice::from_ref(&self.output)
    }

    fn infer_shape(&self, _input_shapes: &[&Vec<u32>]) -> Result<Vec<u32>> {
        Ok(self.out_shape.clone())
    }

    fn pipeline_keys(&self) -> Vec<&'static str> {
        vec!["permute", "permute_backward"]
    }

    fn create_pipelines(
        &self,
        device: &wgpu::Device,
    ) -> Vec<(&'static str, wgpu::ComputePipeline)> {
        let make = |label, src: &'static str| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        vec![
            ("permute", make("permute", SHADER)),
            (
                "permute_backward",
                make("permute_backward", BACKWARD_SHADER),
            ),
        ]
    }

    fn buffers_needed(&self, _shapes: &HashMap<ID, Vec<u32>>) -> Vec<(ID, &'static str, u32)> {
        let input_n: u32 = self.shape.iter().product();
        let output_n: u32 = self.total;
        vec![
            (self.input, buffer_kind::GRAD, input_n),
            (self.output, buffer_kind::GRAD, output_n),
        ]
    }

    fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
        let pipeline = ctx
            .pipelines
            .get("permute")
            .ok_or_else(|| VolticError::Internal("permute pipeline not found".into()))?;

        let input_buf = ctx
            .buffers
            .get(&self.input)
            .ok_or_else(|| VolticError::Internal(format!("buffer not found: {:?}", self.input)))?;
        let output_buf = ctx
            .buffers
            .get(&self.output)
            .ok_or_else(|| VolticError::Internal(format!("buffer not found: {:?}", self.output)))?;

        let dims = self.make_dims();
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("permute_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("permute_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
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
                label: Some("permute_pass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(self.total.div_ceil(256), 1, 1);
        Ok(())
    }

    fn backward(&self, ctx: &mut GpuContext) -> Result<()> {
        let pipeline = ctx
            .pipelines
            .get("permute_backward")
            .ok_or_else(|| VolticError::Internal("permute_backward pipeline not found".into()))?;

        let grad_out_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("permute grad_out not found".into()))?;
        let grad_in_buf = ctx
            .training_buffers
            .get(&(self.input, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("permute grad_input not found".into()))?;

        let dims = self.make_dims();
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("permute_backward_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("permute_backward_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_in_buf.as_entire_binding(),
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
                label: Some("permute_backward_pass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(self.total.div_ceil(256), 1, 1);
        Ok(())
    }
}
