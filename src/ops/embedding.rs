use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::{GpuContext, ID, Result, VolticError, buffer_kind};

const SHADER: &str = include_str!("shaders/embedding.wgsl");
const BACKWARD_SHADER: &str = include_str!("shaders/embedding_backward.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EmbeddingDims {
    seq_len: u32,
    d_model: u32,
}

#[derive(Debug)]
pub struct EmbeddingOp {
    token_ids: ID, // [seq_len] — f32 cast token IDs
    weights: ID,   // [vocab_size, d_model]
    output: ID,    // [seq_len, d_model]
    seq_len: u32,
    d_model: u32,
}

impl EmbeddingOp {
    pub fn new(token_ids: ID, weights: ID, output: ID, seq_len: u32, d_model: u32) -> Self {
        Self {
            token_ids,
            weights,
            output,
            seq_len,
            d_model,
        }
    }
}

impl super::Op for EmbeddingOp {
    fn inputs(&self) -> &[ID] {
        // token_ids is not trainable — only weights gets a grad buffer
        std::slice::from_ref(&self.weights)
    }
    fn outputs(&self) -> &[ID] {
        std::slice::from_ref(&self.output)
    }

    fn infer_shape(&self, _input_shapes: &[&Vec<u32>]) -> Result<Vec<u32>> {
        Ok(vec![self.seq_len, self.d_model])
    }

    fn pipeline_keys(&self) -> Vec<&'static str> {
        vec!["embedding", "embedding_backward"]
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
            ("embedding", make("embedding", SHADER)),
            (
                "embedding_backward",
                make("embedding_backward", BACKWARD_SHADER),
            ),
        ]
    }

    fn buffers_needed(&self, shapes: &HashMap<ID, Vec<u32>>) -> Vec<(ID, &'static str, u32)> {
        let weights_n: u32 = shapes[&self.weights].iter().product();
        let output_n: u32 = self.seq_len * self.d_model;
        vec![
            (self.weights, buffer_kind::GRAD, weights_n),
            (self.output, buffer_kind::GRAD, output_n),
        ]
    }

    fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
        let pipeline = ctx
            .pipelines
            .get("embedding")
            .ok_or_else(|| VolticError::Internal("embedding pipeline not found".into()))?;

        let token_ids_buf = ctx.buffers.get(&self.token_ids).ok_or_else(|| {
            VolticError::Internal(format!("token_ids buffer not found: {:?}", self.token_ids))
        })?;
        let weights_buf = ctx.buffers.get(&self.weights).ok_or_else(|| {
            VolticError::Internal(format!("weights buffer not found: {:?}", self.weights))
        })?;
        let output_buf = ctx.buffers.get(&self.output).ok_or_else(|| {
            VolticError::Internal(format!("output buffer not found: {:?}", self.output))
        })?;

        let dims = EmbeddingDims {
            seq_len: self.seq_len,
            d_model: self.d_model,
        };
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("embedding_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embedding_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: token_ids_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embedding_pass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((self.seq_len * self.d_model).div_ceil(256), 1, 1);
        Ok(())
    }

    fn backward(&self, ctx: &mut GpuContext) -> Result<()> {
        let pipeline = ctx
            .pipelines
            .get("embedding_backward")
            .ok_or_else(|| VolticError::Internal("embedding_backward pipeline not found".into()))?;

        let token_ids_buf = ctx.buffers.get(&self.token_ids).ok_or_else(|| {
            VolticError::Internal(format!("token_ids buffer not found: {:?}", self.token_ids))
        })?;
        let grad_out_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("embedding grad_out not found".into()))?;
        let grad_weights_buf = ctx
            .training_buffers
            .get(&(self.weights, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("embedding grad_weights not found".into()))?;

        let dims = EmbeddingDims {
            seq_len: self.seq_len,
            d_model: self.d_model,
        };
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("embedding_backward_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embedding_backward_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: token_ids_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embedding_backward_pass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((self.seq_len * self.d_model).div_ceil(256), 1, 1);
        Ok(())
    }
}
