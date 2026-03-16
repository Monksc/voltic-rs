use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::{buffer_kind, GpuContext, Result, VolticError, ID};

const SHADER: &str = include_str!("shaders/im2col.wgsl");
const BACKWARD_SHADER: &str = include_str!("shaders/col2im.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Im2ColDims {
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    out_height: u32,
    out_width: u32,
}

#[derive(Debug)]
pub struct Im2ColOp {
    input: ID,
    output: ID,
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    out_height: u32,
    out_width: u32,
}

impl Im2ColOp {
    pub fn new(
        input: ID,
        output: ID,
        batch: u32,
        channels: u32,
        height: u32,
        width: u32,
        kernel_size: u32,
        stride: u32,
        padding: u32,
    ) -> Self {
        let out_height = (height + 2 * padding - kernel_size) / stride + 1;
        let out_width = (width + 2 * padding - kernel_size) / stride + 1;

        Self {
            input,
            output,
            batch,
            channels,
            height,
            width,
            kernel_size,
            stride,
            padding,
            out_height,
            out_width,
        }
    }
}

impl super::Op for Im2ColOp {
    fn inputs(&self) -> &[ID] {
        std::slice::from_ref(&self.input)
    }

    fn outputs(&self) -> &[ID] {
        std::slice::from_ref(&self.output)
    }

    fn infer_shape(&self, _input_shapes: &[&Vec<u32>]) -> Result<Vec<u32>> {
        let out_h = (self.height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (self.width + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let col_h = self.batch * out_h * out_w;
        let col_w = self.kernel_size * self.kernel_size * self.channels;
        Ok(vec![col_h, col_w])
    }

    fn pipeline_keys(&self) -> Vec<&'static str> {
        vec!["im2col", "col2im"]
    }

    fn create_pipelines(
        &self,
        device: &wgpu::Device,
    ) -> Vec<(&'static str, wgpu::ComputePipeline)> {
        let fwd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("im2col_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let fwd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("im2col_pipeline"),
            layout: None,
            module: &fwd_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bwd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("col2im_shader"),
            source: wgpu::ShaderSource::Wgsl(BACKWARD_SHADER.into()),
        });

        let bwd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("col2im_pipeline"),
            layout: None,
            module: &bwd_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        vec![("im2col", fwd_pipeline), ("col2im", bwd_pipeline)]
    }

    fn buffers_needed(&self, shapes: &HashMap<ID, Vec<u32>>) -> Vec<(ID, &'static str, u32)> {
        let input_n: u32 = shapes[&self.input].iter().product();
        vec![(self.input, buffer_kind::GRAD, input_n)]
    }

    fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
        let pipeline = ctx
            .pipelines
            .get("im2col")
            .ok_or_else(|| VolticError::Internal("im2col pipeline not found".into()))?;

        let input_buf = ctx
            .buffers
            .get(&self.input)
            .ok_or_else(|| VolticError::Internal(format!("buffer not found: {:?}", self.input)))?;
        let output_buf = ctx
            .buffers
            .get(&self.output)
            .ok_or_else(|| VolticError::Internal(format!("buffer not found: {:?}", self.output)))?;

        let dims = Im2ColDims {
            batch: self.batch,
            channels: self.channels,
            height: self.height,
            width: self.width,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            out_height: self.out_height,
            out_width: self.out_width,
        };
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("im2col_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("im2col_bind_group"),
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

        let out_h = self.out_height;
        let out_w = self.out_width;
        let col_w = self.kernel_size * self.kernel_size * self.channels;
        let workgroups_x = (self.batch * out_h * out_w * col_w).div_ceil(256);

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("im2col_pass"),
                timestamp_writes: None,
            });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, 1, 1);

        Ok(())
    }

    fn backward(&self, ctx: &mut GpuContext) -> Result<()> {
        let pipeline = ctx
            .pipelines
            .get("col2im")
            .ok_or_else(|| VolticError::Internal("col2im pipeline not found".into()))?;

        let grad_col_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("im2col grad_col not found".into()))?;
        let grad_input_buf = ctx
            .training_buffers
            .get(&(self.input, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("im2col grad_input not found".into()))?;

        let dims = Im2ColDims {
            batch: self.batch,
            channels: self.channels,
            height: self.height,
            width: self.width,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            out_height: self.out_height,
            out_width: self.out_width,
        };
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("col2im_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("col2im_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_col_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let n_elements = self.batch * self.channels * self.height * self.width;
        let workgroups_x = n_elements.div_ceil(256);

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("col2im_pass"),
                timestamp_writes: None,
            });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, 1, 1);

        Ok(())
    }
}
