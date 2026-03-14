use std::collections::HashMap;

use wgpu::{Buffer, util::DeviceExt};

use crate::{BufferKey, ID, Op, Result, VolticError, init};

#[derive(Debug)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub encoder: wgpu::CommandEncoder,
    pub buffers: HashMap<ID, wgpu::Buffer>,
    pub training_buffers: HashMap<BufferKey, wgpu::Buffer>,
    pub pipelines: HashMap<String, wgpu::ComputePipeline>,
}

impl GpuContext {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|_| VolticError::GpuNotAvailable)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| VolticError::GpuBufferError(e.to_string()))?;

        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voltic::encoder"),
        });

        Ok(Self {
            device,
            queue,
            encoder,
            buffers: Default::default(),
            training_buffers: Default::default(),
            pipelines: Default::default(),
        })
    }

    pub fn create_storage_buffer(&mut self, buffer_label: &str, n_elements: u32) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(buffer_label),
            size: (n_elements * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn allocate_buffers(
        &mut self,
        shapes: &HashMap<ID, Vec<u32>>,
        ops: &Vec<Box<dyn Op>>,
    ) -> Result<()> {
        for (id, shape) in shapes {
            if !self.buffers.contains_key(id) {
                let n_elements: u32 = shape.iter().product();
                let data = init::xavier_flat(n_elements);
                let buffer = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("buffer:{:?}", id)),
                        contents: bytemuck::cast_slice(&data),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    });
                self.buffers.insert(*id, buffer);
            }
        }

        for op in ops {
            for (id, kind, n_elements) in op.buffers_needed(shapes) {
                if !self.training_buffers.contains_key(&(id, kind)) {
                    let buffer = self
                        .create_storage_buffer(&format!("buffer:{}:{:?}", kind, id), n_elements);
                    self.training_buffers.insert((id, kind), buffer);
                }
            }
        }
        Ok(())
    }

    pub fn flush(&mut self) {
        let old_encoder = std::mem::replace(
            &mut self.encoder,
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("voltic::encoder"),
                }),
        );

        self.queue.submit(std::iter::once(old_encoder.finish()));
    }

    pub fn prepare_ops(&mut self, operations: &Vec<Box<dyn Op>>) {
        for op in operations {
            for key in op.pipeline_keys() {
                if !self.pipelines.contains_key(key) {
                    for (k, pipeline) in op.create_pipelines(&self.device) {
                        self.pipelines.insert(k.to_string(), pipeline);
                    }
                }
            }
        }
    }

    pub fn execute_ops(&mut self, operations: &Vec<Box<dyn Op>>) -> Result<()> {
        for op in operations {
            op.forward_gpu(self)?;
        }
        self.flush();
        Ok(())
    }

    pub fn clear_buffers(&mut self) {
        self.training_buffers.clear();
    }
}
