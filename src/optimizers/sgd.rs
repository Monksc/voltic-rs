use crate::{ID, Result, VARIABLES_ID_COUNTER, VolticError, buffer_kind, context::Context};
use std::{borrow::Cow, sync::atomic::Ordering};
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("../ops/shaders/sgd.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SgdDims {
    lr: f32,
    n: u32,
}

pub struct Sgd {
    lr: f32,
    cached_ids: Vec<(ID, u32)>, // (id, n_elements)
    cache_generation: u64,
    pipeline: Option<wgpu::ComputePipeline>,
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            cached_ids: Vec::new(),
            cache_generation: u64::MAX, // force build on first step
            pipeline: None,
        }
    }

    pub fn update_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn invalidate(&mut self) {
        self.cached_ids.clear();
        self.cache_generation = u64::MAX;
    }

    fn ensure_pipeline(&mut self, device: &wgpu::Device) {
        if self.pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sgd_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });

        self.pipeline = Some(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sgd_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }),
        );
    }

    fn build_cache(&mut self) {
        let ctx = Context::get();

        // Collect all unique input IDs that have a grad buffer
        let mut seen = std::collections::HashSet::new();
        self.cached_ids = ctx
            .operations()
            .iter()
            .flat_map(|op| op.inputs())
            .filter_map(|id| {
                if !seen.insert(*id) {
                    return None; // deduplicate
                }
                let has_grad = ctx
                    .gpu_context()
                    .as_ref()
                    .and_then(|gpu| gpu.training_buffers.get(&(*id, buffer_kind::GRAD)))
                    .is_some();

                if has_grad {
                    let n = ctx.shape_total_with_context(*id).unwrap_or(0);
                    Some((*id, n))
                } else {
                    None
                }
            })
            .collect();

        self.cache_generation = VARIABLES_ID_COUNTER.load(Ordering::Relaxed);
    }

    pub fn step(&mut self) -> Result<()> {
        // Rebuild cache if graph has changed
        let generation = VARIABLES_ID_COUNTER.load(Ordering::Relaxed);
        if self.cached_ids.is_empty() || self.cache_generation != generation {
            self.build_cache();
        }

        let mut ctx = Context::get_mut();
        let gpu = ctx
            .gpu_context_mut()
            .as_mut()
            .ok_or(VolticError::GpuNotAvailable)?;

        self.ensure_pipeline(&gpu.device);
        let pipeline = self.pipeline.as_ref().unwrap();

        for &(id, n) in &self.cached_ids {
            let weight_buf = gpu
                .buffers
                .get(&id)
                .ok_or_else(|| VolticError::Internal(format!("no weight buffer: {:?}", id)))?;
            let grad_buf = gpu
                .training_buffers
                .get(&(id, buffer_kind::GRAD))
                .ok_or_else(|| VolticError::Internal(format!("no grad buffer: {:?}", id)))?;

            let dims = SgdDims { lr: self.lr, n };
            let dims_buf = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sgd_dims"),
                    contents: bytemuck::bytes_of(&dims),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
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

            let mut pass = gpu
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sgd_pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }

        gpu.flush();
        Ok(())
    }
}
