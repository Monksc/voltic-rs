use std::{
    collections::HashMap,
    sync::{OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use wgpu::util::DeviceExt;

use crate::{BufferKey, GpuContext, ID, Op, Result, VolticError};

#[derive(Debug)]
pub struct Context {
    latest_shape: HashMap<ID, Vec<u32>>,
    operations: Vec<Box<dyn Op>>,

    gpu_context: Option<GpuContext>,
}

impl Context {
    pub fn instance() -> &'static RwLock<Context> {
        static INSTANCE: OnceLock<RwLock<Context>> = OnceLock::new();
        INSTANCE.get_or_init(|| {
            RwLock::new(Context {
                latest_shape: Default::default(),
                operations: Vec::new(),
                gpu_context: None,
            })
        })
    }

    pub fn get() -> RwLockReadGuard<'static, Context> {
        Self::instance().read().unwrap()
    }

    pub fn get_mut() -> RwLockWriteGuard<'static, Context> {
        Self::instance().write().unwrap()
    }

    pub async fn init_gpu_async() -> Result<()> {
        let mut ctx = Self::get_mut();
        ctx.gpu_context = Some(GpuContext::new().await?);

        Ok(())
    }

    pub fn init_gpu() -> Result<()> {
        pollster::block_on(async { Self::init_gpu_async().await })
    }

    pub fn shapes() -> HashMap<ID, Vec<u32>> {
        Self::get().latest_shape.clone()
    }

    pub fn shapes_count() -> usize {
        Self::get().latest_shape.len()
    }

    pub fn shape(id: ID) -> Option<Vec<u32>> {
        Self::get().latest_shape.get(&id).map(|x| x.clone())
    }

    pub fn shape_total_with_context(&self, id: ID) -> Option<u32> {
        self.latest_shape.get(&id).map(|x| x.into_iter().product())
    }

    pub fn shape_total(id: ID) -> Option<u32> {
        Self::get().shape_total_with_context(id)
    }

    pub fn insert_shape(id: ID, shape: Vec<u32>) -> Option<Vec<u32>> {
        Self::get_mut().latest_shape.insert(id, shape)
    }

    pub fn push_operation(op: Box<dyn Op>) {
        Self::get_mut().operations.push(op)
    }

    pub fn operations<'a>(&'a self) -> &'a Vec<Box<dyn Op>> {
        &self.operations
    }

    pub fn gpu_context<'a>(&'a self) -> &'a Option<GpuContext> {
        &self.gpu_context
    }

    pub fn gpu_context_mut<'a>(&'a mut self) -> &'a mut Option<GpuContext> {
        &mut self.gpu_context
    }

    pub fn load(id: ID, data: Vec<Vec<f32>>) -> Result<()> {
        let flat: Vec<f32> = data.into_iter().flatten().collect();
        let mut ctx = Self::get_mut();
        let gpu = ctx
            .gpu_context
            .as_mut()
            .ok_or(VolticError::GpuNotAvailable)?;

        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("buffer:{:?}", id)),
                contents: bytemuck::cast_slice(&flat),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        gpu.buffers.insert(id, buffer);
        Ok(())
    }

    pub fn read(id: BufferKey) -> Result<Vec<f32>> {
        let ctx = Self::get();
        let gpu = ctx
            .gpu_context
            .as_ref()
            .ok_or(VolticError::GpuNotAvailable)?;

        let src_buf = gpu
            .training_buffers
            .get(&id)
            .or_else(|| gpu.buffers.get(&id.0))
            .ok_or_else(|| VolticError::Internal(format!("no buffer for var {:?}", id)))?;

        let size = src_buf.size();

        // Staging buffer — CPU readable
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // One-off encoder just for the copy
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(src_buf, 0, &staging, 0, size);
        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Map and read back
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv()
            .unwrap()
            .map_err(|e| VolticError::GpuBufferError(e.to_string()))?;

        let data = slice.get_mapped_range();
        Ok(bytemuck::cast_slice(&data).to_vec())
    }

    pub fn allocate_buffers() -> Result<()> {
        let mut ctx = Self::get_mut();

        let shapes = ctx.latest_shape.clone();
        let ops = std::mem::take(&mut ctx.operations);

        let Some(ref mut gpu) = ctx.gpu_context else {
            return Err(VolticError::GpuNotAvailable);
        };

        gpu.allocate_buffers(&shapes, &ops)?;

        ctx.operations = ops;
        Ok(())
    }

    pub fn prepare() -> Result<()> {
        let mut ctx = Self::get_mut();
        let ops = std::mem::take(&mut ctx.operations);
        let gpu = ctx
            .gpu_context
            .as_mut()
            .ok_or(VolticError::GpuNotAvailable)?;
        gpu.prepare_ops(&ops);
        ctx.operations = ops;
        Ok(())
    }

    pub fn run() -> Result<()> {
        let mut ctx = Self::get_mut();
        let ops = std::mem::take(&mut ctx.operations);
        let gpu = ctx
            .gpu_context
            .as_mut()
            .ok_or(VolticError::GpuNotAvailable)?;
        gpu.execute_ops(&ops)?;
        ctx.operations = ops;
        Ok(())
    }

    pub fn backward() -> Result<()> {
        let mut ctx = Self::get_mut();
        let ops = std::mem::take(&mut ctx.operations);

        let gpu = ctx
            .gpu_context
            .as_mut()
            .ok_or(VolticError::GpuNotAvailable)?;

        for op in ops.iter().rev() {
            op.backward(gpu)?;
        }

        gpu.flush();

        ctx.operations = ops;

        Ok(())
    }
}
