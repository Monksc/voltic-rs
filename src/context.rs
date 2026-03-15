use std::{
    collections::{HashMap, HashSet},
    sync::{OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use wgpu::util::DeviceExt;

use crate::{BufferKey, GpuContext, ID, Op, Result, Var, VolticError};

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

    /// Clear operations between epochs - keeps model weights
    /// Use when reusing the same model Var IDs across epochs
    pub fn clear() {
        let mut ctx = Self::get_mut();
        ctx.operations.clear();
        if let Some(gpu) = ctx.gpu_context_mut() {
            gpu.clear_buffers();
        }
    }

    /// Reset everything - for starting completely fresh
    /// WARNING: This clears all weights and buffers
    pub fn reset() {
        let mut ctx = Self::get_mut();
        // Reset everything - for starting fresh training
        ctx.operations.clear();
        ctx.latest_shape.clear();
        if let Some(gpu) = ctx.gpu_context_mut() {
            gpu.buffers.clear();
            gpu.training_buffers.clear();
        }
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
        let gpu = ctx.gpu_context.as_mut().ok_or(VolticError::GpuNotAvailable)?;
        let buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("buffer:{:?}", id)),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        gpu.buffers.insert(id, buffer);
        Ok(())
    }

    pub fn load_to_existing(gpu: &GpuContext, id: ID, data: Vec<Vec<f32>>) -> Result<()> {
        let flat: Vec<f32> = data.into_iter().flatten().collect();
        let buffer = gpu.buffers.get(&id).ok_or(VolticError::Internal(format!("No buffer for id {:?}", id)))?;
        gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&flat));
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

    pub fn collect_persistent(&self, starts: &[Var]) -> Vec<Var> {
        let operations = &self.operations;

        let mut ops_by_output: HashMap<ID, Vec<&Box<dyn Op>>> = HashMap::new();
        for op in operations.iter() {
            for output_id in op.outputs() {
                ops_by_output.entry(*output_id).or_default().push(op);
            }
        }

        let mut queue: Vec<ID> = starts.iter().map(|v| v.id()).collect();
        let mut visited: HashSet<ID> = HashSet::new();
        let mut all_ids: Vec<ID> = Vec::new();

        while let Some(id) = queue.pop() {
            if !visited.insert(id) {
                continue;
            }
            all_ids.push(id);

            if let Some(ops) = ops_by_output.get(&id) {
                for op in ops {
                    for input_id in op.inputs() {
                        queue.push(*input_id);
                    }
                }
            }
        }

        let gpu = match &self.gpu_context {
            Some(g) => g,
            None => return vec![],
        };

        let trainable_ids: HashSet<ID> = gpu
            .training_buffers
            .keys()
            .map(|(id, _)| *id)
            .collect();

        all_ids
            .into_iter()
            .filter(|id| trainable_ids.contains(id))
            .map(Var::from_id)
            .collect()
    }

    const CHECKPOINT_MAGIC: u32 = 0x564F4C54; // "VOLT" in hex

    pub fn save(&self, path: &str, vars: &[Var]) -> Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let file = File::create(path).map_err(|e| VolticError::Internal(e.to_string()))?;
        let mut writer = BufWriter::new(file);

        writer.write_all(&Self::CHECKPOINT_MAGIC.to_le_bytes()).map_err(|e| VolticError::Internal(e.to_string()))?;
        let num_vars = vars.len() as u32;
        writer.write_all(&num_vars.to_le_bytes()).map_err(|e| VolticError::Internal(e.to_string()))?;

        for var in vars {
            let shape = Context::shape(var.id()).ok_or(VolticError::EmptyShape)?;
            let data = Context::read((var.id(), ""))?;

            let shape_len = shape.len() as u32;
            writer.write_all(&shape_len.to_le_bytes()).map_err(|e| VolticError::Internal(e.to_string()))?;
            for dim in &shape {
                writer.write_all(&dim.to_le_bytes()).map_err(|e| VolticError::Internal(e.to_string()))?;
            }

            let data_len = data.len() as u32;
            writer.write_all(&data_len.to_le_bytes()).map_err(|e| VolticError::Internal(e.to_string()))?;
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            writer.write_all(&bytes).map_err(|e| VolticError::Internal(e.to_string()))?;
        }

        writer.flush().map_err(|e| VolticError::Internal(e.to_string()))?;
        Ok(())
    }

    pub fn restore(&mut self, path: &str, vars: &mut [Var]) -> Result<()> {
        use std::fs::File;
        use std::io::{BufReader, Read};

        let file = File::open(path).map_err(|e| VolticError::Internal(e.to_string()))?;
        let mut reader = BufReader::new(file);

        let mut magic_bytes = [0u8; 4];
        reader.read_exact(&mut magic_bytes).map_err(|e| VolticError::Internal(e.to_string()))?;
        let magic = u32::from_le_bytes(magic_bytes);
        if magic != Self::CHECKPOINT_MAGIC {
            return Err(VolticError::Internal("Invalid checkpoint file".into()));
        }

        let mut num_vars_bytes = [0u8; 4];
        reader.read_exact(&mut num_vars_bytes).map_err(|e| VolticError::Internal(e.to_string()))?;
        let num_vars = u32::from_le_bytes(num_vars_bytes) as usize;

        if num_vars != vars.len() {
            return Err(VolticError::Internal(format!(
                "Checkpoint has {} vars but expected {}",
                num_vars,
                vars.len()
            )));
        }

        // Get GPU context while we have &mut self
        let gpu = self.gpu_context.as_ref().ok_or(VolticError::GpuNotAvailable)?;

        for var in vars.iter_mut() {
            let mut shape_len_bytes = [0u8; 4];
            reader.read_exact(&mut shape_len_bytes).map_err(|e| VolticError::Internal(e.to_string()))?;
            let shape_len = u32::from_le_bytes(shape_len_bytes) as usize;

            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                let mut dim_bytes = [0u8; 4];
                reader.read_exact(&mut dim_bytes).map_err(|e| VolticError::Internal(e.to_string()))?;
                shape.push(u32::from_le_bytes(dim_bytes));
            }

            let mut data_len_bytes = [0u8; 4];
            reader.read_exact(&mut data_len_bytes).map_err(|e| VolticError::Internal(e.to_string()))?;
            let data_len = u32::from_le_bytes(data_len_bytes) as usize;

            let mut data = Vec::with_capacity(data_len);
            for _ in 0..data_len {
                let mut float_bytes = [0u8; 4];
                reader.read_exact(&mut float_bytes).map_err(|e| VolticError::Internal(e.to_string()))?;
                let val = f32::from_le_bytes(float_bytes);
                data.push(val);
            }

            let shaped_data = vec![data];
            Self::load_to_existing(gpu, var.id(), shaped_data)?;
        }

        Ok(())
    }
}
