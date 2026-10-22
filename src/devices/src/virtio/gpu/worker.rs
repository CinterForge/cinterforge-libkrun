use std::io::Read;
use std::os::fd::{AsRawFd, BorrowedFd, RawFd};
use std::sync::{Arc, Mutex};
use std::thread;

use nix::fcntl::{fcntl, FcntlArg, OFlag};
use utils::eventfd::EventFd;

#[cfg(target_os = "macos")]
use crossbeam_channel::Sender;
use rutabaga_gfx::{
    ResourceCreate3D, ResourceCreateBlob, RutabagaFence, Transfer3D,
    RUTABAGA_PIPE_BIND_RENDER_TARGET, RUTABAGA_PIPE_TEXTURE_2D,
};
#[cfg(target_os = "macos")]
use utils::worker_message::WorkerMessage;
use vm_memory::{GuestAddress, GuestMemoryMmap};

use super::super::descriptor_utils::{Reader, Writer};
use super::super::{DeviceQueue, GpuError, Queue as VirtQueue};
use super::protocol::{
    virtio_gpu_ctrl_hdr, virtio_gpu_mem_entry, GpuCommand, GpuResponse, VirtioGpuResult,
};
use super::virtio_gpu::VirtioGpu;
use crate::virtio::display::{DisplayControl, DisplayInfo};
use crate::virtio::fs::ExportTable;
use crate::virtio::gpu::protocol::{VIRTIO_GPU_FLAG_FENCE, VIRTIO_GPU_FLAG_INFO_RING_IDX};
use crate::virtio::gpu::virtio_gpu::VirtioGpuRing;
use crate::virtio::{InterruptTransport, VirtioShmRegion};
use krun_display::DisplayBackend;
use krun_display::Rect;

pub struct Worker {
    control_evt: EventFd,
    control_queue: Arc<Mutex<VirtQueue>>,
    mem: GuestMemoryMmap,
    interrupt: InterruptTransport,
    shm_region: VirtioShmRegion,
    virgl_flags: u32,
    render_server_fd: Option<RawFd>,
    #[cfg(target_os = "macos")]
    map_sender: Sender<WorkerMessage>,
    export_table: Option<ExportTable>,
    displays: Box<[DisplayInfo]>,
    display_backend: DisplayBackend<'static>,
    display_control: Arc<DisplayControl>,
}

/// Clones a queue eventfd so we have our own file description, set to blocking mode.
fn blocking_eventfd(event: &EventFd) -> EventFd {
    let evt = event.try_clone().unwrap();
    // SAFETY: evt is valid for the duration of the fcntl calls.
    let fd = unsafe { BorrowedFd::borrow_raw(evt.as_raw_fd()) };
    let flags = OFlag::from_bits_retain(fcntl(fd, FcntlArg::F_GETFL).unwrap()) & !OFlag::O_NONBLOCK;
    fcntl(fd, FcntlArg::F_SETFL(flags)).unwrap();
    evt
}

/// Completes every cursor command without acting on it, on its own thread.
/// There is no cursor plane in the display backend API (the host shows its own
/// pointer), but the guest driver blocks its atomic commits once the cursor
/// queue is full, so the queue has to keep moving.
pub fn spawn_cursor_drain(
    cursor_q: DeviceQueue,
    mem: GuestMemoryMmap,
    interrupt: InterruptTransport,
) {
    let evt = blocking_eventfd(&cursor_q.event);
    let mut queue = cursor_q.queue;
    thread::Builder::new()
        .name("gpu cursor".into())
        .spawn(move || loop {
            if let Err(e) = evt.read() {
                error!("Failed to read cursor_evt: {e:?}");
                continue;
            }
            let mut used_any = false;
            while let Some(head) = queue.pop(&mem) {
                if let Err(e) = queue.add_used(&mem, head.index, 0) {
                    error!("failed to add used elements to the cursor queue: {e:?}");
                }
                used_any = true;
            }
            if used_any {
                if let Err(e) = interrupt.try_signal_used_queue() {
                    error!("Error signaling cursor queue: {e:?}");
                }
            }
        })
        .unwrap();
}

impl Worker {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        control_q: DeviceQueue,
        mem: GuestMemoryMmap,
        interrupt: InterruptTransport,
        shm_region: VirtioShmRegion,
        virgl_flags: u32,
        render_server_fd: Option<RawFd>,
        #[cfg(target_os = "macos")] map_sender: Sender<WorkerMessage>,
        export_table: Option<ExportTable>,
        displays: Box<[DisplayInfo]>,
        display_backend: DisplayBackend<'static>,
        display_control: Arc<DisplayControl>,
    ) -> Self {
        Self {
            control_evt: blocking_eventfd(&control_q.event),
            control_queue: Arc::new(Mutex::new(control_q.queue)),
            mem,
            interrupt,
            shm_region,
            virgl_flags,
            render_server_fd,
            #[cfg(target_os = "macos")]
            map_sender,
            export_table,
            displays,
            display_backend,
            display_control,
        }
    }

    pub fn run(self) {
        thread::Builder::new()
            .name("gpu worker".into())
            .spawn(|| self.work())
            .unwrap();
    }

    fn work(mut self) {
        #[cfg(target_os = "macos")]
        utils::macos::set_thread_qos_user_interactive();
        let mut virtio_gpu = VirtioGpu::new(
            self.mem.clone(),
            self.control_queue.clone(),
            self.interrupt.clone(),
            self.virgl_flags,
            self.render_server_fd,
            #[cfg(target_os = "macos")]
            self.map_sender.clone(),
            self.export_table.take(),
            self.displays.clone(),
            self.display_backend,
        );

        // Resizes requested before the driver started: nothing to signal, it
        // has not asked for the display configuration yet.
        self.apply_display_resizes(&mut virtio_gpu);

        loop {
            if let Err(e) = self.control_evt.read() {
                error!("Failed to read control_evt: {e:?}");
                continue;
            }
            if self.apply_display_resizes(&mut virtio_gpu) {
                self.display_control.raise_display_event();
                if let Err(e) = self.interrupt.try_signal_config_change() {
                    error!("Error signaling display change: {e:?}");
                }
            }
            if self.process_queue(&mut virtio_gpu, &self.control_queue.clone()) {
                if let Err(e) = self.interrupt.try_signal_used_queue() {
                    error!("Error signaling queue: {e:?}");
                }
            }
        }
    }

    /// Applies every queued display resize (a live window resize queues many);
    /// returns whether the guest-visible configuration changed.
    fn apply_display_resizes(&self, virtio_gpu: &mut VirtioGpu) -> bool {
        let mut changed = false;
        for resize in self.display_control.requests().try_iter() {
            match virtio_gpu.resize_display(&resize) {
                Ok(c) => changed |= c,
                Err(e) => error!("display resize {resize:?} failed: {e:?}"),
            }
        }
        changed
    }

    fn process_gpu_command(
        &mut self,
        virtio_gpu: &mut VirtioGpu,
        mem: &GuestMemoryMmap,
        hdr: virtio_gpu_ctrl_hdr,
        cmd: GpuCommand,
        reader: &mut Reader,
    ) -> VirtioGpuResult {
        virtio_gpu.force_ctx_0();

        match cmd {
            GpuCommand::GetDisplayInfo => virtio_gpu.display_info(),
            GpuCommand::GetEdid(info) => virtio_gpu.get_edid(info.scanout),
            GpuCommand::ResourceCreate2d(info) => {
                let resource_id = info.resource_id;

                let resource_create_3d = ResourceCreate3D {
                    target: RUTABAGA_PIPE_TEXTURE_2D,
                    format: info.format,
                    bind: RUTABAGA_PIPE_BIND_RENDER_TARGET,
                    width: info.width,
                    height: info.height,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: 0,
                };

                virtio_gpu.resource_create_3d(resource_id, resource_create_3d)
            }
            GpuCommand::ResourceUnref(info) => virtio_gpu.unref_resource(info.resource_id),
            GpuCommand::SetScanout(info) => virtio_gpu.set_scanout(
                info.scanout_id,
                info.resource_id,
                info.r.width,
                info.r.height,
            ),
            GpuCommand::ResourceFlush(info) => {
                let rect = Rect {
                    x: info.r.x,
                    y: info.r.y,
                    width: info.r.width,
                    height: info.r.height,
                };
                virtio_gpu.flush_resource(info.resource_id, rect)
            }
            GpuCommand::TransferToHost2d(info) => {
                let resource_id = info.resource_id;
                let transfer = Transfer3D::new_2d(info.r.x, info.r.y, info.r.width, info.r.height);
                virtio_gpu.transfer_write(0, resource_id, transfer)
            }
            GpuCommand::ResourceAttachBacking(info) => {
                let available_bytes = reader.available_bytes();
                if available_bytes != 0 {
                    let entry_count = info.nr_entries as usize;
                    let mut vecs = Vec::with_capacity(entry_count);
                    for _ in 0..entry_count {
                        match reader.read_obj::<virtio_gpu_mem_entry>() {
                            Ok(entry) => {
                                let addr = GuestAddress(entry.addr);
                                let len = entry.length as usize;
                                vecs.push((addr, len))
                            }
                            Err(_) => return Err(GpuResponse::ErrUnspec),
                        }
                    }
                    virtio_gpu.attach_backing(info.resource_id, mem, vecs)
                } else {
                    error!("missing data for command {cmd:?}");
                    Err(GpuResponse::ErrUnspec)
                }
            }
            GpuCommand::ResourceDetachBacking(info) => virtio_gpu.detach_backing(info.resource_id),
            GpuCommand::UpdateCursor(_info) => {
                panic!("virtio_gpu: GpuCommand:UpdateCursor unimplemented");
            }
            GpuCommand::MoveCursor(_info) => {
                panic!("virtio_gpu: GpuCommand::MoveCursor unimplemented");
            }
            GpuCommand::ResourceAssignUuid(info) => {
                let resource_id = info.resource_id;
                virtio_gpu.resource_assign_uuid(resource_id)
            }
            GpuCommand::GetCapsetInfo(info) => virtio_gpu.get_capset_info(info.capset_index),
            GpuCommand::GetCapset(info) => {
                virtio_gpu.get_capset(info.capset_id, info.capset_version)
            }

            GpuCommand::CtxCreate(info) => {
                let context_name: Option<String> = String::from_utf8(info.debug_name.to_vec()).ok();
                virtio_gpu.create_context(hdr.ctx_id, info.context_init, context_name.as_deref())
            }
            GpuCommand::CtxDestroy(_info) => virtio_gpu.destroy_context(hdr.ctx_id),
            GpuCommand::CtxAttachResource(info) => {
                virtio_gpu.context_attach_resource(hdr.ctx_id, info.resource_id)
            }
            GpuCommand::CtxDetachResource(info) => {
                virtio_gpu.context_detach_resource(hdr.ctx_id, info.resource_id)
            }
            GpuCommand::ResourceCreate3d(info) => {
                let resource_id = info.resource_id;
                let resource_create_3d = ResourceCreate3D {
                    target: info.target,
                    format: info.format,
                    bind: info.bind,
                    width: info.width,
                    height: info.height,
                    depth: info.depth,
                    array_size: info.array_size,
                    last_level: info.last_level,
                    nr_samples: info.nr_samples,
                    flags: info.flags,
                };

                virtio_gpu.resource_create_3d(resource_id, resource_create_3d)
            }
            GpuCommand::TransferToHost3d(info) => {
                let ctx_id = hdr.ctx_id;
                let resource_id = info.resource_id;

                let transfer = Transfer3D {
                    x: info.box_.x,
                    y: info.box_.y,
                    z: info.box_.z,
                    w: info.box_.w,
                    h: info.box_.h,
                    d: info.box_.d,
                    level: info.level,
                    stride: info.stride,
                    layer_stride: info.layer_stride,
                    offset: info.offset,
                };

                virtio_gpu.transfer_write(ctx_id, resource_id, transfer)
            }
            GpuCommand::TransferFromHost3d(info) => {
                let ctx_id = hdr.ctx_id;
                let resource_id = info.resource_id;

                let transfer = Transfer3D {
                    x: info.box_.x,
                    y: info.box_.y,
                    z: info.box_.z,
                    w: info.box_.w,
                    h: info.box_.h,
                    d: info.box_.d,
                    level: info.level,
                    stride: info.stride,
                    layer_stride: info.layer_stride,
                    offset: info.offset,
                };

                virtio_gpu.transfer_read(ctx_id, resource_id, transfer, None)
            }
            GpuCommand::CmdSubmit3d(info) => {
                if reader.available_bytes() != 0 {
                    let num_in_fences = info.num_in_fences as usize;
                    let cmd_size = info.size as usize;
                    let mut cmd_buf = vec![0; cmd_size];
                    let mut fence_ids: Vec<u64> = Vec::with_capacity(num_in_fences);

                    for _ in 0..num_in_fences {
                        match reader.read_obj::<u64>() {
                            Ok(fence_id) => {
                                fence_ids.push(fence_id);
                            }
                            Err(_) => return Err(GpuResponse::ErrUnspec),
                        }
                    }

                    if reader.read_exact(&mut cmd_buf[..]).is_ok() {
                        virtio_gpu.submit_command(hdr.ctx_id, &mut cmd_buf[..], &fence_ids)
                    } else {
                        Err(GpuResponse::ErrInvalidParameter)
                    }
                } else {
                    // Silently accept empty command buffers to allow for
                    // benchmarking.
                    Ok(GpuResponse::OkNoData)
                }
            }
            GpuCommand::ResourceCreateBlob(info) => {
                let resource_id = info.resource_id;
                let ctx_id = hdr.ctx_id;

                let resource_create_blob = ResourceCreateBlob {
                    blob_mem: info.blob_mem,
                    blob_flags: info.blob_flags,
                    blob_id: info.blob_id,
                    size: info.size,
                };

                let entry_count = info.nr_entries;
                if reader.available_bytes() == 0 && entry_count > 0 {
                    return Err(GpuResponse::ErrUnspec);
                }

                let mut vecs = Vec::with_capacity(entry_count as usize);
                for _ in 0..entry_count {
                    match reader.read_obj::<virtio_gpu_mem_entry>() {
                        Ok(entry) => {
                            let addr = GuestAddress(entry.addr);
                            let len = entry.length as usize;
                            vecs.push((addr, len))
                        }
                        Err(_) => return Err(GpuResponse::ErrUnspec),
                    }
                }

                virtio_gpu.resource_create_blob(
                    ctx_id,
                    resource_id,
                    resource_create_blob,
                    vecs,
                    mem,
                )
            }
            GpuCommand::SetScanoutBlob(info) => {
                let rect = Rect {
                    x: info.r.x,
                    y: info.r.y,
                    width: info.r.width,
                    height: info.r.height,
                };
                // Only single-plane RGB formats can be scanned out.
                virtio_gpu.set_scanout_blob(
                    info.scanout_id,
                    info.resource_id,
                    info.width,
                    info.height,
                    info.format,
                    info.strides[0],
                    info.offsets[0],
                    rect,
                )
            }
            GpuCommand::ResourceMapBlob(info) => {
                let resource_id = info.resource_id;
                let offset = info.offset;
                virtio_gpu.resource_map_blob(resource_id, &self.shm_region, offset)
            }
            GpuCommand::ResourceUnmapBlob(info) => {
                let resource_id = info.resource_id;
                virtio_gpu.resource_unmap_blob(resource_id, &self.shm_region)
            }
        }
    }

    fn process_queue(
        &mut self,
        virtio_gpu: &mut VirtioGpu,
        control_queue: &Arc<Mutex<VirtQueue>>,
    ) -> bool {
        let mut used_any = false;
        let mem = self.mem.clone();

        loop {
            let head = control_queue.lock().unwrap().pop(&mem);

            if let Some(head) = head {
                self.display_control.note_ctrl_command();
                let mut reader = Reader::new(&mem, head.clone())
                    .map_err(GpuError::QueueReader)
                    .unwrap();
                let mut writer = Writer::new(&mem, head.clone())
                    .map_err(GpuError::QueueWriter)
                    .unwrap();

                let mut resp = Err(GpuResponse::ErrUnspec);
                let mut gpu_cmd = None;
                let mut ctrl_hdr = None;
                let mut len = 0;

                match GpuCommand::decode(&mut reader) {
                    Ok((hdr, cmd)) => {
                        resp = self.process_gpu_command(virtio_gpu, &mem, hdr, cmd, &mut reader);
                        ctrl_hdr = Some(hdr);
                        gpu_cmd = Some(cmd);
                    }
                    Err(e) => debug!("descriptor decode error: {e:?}"),
                }

                let mut gpu_response = match resp {
                    Ok(gpu_response) => gpu_response,
                    Err(gpu_response) => {
                        debug!("{gpu_cmd:?} -> {gpu_response:?}");
                        gpu_response
                    }
                };

                let mut add_to_queue = true;

                if writer.available_bytes() != 0 {
                    let mut fence_id = 0;
                    let mut ctx_id = 0;
                    let mut flags = 0;
                    let mut ring_idx = 0;
                    let mut fence_created = false;
                    if let Some(_cmd) = gpu_cmd {
                        let ctrl_hdr = ctrl_hdr.unwrap();
                        if ctrl_hdr.flags & VIRTIO_GPU_FLAG_FENCE != 0 {
                            flags = ctrl_hdr.flags;
                            fence_id = ctrl_hdr.fence_id;
                            ctx_id = ctrl_hdr.ctx_id;
                            ring_idx = ctrl_hdr.ring_idx;

                            let fence = RutabagaFence {
                                flags,
                                fence_id,
                                ctx_id,
                                ring_idx,
                            };
                            gpu_response = match virtio_gpu.create_fence(fence) {
                                Ok(_) => {
                                    fence_created = true;
                                    gpu_response
                                }
                                Err(fence_resp) => {
                                    warn!("create_fence {fence_id} -> {fence_resp:?}");
                                    fence_resp
                                }
                            };
                        }
                    }

                    // Prepare the response now, even if it is going to wait until
                    // fence is complete.
                    match gpu_response.encode(flags, fence_id, ctx_id, ring_idx, &mut writer) {
                        Ok(l) => len = l,
                        Err(e) => debug!("ctrl queue response encode error: {e:?}"),
                    }

                    // A fence the renderer refused will never signal: answer now
                    // (still flagged, so the guest retires its fence) instead of
                    // parking the descriptor forever and wedging the queue.
                    if fence_created {
                        let ring = match flags & VIRTIO_GPU_FLAG_INFO_RING_IDX {
                            0 => VirtioGpuRing::Global,
                            _ => VirtioGpuRing::ContextSpecific { ctx_id, ring_idx },
                        };

                        add_to_queue = virtio_gpu.process_fence(ring, fence_id, head.index, len);
                    }
                }

                if add_to_queue {
                    if let Err(e) = control_queue
                        .lock()
                        .unwrap()
                        .add_used(&mem, head.index, len)
                    {
                        error!("failed to add used elements to the queue: {e:?}");
                    }
                    used_any = true;
                }
            } else {
                break;
            }
        }

        debug!("gpu: process_queue exit");
        used_any
    }
}
