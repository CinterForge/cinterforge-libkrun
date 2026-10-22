use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use super::edid::EdidInfo;
use crossbeam_channel::{unbounded, Receiver, Sender};
use krun_display::{
    DisplayBackendBasicFramebuffer, DisplayBackendError, DisplayBackendNew, Rect, ResourceFormat,
};
use utils::eventfd::EventFd;
use virtio_bindings::virtio_gpu::VIRTIO_GPU_MAX_SCANOUTS;

#[derive(Clone, Debug)]
pub struct DisplayInfo {
    pub width: u32,
    pub height: u32,
    pub edid: DisplayInfoEdid,
}

impl DisplayInfo {
    pub fn edid_bytes(&self) -> Box<[u8]> {
        match &self.edid {
            DisplayInfoEdid::Provided(edid_bytes) => edid_bytes.clone(),
            DisplayInfoEdid::Generated(edid_params) => {
                let edid_info = EdidInfo::new(self.width, self.height, edid_params);
                edid_info.bytes()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum DisplayInfoEdid {
    Generated(EdidParams),
    Provided(Box<[u8]>),
}

#[derive(Debug, Clone, Copy)]
pub struct EdidParams {
    pub refresh_rate: u32,
    pub physical_size: PhysicalSize,
}

impl Default for EdidParams {
    fn default() -> Self {
        EdidParams {
            refresh_rate: 60,
            physical_size: PhysicalSize::Dpi(300),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PhysicalSize {
    Dpi(u32),
    DimensionsMillimeters(u16, u16),
}

impl DisplayInfo {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            edid: DisplayInfoEdid::Generated(EdidParams::default()),
        }
    }

    /// Applies a runtime resize: the new size, and the new physical size in the
    /// generated EDID. Returns whether anything the guest can see changed.
    pub fn resize(&mut self, resize: &DisplayResize) -> Result<bool, DisplayResizeError> {
        resize.check()?;
        let DisplayInfoEdid::Generated(params) = &mut self.edid else {
            return Err(DisplayResizeError::FixedEdid);
        };
        let physical_size = PhysicalSize::DimensionsMillimeters(resize.width_mm, resize.height_mm);
        let changed = self.width != resize.width
            || self.height != resize.height
            || params.physical_size != physical_size;
        self.width = resize.width;
        self.height = resize.height;
        params.physical_size = physical_size;
        Ok(changed)
    }
}

/// The largest width, height (pixels) and physical size (mm) a generated EDID
/// can describe: its detailed timing descriptor has 12-bit fields.
pub const MAX_RESIZE_DIMENSION: u32 = 4095;

/// A display size change requested at runtime (krun_display_resize).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisplayResize {
    pub display_id: u32,
    pub width: u32,
    pub height: u32,
    pub width_mm: u16,
    pub height_mm: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayResizeError {
    /// Zero size, or larger than MAX_RESIZE_DIMENSION.
    InvalidSize,
    /// No display with this id.
    InvalidDisplay,
    /// The display has a caller-provided EDID (krun_display_set_edid), which
    /// cannot follow a new size.
    FixedEdid,
}

impl DisplayResize {
    fn check(&self) -> Result<(), DisplayResizeError> {
        let fits = |v: u32| (1..=MAX_RESIZE_DIMENSION).contains(&v);
        if !fits(self.width)
            || !fits(self.height)
            || u32::from(self.width_mm) > MAX_RESIZE_DIMENSION
            || u32::from(self.height_mm) > MAX_RESIZE_DIMENSION
        {
            return Err(DisplayResizeError::InvalidSize);
        }
        Ok(())
    }
}

/// VIRTIO_GPU_EVENT_DISPLAY: the display configuration changed, the driver
/// should fetch it again (virtio spec, GPU device configuration layout).
pub const VIRTIO_GPU_EVENT_DISPLAY: u32 = 1 << 0;

#[derive(Default)]
struct DisplayEvents {
    /// `events_read` of the device configuration.
    pending: u32,
    /// Number of display events raised so far.
    raised: u64,
    /// `raised` when the driver last read `events_read`.
    seen: u64,
}

/// Runtime display changes of a running VM. Requests come from any host thread
/// and are queued; the GPU worker applies them between control commands, so
/// the display list is only ever touched by the worker, then raises
/// VIRTIO_GPU_EVENT_DISPLAY with a configuration change interrupt.
pub struct DisplayControl {
    /// Per display: whether its EDID is generated (and so can follow a resize).
    resizable: Box<[bool]>,
    sender: Sender<DisplayResize>,
    receiver: Receiver<DisplayResize>,
    /// The control queue's eventfd once the device is active; written to wake
    /// the worker, which waits on it.
    waker: Mutex<Option<EventFd>>,
    events: Mutex<DisplayEvents>,
    /// Control-queue commands the worker has taken (krun_gpu_get_activity).
    ctrl_commands: AtomicU64,
}

impl DisplayControl {
    pub fn new(displays: &[DisplayInfo]) -> Self {
        let (sender, receiver) = unbounded();
        Self {
            resizable: displays
                .iter()
                .map(|d| matches!(d.edid, DisplayInfoEdid::Generated(_)))
                .collect(),
            sender,
            receiver,
            waker: Mutex::new(None),
            events: Mutex::default(),
            ctrl_commands: AtomicU64::new(0),
        }
    }

    /// Queues a resize for the GPU worker. Before the device is activated the
    /// request waits in the queue and is applied before the first command.
    pub fn request_resize(&self, resize: DisplayResize) -> Result<(), DisplayResizeError> {
        match self.resizable.get(resize.display_id as usize) {
            None => return Err(DisplayResizeError::InvalidDisplay),
            Some(false) => return Err(DisplayResizeError::FixedEdid),
            Some(true) => resize.check()?,
        }
        // Cannot fail: self holds a receiver.
        let _ = self.sender.send(resize);
        if let Some(waker) = self.waker.lock().unwrap().as_ref() {
            if let Err(e) = waker.write(1) {
                error!("Failed to wake the gpu worker for a display resize: {e:?}");
            }
        }
        Ok(())
    }

    pub(crate) fn set_waker(&self, waker: EventFd) {
        *self.waker.lock().unwrap() = Some(waker);
    }

    pub(crate) fn requests(&self) -> Receiver<DisplayResize> {
        self.receiver.clone()
    }

    /// The worker took a control-queue command off the queue.
    pub(crate) fn note_ctrl_command(&self) {
        self.ctrl_commands.fetch_add(1, Ordering::Relaxed);
    }

    /// Control-queue commands taken so far (cursor queue excluded).
    pub fn ctrl_commands(&self) -> u64 {
        self.ctrl_commands.load(Ordering::Relaxed)
    }

    fn events(&self) -> MutexGuard<'_, DisplayEvents> {
        self.events.lock().unwrap()
    }

    /// Marks a display change for the driver; the caller then raises the
    /// configuration change interrupt.
    pub(crate) fn raise_display_event(&self) {
        let mut events = self.events();
        events.pending |= VIRTIO_GPU_EVENT_DISPLAY;
        events.raised += 1;
    }

    /// `events_read`, as read by the driver.
    pub(crate) fn read_events(&self) -> u32 {
        let mut events = self.events();
        events.seen = events.raised;
        events.pending
    }

    /// A driver write to `events_clear`. A display event raised after the
    /// driver's last `events_read` stays pending: the driver acknowledges what
    /// it read, and the interrupt of the newer event makes it read again.
    pub(crate) fn clear_events(&self, clear: u32) {
        let mut events = self.events();
        let keep = if events.raised != events.seen {
            VIRTIO_GPU_EVENT_DISPLAY
        } else {
            0
        };
        events.pending &= !(clear & !keep);
    }
}

pub const MAX_DISPLAYS: usize = VIRTIO_GPU_MAX_SCANOUTS as usize;

pub struct NoopDisplayBackend;

impl DisplayBackendNew<()> for NoopDisplayBackend {
    fn new(_userdata: Option<&()>) -> Self {
        Self
    }
}

impl DisplayBackendBasicFramebuffer for NoopDisplayBackend {
    fn configure_scanout(
        &mut self,
        _scanout_id: u32,
        _display_width: u32,
        _display_height: u32,
        _width: u32,
        _height: u32,
        _format: ResourceFormat,
    ) -> Result<(), DisplayBackendError> {
        Err(DisplayBackendError::InvalidScanoutId)
    }

    fn disable_scanout(&mut self, _scanout_id: u32) -> Result<(), DisplayBackendError> {
        Err(DisplayBackendError::InvalidScanoutId)
    }

    fn alloc_frame(&mut self, _scanout_id: u32) -> Result<(u32, &mut [u8]), DisplayBackendError> {
        Err(DisplayBackendError::InvalidScanoutId)
    }

    fn present_frame(
        &mut self,
        _scanout_id: u32,
        _frame_id: u32,
        _rect: Option<&Rect>,
    ) -> Result<(), DisplayBackendError> {
        Err(DisplayBackendError::InvalidScanoutId)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The preferred mode in an EDID: (width, height, refresh Hz, width mm, height mm).
    fn preferred_mode(edid: &[u8]) -> (u32, u32, f64, u16, u16) {
        assert_eq!(edid.len(), 128);
        assert_eq!(edid.iter().fold(0u8, |sum, b| sum.wrapping_add(*b)), 0);
        let dtd = &edid[54..72];
        let clock_hz = u16::from_le_bytes([dtd[0], dtd[1]]) as f64 * 10_000.0;
        let hi = |lsb: u8, msb: u8| u32::from(lsb) | (u32::from(msb) << 8);
        let width = hi(dtd[2], dtd[4] >> 4);
        let hblank = hi(dtd[3], dtd[4] & 0xf);
        let height = hi(dtd[5], dtd[7] >> 4);
        let vblank = hi(dtd[6], dtd[7] & 0xf);
        let width_mm = hi(dtd[12], dtd[14] >> 4) as u16;
        let height_mm = hi(dtd[13], dtd[14] & 0xf) as u16;
        let refresh = clock_hz / f64::from((width + hblank) * (height + vblank));
        (width, height, refresh, width_mm, height_mm)
    }

    fn resize(width: u32, height: u32, width_mm: u16, height_mm: u16) -> DisplayResize {
        DisplayResize {
            display_id: 0,
            width,
            height,
            width_mm,
            height_mm,
        }
    }

    #[test]
    fn resize_regenerates_edid() {
        let mut display = DisplayInfo::new(1280, 800);
        assert_eq!(display.resize(&resize(1280, 800, 339, 212)), Ok(true));
        assert_eq!(display.resize(&resize(1280, 800, 339, 212)), Ok(false));
        let (w, h, hz, wmm, hmm) = preferred_mode(&display.edid_bytes());
        assert_eq!((w, h, wmm, hmm), (1280, 800, 339, 212));
        assert!((hz - 60.0).abs() < 0.5, "{hz} Hz");

        assert_eq!(display.resize(&resize(1600, 1000, 423, 265)), Ok(true));
        assert_eq!((display.width, display.height), (1600, 1000));
        let edid = display.edid_bytes();
        let (w, h, hz, wmm, hmm) = preferred_mode(&edid);
        assert_eq!((w, h, wmm, hmm), (1600, 1000, 423, 265));
        assert!((hz - 60.0).abs() < 0.5, "{hz} Hz");
        // Screen size in cm.
        assert_eq!((edid[21], edid[22]), (42, 26));
    }

    #[test]
    fn large_modes_keep_the_refresh_rate() {
        let mut display = DisplayInfo::new(1280, 800);
        // Does not fit the pixel clock with the default blanking.
        display.resize(&resize(4095, 2304, 1083, 610)).unwrap();
        let (w, h, hz, ..) = preferred_mode(&display.edid_bytes());
        assert_eq!((w, h), (4095, 2304));
        assert!((hz - 60.0).abs() < 0.5, "{hz} Hz");
        // Does not fit even with reduced blanking: clamped clock, lower refresh.
        display.resize(&resize(4095, 4095, 4095, 4095)).unwrap();
        let (w, h, hz, wmm, hmm) = preferred_mode(&display.edid_bytes());
        assert_eq!((w, h, wmm, hmm), (4095, 4095, 4095, 4095));
        assert!(hz > 30.0 && hz < 60.0, "{hz} Hz");
    }

    #[test]
    fn resize_rejects_what_the_edid_cannot_describe() {
        let mut display = DisplayInfo::new(1280, 800);
        for bad in [
            resize(0, 800, 339, 212),
            resize(1280, 0, 339, 212),
            resize(4096, 800, 339, 212),
            resize(1280, 800, 4096, 212),
        ] {
            assert_eq!(display.resize(&bad), Err(DisplayResizeError::InvalidSize));
        }
        assert_eq!((display.width, display.height), (1280, 800));

        display.edid = DisplayInfoEdid::Provided(Box::new([0; 128]));
        assert_eq!(
            display.resize(&resize(1600, 1000, 423, 265)),
            Err(DisplayResizeError::FixedEdid)
        );
    }

    #[test]
    fn control_validates_and_queues() {
        let mut fixed = DisplayInfo::new(800, 600);
        fixed.edid = DisplayInfoEdid::Provided(Box::new([0; 128]));
        let control = DisplayControl::new(&[DisplayInfo::new(1280, 800), fixed]);
        let requests = control.requests();

        let ok = resize(1600, 1000, 423, 265);
        assert_eq!(control.request_resize(ok), Ok(()));
        let fixed = DisplayResize {
            display_id: 1,
            ..ok
        };
        assert_eq!(
            control.request_resize(fixed),
            Err(DisplayResizeError::FixedEdid)
        );
        let missing = DisplayResize {
            display_id: 2,
            ..ok
        };
        assert_eq!(
            control.request_resize(missing),
            Err(DisplayResizeError::InvalidDisplay)
        );
        let zero = resize(0, 1000, 423, 265);
        assert_eq!(
            control.request_resize(zero),
            Err(DisplayResizeError::InvalidSize)
        );

        assert_eq!(requests.try_iter().collect::<Vec<_>>(), [ok]);
    }

    #[test]
    fn display_event_survives_a_stale_clear() {
        let control = DisplayControl::new(&[DisplayInfo::new(1280, 800)]);
        assert_eq!(control.read_events(), 0);

        control.raise_display_event();
        assert_eq!(control.read_events(), VIRTIO_GPU_EVENT_DISPLAY);
        control.clear_events(VIRTIO_GPU_EVENT_DISPLAY);
        assert_eq!(control.read_events(), 0);

        // Raised again while the driver handles the previous event: its clear
        // must not drop the new one.
        control.raise_display_event();
        assert_eq!(control.read_events(), VIRTIO_GPU_EVENT_DISPLAY);
        control.raise_display_event();
        control.clear_events(VIRTIO_GPU_EVENT_DISPLAY);
        assert_eq!(control.read_events(), VIRTIO_GPU_EVENT_DISPLAY);
        control.clear_events(VIRTIO_GPU_EVENT_DISPLAY);
        assert_eq!(control.read_events(), 0);
    }

    #[test]
    fn ctrl_commands_count_up() {
        let control = DisplayControl::new(&[DisplayInfo::new(1280, 800)]);
        assert_eq!(control.ctrl_commands(), 0);
        control.note_ctrl_command();
        control.note_ctrl_command();
        assert_eq!(control.ctrl_commands(), 2);
    }
}
