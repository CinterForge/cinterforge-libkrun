// SPDX-License-Identifier: Apache-2.0
//
// CoreAudio backend for virtio-snd (macOS).
//
// Playback: a HAL AudioUnit bound to the selected or current default output device.
// Speaker layouts are detected at boot. The unit follows default-output and device-list
// notifications at once and rechecks the layout once a second. The guest's PCM stays in its
// own format in the lock-free SPSC ring; the callback converts it to interleaved Float32 in
// physical speaker order, with a precomputed matrix for downmixing missing speakers. Devices
// wider than the mixer get stereo and the HAL's own channel mapping. A playback message is
// completed once the device has
// played up to a host FIFO before its end (fifo_bytes: one device IO buffer plus safety offset,
// plus the SndControl buffer, at most the guest buffer minus two periods), with the callback's
// chunks spread over their play time, so the guest's hw_ptr advances smoothly at the device
// clock and never runs further ahead than the FIFO. latency_bytes reports the FIFO plus the
// CoreAudio path after the callback (IO buffer, safety offset, device + stream latency, unit
// latency), which the guest driver feeds into runtime->delay.
//
// Capture: a HAL output unit with input enabled on the default input device (re-selected when
// the default changes) renders Float32 at the device rate into a second ring; the pump thread
// converts it (AudioConverter: rate and format) into the guest's buffers and completes each
// one when it is full. The input unit runs only between START and STOP, so the microphone (and
// its permission prompt) is only touched while a guest application records.
//
// Underruns: after START, and again after the ring ran dry, the callback plays silence until
// the ring holds a FIFO's worth (or a FIFO's time has passed), then plays on; each gap followed
// by more data counts as one underrun and is logged by the pump thread. Device and device
// rate changes are picked up once a second (FIFO, latency and log line follow).
//
// STEAMAC_SND_DUMP=/path.wav writes guest PCM before speaker mapping for the playback stream
// while it runs (underrun silence included) to a WAV file (a new file per format change).
// STEAMAC_SND_TRACE=1 logs PCM commands and a per-second heartbeat of the stream state.

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::ffi::{c_char, c_void};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::super::{
    stream::{Buffer, Error as StreamError, PCMState, PcmParams},
    virtio_sound::*,
    Direction, Error, Result, SndControl, Stream,
};
use super::channel_layout::{self, Mixer, MAX_CHANNELS, MAX_DEVICE_CHANNELS};
use super::AudioBackend;

#[allow(non_upper_case_globals, non_snake_case, clippy::upper_case_acronyms)]
mod ffi {
    use std::ffi::{c_char, c_void};

    pub type OSStatus = i32;
    pub type AudioUnit = *mut c_void;
    pub type AudioComponent = *mut c_void;
    pub type AudioConverterRef = *mut c_void;
    pub type AudioObjectID = u32;

    #[repr(C)]
    pub struct AudioComponentDescription {
        pub componentType: u32,
        pub componentSubType: u32,
        pub componentManufacturer: u32,
        pub componentFlags: u32,
        pub componentFlagsMask: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct AudioStreamBasicDescription {
        pub mSampleRate: f64,
        pub mFormatID: u32,
        pub mFormatFlags: u32,
        pub mBytesPerPacket: u32,
        pub mFramesPerPacket: u32,
        pub mBytesPerFrame: u32,
        pub mChannelsPerFrame: u32,
        pub mBitsPerChannel: u32,
        pub mReserved: u32,
    }

    #[repr(C)]
    pub struct AudioBuffer {
        pub mNumberChannels: u32,
        pub mDataByteSize: u32,
        pub mData: *mut c_void,
    }

    /// AudioBufferList with room for the single interleaved buffer used everywhere here.
    #[repr(C)]
    pub struct AudioBufferList {
        pub mNumberBuffers: u32,
        pub mBuffers: [AudioBuffer; 1],
    }

    /// (refcon, ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, ioData)
    pub type AURenderCallback = extern "C" fn(
        *mut c_void,
        *mut u32,
        *const c_void,
        u32,
        u32,
        *mut AudioBufferList,
    ) -> OSStatus;

    #[repr(C)]
    pub struct AURenderCallbackStruct {
        pub inputProc: AURenderCallback,
        pub inputProcRefCon: *mut c_void,
    }

    /// (converter, ioNumberDataPackets, ioData, outDataPacketDescription, inUserData)
    pub type AudioConverterComplexInputDataProc = extern "C" fn(
        AudioConverterRef,
        *mut u32,
        *mut AudioBufferList,
        *mut *mut c_void,
        *mut c_void,
    ) -> OSStatus;

    #[repr(C)]
    pub struct AudioObjectPropertyAddress {
        pub mSelector: u32,
        pub mScope: u32,
        pub mElement: u32,
    }

    pub type AudioObjectPropertyListenerProc = extern "C" fn(
        AudioObjectID,
        u32,
        *const AudioObjectPropertyAddress,
        *mut c_void,
    ) -> OSStatus;

    #[repr(C)]
    pub struct MachTimespec {
        pub tv_sec: u32,
        pub tv_nsec: i32,
    }

    #[repr(C)]
    pub struct MachTimebaseInfo {
        pub numer: u32,
        pub denom: u32,
    }

    const fn fourcc(s: &[u8; 4]) -> u32 {
        ((s[0] as u32) << 24) | ((s[1] as u32) << 16) | ((s[2] as u32) << 8) | s[3] as u32
    }

    pub const kAudioUnitType_Output: u32 = fourcc(b"auou");
    pub const kAudioUnitSubType_HALOutput: u32 = fourcc(b"ahal");
    pub const kAudioUnitManufacturer_Apple: u32 = fourcc(b"appl");

    pub const kAudioUnitProperty_StreamFormat: u32 = 8;
    pub const kAudioUnitProperty_Latency: u32 = 12;
    pub const kAudioUnitProperty_SetRenderCallback: u32 = 23;
    pub const kAudioOutputUnitProperty_CurrentDevice: u32 = 2000;
    pub const kAudioOutputUnitProperty_ChannelMap: u32 = 2002;
    pub const kAudioOutputUnitProperty_EnableIO: u32 = 2003;
    pub const kAudioOutputUnitProperty_SetInputCallback: u32 = 2005;

    pub const kAudioUnitScope_Global: u32 = 0;
    pub const kAudioUnitScope_Input: u32 = 1;
    pub const kAudioUnitScope_Output: u32 = 2;

    pub const kAudioFormatLinearPCM: u32 = fourcc(b"lpcm");
    pub const kAudioFormatFlagIsFloat: u32 = 1 << 0;
    pub const kAudioFormatFlagIsSignedInteger: u32 = 1 << 2;
    pub const kAudioFormatFlagIsPacked: u32 = 1 << 3;
    pub const kAudioUnitRenderAction_OutputIsSilence: u32 = 1 << 4;

    pub const kAudioObjectSystemObject: AudioObjectID = 1;
    pub const kAudioHardwarePropertyDefaultOutputDevice: u32 = fourcc(b"dOut");
    pub const kAudioHardwarePropertyDevices: u32 = fourcc(b"dev#");
    pub const kAudioDevicePropertyStreamConfiguration: u32 = fourcc(b"slay");
    pub const kAudioDevicePropertyPreferredChannelLayout: u32 = fourcc(b"srnd");
    pub const kAudioDevicePropertyPreferredChannelsForStereo: u32 = fourcc(b"dch2");
    pub const kAudioFormatProperty_ChannelLayoutForTag: u32 = fourcc(b"cmpl");
    pub const kAudioFormatProperty_ChannelLayoutForBitmap: u32 = fourcc(b"cmpb");
    pub const kAudioHardwarePropertyDefaultInputDevice: u32 = fourcc(b"dIn ");
    pub const kAudioHardwarePropertyTranslateUIDToDevice: u32 = fourcc(b"uidd");
    pub const kHALOutputParam_Volume: u32 = 14;
    pub const kAudioObjectPropertyScopeGlobal: u32 = fourcc(b"glob");
    pub const kAudioObjectPropertyScopeInput: u32 = fourcc(b"inpt");
    pub const kAudioObjectPropertyScopeOutput: u32 = fourcc(b"outp");
    pub const kAudioObjectPropertyElementMain: u32 = 0;
    pub const kAudioObjectPropertyName: u32 = fourcc(b"lnam");
    pub const kAudioDevicePropertyLatency: u32 = fourcc(b"ltnc");
    pub const kAudioDevicePropertySafetyOffset: u32 = fourcc(b"saft");
    pub const kAudioDevicePropertyBufferFrameSize: u32 = fourcc(b"fsiz");
    pub const kAudioDevicePropertyNominalSampleRate: u32 = fourcc(b"nsrt");
    pub const kAudioDevicePropertyStreams: u32 = fourcc(b"stm#");
    pub const kAudioStreamPropertyLatency: u32 = fourcc(b"ltnc");

    pub const kCFStringEncodingUTF8: u32 = 0x0800_0100;
    pub const SYNC_POLICY_FIFO: i32 = 0;
    pub const KERN_SUCCESS: i32 = 0;

    #[link(name = "AudioToolbox", kind = "framework")]
    extern "C" {
        pub fn AudioComponentFindNext(
            inComponent: AudioComponent,
            inDesc: *const AudioComponentDescription,
        ) -> AudioComponent;
        pub fn AudioComponentInstanceNew(c: AudioComponent, out: *mut AudioUnit) -> OSStatus;
        pub fn AudioComponentInstanceDispose(unit: AudioUnit) -> OSStatus;
        pub fn AudioUnitInitialize(unit: AudioUnit) -> OSStatus;
        pub fn AudioUnitUninitialize(unit: AudioUnit) -> OSStatus;
        pub fn AudioUnitSetProperty(
            unit: AudioUnit,
            id: u32,
            scope: u32,
            element: u32,
            data: *const c_void,
            size: u32,
        ) -> OSStatus;
        pub fn AudioUnitGetProperty(
            unit: AudioUnit,
            id: u32,
            scope: u32,
            element: u32,
            data: *mut c_void,
            size: *mut u32,
        ) -> OSStatus;
        pub fn AudioUnitRender(
            unit: AudioUnit,
            ioActionFlags: *mut u32,
            inTimeStamp: *const c_void,
            inOutputBusNumber: u32,
            inNumberFrames: u32,
            ioData: *mut AudioBufferList,
        ) -> OSStatus;
        pub fn AudioOutputUnitStart(unit: AudioUnit) -> OSStatus;
        pub fn AudioOutputUnitStop(unit: AudioUnit) -> OSStatus;
        pub fn AudioUnitSetParameter(
            unit: AudioUnit,
            id: u32,
            scope: u32,
            element: u32,
            value: f32,
            offset_frames: u32,
        ) -> OSStatus;

        pub fn AudioFormatGetPropertyInfo(
            id: u32,
            spec_size: u32,
            spec: *const c_void,
            size: *mut u32,
        ) -> OSStatus;
        pub fn AudioFormatGetProperty(
            id: u32,
            spec_size: u32,
            spec: *const c_void,
            size: *mut u32,
            data: *mut c_void,
        ) -> OSStatus;
        pub fn AudioConverterNew(
            inSourceFormat: *const AudioStreamBasicDescription,
            inDestinationFormat: *const AudioStreamBasicDescription,
            outAudioConverter: *mut AudioConverterRef,
        ) -> OSStatus;
        pub fn AudioConverterDispose(conv: AudioConverterRef) -> OSStatus;
        pub fn AudioConverterFillComplexBuffer(
            conv: AudioConverterRef,
            inInputDataProc: AudioConverterComplexInputDataProc,
            inInputDataProcUserData: *mut c_void,
            ioOutputDataPacketSize: *mut u32,
            outOutputData: *mut AudioBufferList,
            outPacketDescription: *mut c_void,
        ) -> OSStatus;
    }

    #[link(name = "CoreAudio", kind = "framework")]
    extern "C" {
        pub fn AudioObjectGetPropertyData(
            id: AudioObjectID,
            address: *const AudioObjectPropertyAddress,
            qualifierSize: u32,
            qualifier: *const c_void,
            ioDataSize: *mut u32,
            outData: *mut c_void,
        ) -> OSStatus;
        pub fn AudioObjectGetPropertyDataSize(
            id: AudioObjectID,
            address: *const AudioObjectPropertyAddress,
            qualifierSize: u32,
            qualifier: *const c_void,
            outDataSize: *mut u32,
        ) -> OSStatus;
        pub fn AudioObjectAddPropertyListener(
            id: AudioObjectID,
            address: *const AudioObjectPropertyAddress,
            listener: AudioObjectPropertyListenerProc,
            clientData: *mut c_void,
        ) -> OSStatus;
        pub fn AudioObjectRemovePropertyListener(
            id: AudioObjectID,
            address: *const AudioObjectPropertyAddress,
            listener: AudioObjectPropertyListenerProc,
            clientData: *mut c_void,
        ) -> OSStatus;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        pub fn CFStringGetCString(s: *const c_void, buf: *mut c_char, size: isize, enc: u32) -> u8;
        pub fn CFRelease(cf: *const c_void);
        pub fn CFStringCreateWithCString(
            alloc: *const c_void,
            s: *const c_char,
            enc: u32,
        ) -> *const c_void;
    }

    extern "C" {
        pub static mach_task_self_: u32;
        pub fn semaphore_create(task: u32, sem: *mut u32, policy: i32, value: i32) -> i32;
        pub fn semaphore_destroy(task: u32, sem: u32) -> i32;
        pub fn semaphore_signal(sem: u32) -> i32;
        pub fn semaphore_timedwait(sem: u32, wait_time: MachTimespec) -> i32;
        pub fn mach_absolute_time() -> u64;
        pub fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
    }
}

use ffi::*;

/// Ring capacity per stream: ~2.7 s of 48 kHz FLOAT stereo, ~0.45 s at twelve channels. A
/// guest buffer that does not fit waits in the queue until the ring drains.
const RING_BYTES: usize = 1 << 20;
/// Capture: largest IO slice the input callback accepts (frames).
const CAPTURE_MAX_FRAMES: usize = 16384;
const HOUSEKEEPING: Duration = Duration::from_secs(1);
/// Non-zero status the converter input proc returns when the ring has no more frames now.
const NO_MORE_INPUT: OSStatus = 0x6e6f6d6f; // 'nomo'

/// STEAMAC_SND_TRACE=1: log PCM commands with the host stream state, plus a once-a-second
/// heartbeat of each running stream (render callbacks, ring and clock positions), on stderr.
static TRACE: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("STEAMAC_SND_TRACE").is_some_and(|v| !v.is_empty() && v != "0")
});
static TRACE_EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

macro_rules! trace {
    ($($arg:tt)*) => {
        if *TRACE {
            eprintln!(
                "[snd-trace {:9.3}] {}",
                TRACE_EPOCH.elapsed().as_secs_f64(),
                format_args!($($arg)*)
            );
        }
    };
}

fn check(status: OSStatus, what: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::UnexpectedAudioBackendError(format!(
            "{what} failed: OSStatus {status}"
        )))
    }
}

/// Mach semaphore: the one wake primitive that is safe to signal from a CoreAudio IO thread.
struct Semaphore(u32);

impl Semaphore {
    fn new() -> Result<Self> {
        let mut sem = 0;
        // SAFETY: plain mach call with a valid out pointer.
        let kr = unsafe { semaphore_create(mach_task_self_, &mut sem, SYNC_POLICY_FIFO, 0) };
        if kr != KERN_SUCCESS {
            return Err(Error::UnexpectedAudioBackendError(format!(
                "semaphore_create failed: {kr}"
            )));
        }
        Ok(Self(sem))
    }

    fn signal(&self) {
        // SAFETY: the semaphore lives as long as self.
        unsafe { semaphore_signal(self.0) };
    }

    fn wait(&self, timeout: Duration) {
        let ts = MachTimespec {
            tv_sec: timeout.as_secs() as u32,
            tv_nsec: timeout.subsec_nanos() as i32,
        };
        // SAFETY: the semaphore lives as long as self.
        unsafe { semaphore_timedwait(self.0, ts) };
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        // SAFETY: created by semaphore_create on this task.
        unsafe { semaphore_destroy(mach_task_self_, self.0) };
    }
}

/// Lock-free single-producer single-consumer byte ring. `head`/`tail` count bytes ever
/// written/read, so positions double as stream offsets.
struct Ring {
    buf: Box<[UnsafeCell<u8>]>,
    head: AtomicU64,
    tail: AtomicU64,
}

impl Ring {
    fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two());
        Self {
            buf: (0..capacity).map(|_| UnsafeCell::new(0)).collect(),
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
        }
    }

    fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        (self.head.load(Ordering::Acquire) - tail) as usize
    }

    fn written(&self) -> u64 {
        self.head.load(Ordering::Acquire)
    }

    fn consumed(&self) -> u64 {
        self.tail.load(Ordering::Acquire)
    }

    /// Drop everything buffered. Only while neither side is running.
    fn clear(&self) {
        self.tail
            .store(self.head.load(Ordering::Acquire), Ordering::Release);
    }

    fn data(&self) -> *mut u8 {
        UnsafeCell::raw_get(self.buf.as_ptr())
    }

    /// Producer side: offers up to `max` free bytes to `fill` in at most two contiguous
    /// pieces; `fill` returns how much it wrote (stopping early ends the call). Returns the
    /// total published.
    ///
    /// # Safety
    /// Only one thread may produce at a time.
    unsafe fn produce(&self, max: usize, mut fill: impl FnMut(&mut [u8]) -> usize) -> usize {
        let cap = self.buf.len();
        let head = self.head.load(Ordering::Relaxed);
        let free = cap - (head - self.tail.load(Ordering::Acquire)) as usize;
        let mut todo = max.min(free);
        let mut done = 0;
        while todo > 0 {
            let off = (head as usize + done) & (cap - 1);
            let n = todo.min(cap - off);
            let w = fill(std::slice::from_raw_parts_mut(self.data().add(off), n)).min(n);
            done += w;
            todo -= w;
            if w < n {
                break;
            }
        }
        self.head.store(head + done as u64, Ordering::Release);
        done
    }

    /// Consumer side, mirror of `produce`.
    ///
    /// # Safety
    /// Only one thread may consume at a time.
    unsafe fn consume(&self, max: usize, mut drain: impl FnMut(&[u8]) -> usize) -> usize {
        let cap = self.buf.len();
        let tail = self.tail.load(Ordering::Relaxed);
        let avail = (self.head.load(Ordering::Acquire) - tail) as usize;
        let mut todo = max.min(avail);
        let mut done = 0;
        while todo > 0 {
            let off = (tail as usize + done) & (cap - 1);
            let n = todo.min(cap - off);
            let r = drain(std::slice::from_raw_parts(self.data().add(off), n)).min(n);
            done += r;
            todo -= r;
            if r < n {
                break;
            }
        }
        self.tail.store(tail + done as u64, Ordering::Release);
        done
    }
}

/// Where the render callback last left the playback ring (seqlock; the IO thread is the only
/// writer). Lets the pump complete playback messages at the pace the device plays them instead
/// of in IO-buffer-sized bursts: a guest scheduling its writes by timer (PipeWire) then sees a
/// steadily moving hw_ptr.
#[derive(Default)]
struct PlayClock {
    seq: AtomicU64,
    /// mach_absolute_time() of the callback.
    time: AtomicU64,
    /// Ring read position after the callback.
    consumed: AtomicU64,
    /// Bytes the callback took from the ring.
    chunk: AtomicU64,
}

impl PlayClock {
    fn publish(&self, time: u64, consumed: u64, chunk: u64) {
        let seq = self.seq.load(Ordering::Relaxed);
        self.seq.store(seq + 1, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Release);
        self.time.store(time, Ordering::Relaxed);
        self.consumed.store(consumed, Ordering::Relaxed);
        self.chunk.store(chunk, Ordering::Relaxed);
        self.seq.store(seq + 2, Ordering::Release);
    }

    /// (time, consumed, chunk)
    fn read(&self) -> (u64, u64, u64) {
        loop {
            let seq = self.seq.load(Ordering::Acquire);
            let v = (
                self.time.load(Ordering::Relaxed),
                self.consumed.load(Ordering::Relaxed),
                self.chunk.load(Ordering::Relaxed),
            );
            std::sync::atomic::fence(Ordering::Acquire);
            if seq & 1 == 0 && self.seq.load(Ordering::Relaxed) == seq {
                return v;
            }
            std::hint::spin_loop();
        }
    }
}

fn ticks_to_nanos(ticks: u64) -> u64 {
    static TIMEBASE: LazyLock<(u128, u128)> = LazyLock::new(|| {
        let mut tb = MachTimebaseInfo { numer: 0, denom: 0 };
        // SAFETY: valid out pointer.
        unsafe { mach_timebase_info(&mut tb) };
        (tb.numer.max(1) as u128, tb.denom.max(1) as u128)
    });
    let (numer, denom) = *TIMEBASE;
    (ticks as u128 * numer / denom) as u64
}

/// State shared with a CoreAudio IO thread. Everything the callbacks touch is lock-free.
struct Shared {
    ring: Ring,
    /// Playback only: guest PCM consumed by the callback, before speaker mapping.
    dump: Option<Ring>,
    /// Between START and STOP.
    running: AtomicBool,
    /// Playback: ring position the callback may play up to (everything while running; up to
    /// what the guest has seen completed once it stops, so the host FIFO still plays out).
    limit: AtomicU64,
    /// Playback is under way since START or the last underrun. Until then the callback plays
    /// silence (and the clock stands still) while the ring holds less than `prime_bytes`, for
    /// at most `prime_nanos` after `prime_since`: playing a trickle as it arrives would turn
    /// one late guest into a run of underruns.
    primed: AtomicBool,
    prime_bytes: AtomicU64,
    prime_nanos: AtomicU64,
    /// mach_absolute_time() of START or of the last underrun.
    prime_since: AtomicU64,
    /// The ring ran dry while running.
    starved: AtomicBool,
    /// Underruns (playback) or overruns (capture).
    xruns: AtomicU64,
    /// Bytes per frame of the format the ring carries.
    frame_bytes: AtomicU32,
    channels: AtomicU32,
    silence: AtomicU8,
    /// Playback: position of the last render callback.
    clock: PlayClock,
    /// Render/input callbacks so far and mach_absolute_time() of the last one (heartbeat).
    callbacks: AtomicU64,
    last_callback: AtomicU64,
    /// Playback: silence bytes the callback filled in while running (priming, underruns).
    silence_bytes: AtomicU64,
    /// Capture: the unit to pull the input from.
    unit: AtomicPtr<c_void>,
    /// Capture: AudioUnitRender target, touched only by the IO thread.
    scratch: UnsafeCell<Box<[u8]>>,
    /// Changed only while the output unit is stopped/uninitialized.
    mixer: UnsafeCell<Mixer>,
    wake: Arc<Semaphore>,
}

// SAFETY: `scratch` is only accessed from the IO thread of the single unit using this
// `Shared`; the rings are SPSC with one producer and one consumer thread each; everything
// else is atomic. The mixer is changed only with the unit stopped/uninitialized.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

impl Shared {
    fn new(direction: Direction, dump: bool, wake: Arc<Semaphore>) -> Self {
        let scratch = match direction {
            Direction::Input => vec![0u8; CAPTURE_MAX_FRAMES * 2 * 4].into_boxed_slice(),
            Direction::Output => {
                vec![0u8; CAPTURE_MAX_FRAMES * MAX_CHANNELS * 4].into_boxed_slice()
            }
        };
        Self {
            ring: Ring::new(RING_BYTES),
            dump: dump.then(|| Ring::new(RING_BYTES)),
            running: AtomicBool::new(false),
            limit: AtomicU64::new(0),
            primed: AtomicBool::new(false),
            prime_bytes: AtomicU64::new(0),
            prime_nanos: AtomicU64::new(0),
            prime_since: AtomicU64::new(0),
            starved: AtomicBool::new(false),
            xruns: AtomicU64::new(0),
            frame_bytes: AtomicU32::new(4),
            channels: AtomicU32::new(2),
            silence: AtomicU8::new(0),
            clock: PlayClock::default(),
            callbacks: AtomicU64::new(0),
            last_callback: AtomicU64::new(0),
            silence_bytes: AtomicU64::new(0),
            unit: AtomicPtr::new(ptr::null_mut()),
            scratch: UnsafeCell::new(scratch),
            mixer: UnsafeCell::new(Mixer::default()),
            wake,
        }
    }
}

/// Playback render callback (CoreAudio IO thread): ring → device, silence when empty.
extern "C" fn render_output(
    refcon: *mut c_void,
    flags: *mut u32,
    _ts: *const c_void,
    _bus: u32,
    _frames: u32,
    io: *mut AudioBufferList,
) -> OSStatus {
    // SAFETY: refcon is the Arc<Shared> kept alive by the HostStream owning the unit, and the
    // unit is stopped and disposed before that Arc is released. `io` holds one interleaved
    // buffer of `mDataByteSize` bytes (the format set on the unit's input scope).
    let (sh, device_out) = unsafe {
        let buf = &mut (*io).mBuffers[0];
        if buf.mData.is_null() {
            return 0;
        }
        (
            &*(refcon as *const Shared),
            std::slice::from_raw_parts_mut(buf.mData as *mut u8, buf.mDataByteSize as usize),
        )
    };
    // SAFETY: plain clock read.
    let now = unsafe { mach_absolute_time() };
    sh.callbacks.fetch_add(1, Ordering::Relaxed);
    sh.last_callback.store(now, Ordering::Relaxed);
    let fb = sh.frame_bytes.load(Ordering::Relaxed) as usize;
    // SAFETY: the callback exclusively owns scratch; mixer is immutable while the unit runs.
    let (scratch, mixer) = unsafe { (&mut *sh.scratch.get(), &*sh.mixer.get()) };
    let frames = device_out.len() / (mixer.output_channels * 4);
    let want = frames.min(scratch.len() / fb) * fb;
    let out = &mut scratch[..want];
    device_out.fill(0);
    let running = sh.running.load(Ordering::Acquire);
    let playable = sh
        .limit
        .load(Ordering::Acquire)
        .saturating_sub(sh.ring.consumed()) as usize;
    let mut n = sh.ring.len().min(playable).min(want) / fb * fb;
    if running
        && !sh.primed.load(Ordering::Relaxed)
        && (sh.ring.len() as u64) < sh.prime_bytes.load(Ordering::Relaxed)
        && ticks_to_nanos(now.saturating_sub(sh.prime_since.load(Ordering::Relaxed)))
            < sh.prime_nanos.load(Ordering::Relaxed)
    {
        n = 0; // still priming
    }
    if n == 0 && !running {
        out.fill(sh.silence.load(Ordering::Relaxed));
        // SAFETY: flags points to the caller's action flags.
        unsafe { *flags |= kAudioUnitRenderAction_OutputIsSilence };
        return 0;
    }
    let mut off = 0;
    // SAFETY: the render callback is the ring's only consumer.
    unsafe {
        sh.ring.consume(n, |src| {
            out[off..off + src.len()].copy_from_slice(src);
            off += src.len();
            src.len()
        })
    };
    if off < want {
        out[off..].fill(sh.silence.load(Ordering::Relaxed));
    }
    mixer.render(out, device_out);
    if running {
        if off < want {
            sh.silence_bytes
                .fetch_add((want - off) as u64, Ordering::Relaxed);
        }
        if off > 0 {
            sh.primed.store(true, Ordering::Relaxed);
            if sh.starved.swap(false, Ordering::Relaxed) {
                sh.xruns.fetch_add(1, Ordering::Relaxed);
            }
        }
        if off < want && sh.primed.load(Ordering::Relaxed) {
            sh.starved.store(true, Ordering::Relaxed);
            sh.primed.store(false, Ordering::Relaxed);
            sh.prime_since.store(now, Ordering::Relaxed);
        }
    }
    if let Some(dump) = &sh.dump {
        let mut doff = 0;
        // SAFETY: the render callback is the dump ring's only producer.
        unsafe {
            dump.produce(want, |dst| {
                dst.copy_from_slice(&out[doff..doff + dst.len()]);
                doff += dst.len();
                dst.len()
            })
        };
    }
    if off > 0 {
        sh.clock.publish(now, sh.ring.consumed(), off as u64);
        sh.wake.signal();
    }
    0
}

/// Capture input callback (CoreAudio IO thread): device → ring (Float32, device rate).
extern "C" fn render_input(
    refcon: *mut c_void,
    flags: *mut u32,
    ts: *const c_void,
    bus: u32,
    frames: u32,
    _io: *mut AudioBufferList,
) -> OSStatus {
    // SAFETY: as in render_output; scratch is only used by this IO thread.
    let sh = unsafe { &*(refcon as *const Shared) };
    let scratch = unsafe { &mut *sh.scratch.get() };
    sh.callbacks.fetch_add(1, Ordering::Relaxed);
    // SAFETY: plain clock read.
    sh.last_callback
        .store(unsafe { mach_absolute_time() }, Ordering::Relaxed);
    let fb = sh.frame_bytes.load(Ordering::Relaxed) as usize;
    let frames = (frames as usize).min(scratch.len() / fb);
    let mut abl = AudioBufferList {
        mNumberBuffers: 1,
        mBuffers: [AudioBuffer {
            mNumberChannels: sh.channels.load(Ordering::Relaxed),
            mDataByteSize: (frames * fb) as u32,
            mData: scratch.as_mut_ptr() as *mut c_void,
        }],
    };
    // SAFETY: the unit is alive while its callbacks run; abl describes scratch.
    let status = unsafe {
        AudioUnitRender(
            sh.unit.load(Ordering::Acquire),
            flags,
            ts,
            bus,
            frames as u32,
            &mut abl,
        )
    };
    if status != 0 || !sh.running.load(Ordering::Acquire) {
        return status;
    }
    let got = (abl.mBuffers[0].mDataByteSize as usize).min(scratch.len());
    let mut off = 0;
    // SAFETY: the input callback is the ring's only producer.
    let n = unsafe {
        sh.ring.produce(got, |dst| {
            dst.copy_from_slice(&scratch[off..off + dst.len()]);
            off += dst.len();
            dst.len()
        })
    };
    if n < got {
        sh.xruns.fetch_add(1, Ordering::Relaxed);
    }
    sh.wake.signal();
    0
}

/// Guest stream format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Format {
    format: u8,
    rate: u32,
    channels: u32,
}

impl Format {
    fn from_params(p: &PcmParams) -> Option<Self> {
        let rate = match p.rate {
            VIRTIO_SND_PCM_RATE_5512 => 5512,
            VIRTIO_SND_PCM_RATE_8000 => 8000,
            VIRTIO_SND_PCM_RATE_11025 => 11025,
            VIRTIO_SND_PCM_RATE_16000 => 16000,
            VIRTIO_SND_PCM_RATE_22050 => 22050,
            VIRTIO_SND_PCM_RATE_32000 => 32000,
            VIRTIO_SND_PCM_RATE_44100 => 44100,
            VIRTIO_SND_PCM_RATE_48000 => 48000,
            VIRTIO_SND_PCM_RATE_64000 => 64000,
            VIRTIO_SND_PCM_RATE_88200 => 88200,
            VIRTIO_SND_PCM_RATE_96000 => 96000,
            VIRTIO_SND_PCM_RATE_176400 => 176400,
            VIRTIO_SND_PCM_RATE_192000 => 192000,
            _ => return None,
        };
        match p.format {
            VIRTIO_SND_PCM_FMT_U8
            | VIRTIO_SND_PCM_FMT_S16
            | VIRTIO_SND_PCM_FMT_S32
            | VIRTIO_SND_PCM_FMT_FLOAT => {}
            _ => return None,
        }
        (1..=MAX_CHANNELS as u8)
            .contains(&p.channels)
            .then_some(Self {
                format: p.format,
                rate,
                channels: p.channels as u32,
            })
    }

    fn sample_bytes(&self) -> u32 {
        match self.format {
            VIRTIO_SND_PCM_FMT_U8 => 1,
            VIRTIO_SND_PCM_FMT_S16 => 2,
            _ => 4,
        }
    }

    fn frame_bytes(&self) -> u32 {
        self.sample_bytes() * self.channels
    }

    fn silence(&self) -> u8 {
        if self.format == VIRTIO_SND_PCM_FMT_U8 {
            0x80
        } else {
            0
        }
    }

    fn asbd(&self) -> AudioStreamBasicDescription {
        let flags = match self.format {
            VIRTIO_SND_PCM_FMT_FLOAT => kAudioFormatFlagIsFloat,
            VIRTIO_SND_PCM_FMT_U8 => 0,
            _ => kAudioFormatFlagIsSignedInteger,
        } | kAudioFormatFlagIsPacked;
        AudioStreamBasicDescription {
            mSampleRate: self.rate as f64,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: flags,
            mBytesPerPacket: self.frame_bytes(),
            mFramesPerPacket: 1,
            mBytesPerFrame: self.frame_bytes(),
            mChannelsPerFrame: self.channels,
            mBitsPerChannel: self.sample_bytes() * 8,
            mReserved: 0,
        }
    }
}

/// Float32 interleaved at `rate`: what the capture unit delivers.
fn float_asbd(rate: f64, channels: u32) -> AudioStreamBasicDescription {
    AudioStreamBasicDescription {
        mSampleRate: rate,
        mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
        mBytesPerPacket: 4 * channels,
        mFramesPerPacket: 1,
        mBytesPerFrame: 4 * channels,
        mChannelsPerFrame: channels,
        mBitsPerChannel: 32,
        mReserved: 0,
    }
}

fn property_address(selector: u32, scope: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain,
    }
}

fn object_property<T: Copy + Default>(
    object: AudioObjectID,
    selector: u32,
    scope: u32,
) -> Option<T> {
    let addr = property_address(selector, scope);
    let mut value = T::default();
    let mut size = std::mem::size_of::<T>() as u32;
    // SAFETY: value is a T-sized buffer and size says so.
    let status = unsafe {
        AudioObjectGetPropertyData(
            object,
            &addr,
            0,
            ptr::null(),
            &mut size,
            &mut value as *mut T as *mut c_void,
        )
    };
    (status == 0).then_some(value)
}

/// Aligned backing storage also handles AudioBufferList's pointer-sized fields.
fn property_bytes(device: AudioObjectID, selector: u32) -> Option<Vec<u8>> {
    let addr = property_address(selector, kAudioObjectPropertyScopeOutput);
    let mut size = 0;
    // SAFETY: valid address and size pointer; no qualifier.
    if unsafe { AudioObjectGetPropertyDataSize(device, &addr, 0, ptr::null(), &mut size) } != 0
        || size == 0
        || size > 65536
    {
        return None;
    }
    let mut storage = vec![0u64; (size as usize).div_ceil(8)];
    let capacity = storage.len() * 8;
    // SAFETY: u64 allocation is suitably aligned for all CoreAudio property structures.
    if unsafe {
        AudioObjectGetPropertyData(
            device,
            &addr,
            0,
            ptr::null(),
            &mut size,
            storage.as_mut_ptr().cast(),
        )
    } != 0
        || size as usize > capacity
    {
        return None;
    }
    // SAFETY: size bytes were initialized in the zero-initialized allocation.
    Some(unsafe { std::slice::from_raw_parts(storage.as_ptr().cast(), size as usize) }.to_vec())
}

fn device_channels(device: AudioObjectID) -> Option<usize> {
    let bytes = property_bytes(device, kAudioDevicePropertyStreamConfiguration)?;
    let count = u32::from_ne_bytes(bytes.get(..4)?.try_into().ok()?) as usize;
    let offset = std::mem::offset_of!(AudioBufferList, mBuffers);
    let stride = std::mem::size_of::<AudioBuffer>();
    if bytes.len() < offset + count * stride {
        return None;
    }
    let channels: usize = (0..count)
        .map(|i| {
            u32::from_ne_bytes(
                bytes[offset + i * stride..offset + i * stride + 4]
                    .try_into()
                    .unwrap(),
            ) as usize
        })
        .sum();
    (channels > 0).then_some(channels)
}

fn expanded_layout(mut bytes: Vec<u8>) -> Option<Vec<u8>> {
    let tag = u32::from_ne_bytes(bytes.get(..4)?.try_into().ok()?);
    if tag == 0 {
        return Some(bytes);
    }
    let (property, spec) = if tag == 1 << 16 {
        (
            kAudioFormatProperty_ChannelLayoutForBitmap,
            u32::from_ne_bytes(bytes.get(4..8)?.try_into().ok()?),
        )
    } else {
        (kAudioFormatProperty_ChannelLayoutForTag, tag)
    };
    let mut size = 0;
    // SAFETY: scalar specifier and writable size pointer.
    if unsafe { AudioFormatGetPropertyInfo(property, 4, (&spec as *const u32).cast(), &mut size) }
        != 0
        || size < 12
        || size > 65536
    {
        return None;
    }
    let mut storage = vec![0u32; (size as usize).div_ceil(4)];
    let capacity = storage.len() * 4;
    // SAFETY: aligned writable buffer of at least size bytes.
    if unsafe {
        AudioFormatGetProperty(
            property,
            4,
            (&spec as *const u32).cast(),
            &mut size,
            storage.as_mut_ptr().cast(),
        )
    } != 0
        || size as usize > capacity
    {
        return None;
    }
    bytes = unsafe { std::slice::from_raw_parts(storage.as_ptr().cast(), size as usize) }.to_vec();
    Some(bytes)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OutputRoute {
    device: AudioObjectID,
    /// Physical device order; NONE leaves an unclassified output silent. Shorter than
    /// `device_channels` only for a device wider than the mixer (stereo, HAL mapping).
    positions: Vec<u8>,
    device_channels: usize,
}

fn output_route(control: &SndControl) -> Option<OutputRoute> {
    let device = control
        .output_device()
        .as_deref()
        .and_then(resolve_output_device)
        .or_else(|| {
            object_property(
                kAudioObjectSystemObject,
                kAudioHardwarePropertyDefaultOutputDevice,
                kAudioObjectPropertyScopeGlobal,
            )
        })
        .filter(|id| *id != 0)?;
    let channels = device_channels(device)?;
    if channels > MAX_DEVICE_CHANNELS {
        // Wider than the mixer (BlackHole 64ch, aggregates, large interfaces): stereo into the
        // HAL's default channel mapping, as before speaker routing.
        return Some(OutputRoute {
            device,
            positions: vec![VIRTIO_SND_CHMAP_FL, VIRTIO_SND_CHMAP_FR],
            device_channels: channels,
        });
    }
    let positions = property_bytes(device, kAudioDevicePropertyPreferredChannelLayout)
        .and_then(expanded_layout)
        .and_then(|bytes| channel_layout::descriptions(&bytes, channels))
        .filter(|p| {
            channels == 1 || (p.contains(&VIRTIO_SND_CHMAP_FL) && p.contains(&VIRTIO_SND_CHMAP_FR))
        })
        .unwrap_or_else(|| {
            let mut positions = vec![VIRTIO_SND_CHMAP_NONE; channels];
            if channels == 1 {
                positions[0] = VIRTIO_SND_CHMAP_MONO;
            } else {
                let pair: [u32; 2] = object_property(
                    device,
                    kAudioDevicePropertyPreferredChannelsForStereo,
                    kAudioObjectPropertyScopeOutput,
                )
                .unwrap_or([1, 2]);
                let valid =
                    pair[0] != pair[1] && pair.iter().all(|c| *c > 0 && *c as usize <= channels);
                let pair = if valid { pair } else { [1, 2] };
                positions[pair[0] as usize - 1] = VIRTIO_SND_CHMAP_FL;
                positions[pair[1] as usize - 1] = VIRTIO_SND_CHMAP_FR;
            }
            positions
        });
    Some(OutputRoute {
        device,
        positions,
        device_channels: channels,
    })
}

pub fn playback_maps(control: &SndControl) -> Vec<Vec<u8>> {
    let route = output_route(control);
    let maps = channel_layout::guest_maps(
        route
            .as_ref()
            .map_or(&[VIRTIO_SND_CHMAP_FL, VIRTIO_SND_CHMAP_FR][..], |r| {
                r.positions.as_slice()
            }),
    );
    log::info!(
        "snd: boot playback supports {} channels; output {:?}; guest maps {:?}",
        maps.len(),
        route,
        maps
    );
    maps
}

fn device_name(device: AudioObjectID) -> String {
    let addr = property_address(kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal);
    let mut cfstr: *const c_void = ptr::null();
    let mut size = std::mem::size_of::<*const c_void>() as u32;
    // SAFETY: the property is a CFStringRef we own (released below).
    let status = unsafe {
        AudioObjectGetPropertyData(
            device,
            &addr,
            0,
            ptr::null(),
            &mut size,
            &mut cfstr as *mut _ as *mut c_void,
        )
    };
    if status != 0 || cfstr.is_null() {
        return format!("device {device}");
    }
    let mut buf = [0 as c_char; 256];
    // SAFETY: buf is 256 bytes; cfstr is a valid CFString released exactly once.
    let ok = unsafe {
        let ok = CFStringGetCString(
            cfstr,
            buf.as_mut_ptr(),
            buf.len() as isize,
            kCFStringEncodingUTF8,
        );
        CFRelease(cfstr);
        ok
    };
    if ok == 0 {
        return format!("device {device}");
    }
    // SAFETY: CFStringGetCString NUL-terminated buf.
    unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// The output device with this CoreAudio UID, if it is present and has output streams.
fn resolve_output_device(uid: &str) -> Option<AudioObjectID> {
    let uid = std::ffi::CString::new(uid).ok()?;
    // SAFETY: CFString created from a NUL-terminated UTF-8 string, released below; the
    // property takes a CFStringRef qualifier and returns an AudioObjectID.
    let device = unsafe {
        let cfuid = CFStringCreateWithCString(ptr::null(), uid.as_ptr(), kCFStringEncodingUTF8);
        if cfuid.is_null() {
            return None;
        }
        let addr = property_address(
            kAudioHardwarePropertyTranslateUIDToDevice,
            kAudioObjectPropertyScopeGlobal,
        );
        let mut device: AudioObjectID = 0;
        let mut size = std::mem::size_of::<AudioObjectID>() as u32;
        let status = AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &addr,
            std::mem::size_of::<*const c_void>() as u32,
            &cfuid as *const _ as *const c_void,
            &mut size,
            &mut device as *mut _ as *mut c_void,
        );
        CFRelease(cfuid);
        if status != 0 {
            return None;
        }
        device
    };
    if device == 0 {
        return None;
    }
    let addr = property_address(kAudioDevicePropertyStreams, kAudioObjectPropertyScopeOutput);
    let mut size = 0u32;
    // SAFETY: size query of a valid property address.
    let status =
        unsafe { AudioObjectGetPropertyDataSize(device, &addr, 0, ptr::null(), &mut size) };
    (status == 0 && size > 0).then_some(device)
}

/// Whether a CoreAudio output device with this UID is present (krun_snd_set_output_device).
pub fn output_device_exists(uid: &str) -> bool {
    resolve_output_device(uid).is_some()
}

/// Timing of a device in one scope, in device frames.
struct DeviceTiming {
    /// From the end of the IO buffer to the speaker/microphone: device + stream latency.
    latency: u32,
    /// What the IO thread takes per cycle plus how far ahead of the hardware it works: the
    /// IO buffer and the safety offset.
    io: u32,
    rate: f64,
}

fn device_timing(device: AudioObjectID, scope: u32) -> DeviceTiming {
    let lat: u32 = object_property(device, kAudioDevicePropertyLatency, scope).unwrap_or(0);
    let safety: u32 = object_property(device, kAudioDevicePropertySafetyOffset, scope).unwrap_or(0);
    let io: u32 = object_property(device, kAudioDevicePropertyBufferFrameSize, scope).unwrap_or(0);
    let rate: f64 = object_property(
        device,
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeGlobal,
    )
    .unwrap_or(48000.0);

    // The stream term is separate from the device term and usually the larger one.
    let mut stream_lat = 0;
    let addr = property_address(kAudioDevicePropertyStreams, scope);
    let mut size = 0u32;
    // SAFETY: size query, then a read into a buffer of that size.
    unsafe {
        if AudioObjectGetPropertyDataSize(device, &addr, 0, ptr::null(), &mut size) == 0
            && size >= 4
        {
            let mut ids = vec![0u32; size as usize / 4];
            if AudioObjectGetPropertyData(
                device,
                &addr,
                0,
                ptr::null(),
                &mut size,
                ids.as_mut_ptr() as *mut c_void,
            ) == 0
            {
                stream_lat = object_property(
                    ids[0],
                    kAudioStreamPropertyLatency,
                    kAudioObjectPropertyScopeGlobal,
                )
                .unwrap_or(0);
            }
        }
    }
    DeviceTiming {
        latency: lat + stream_lat,
        io: io + safety,
        rate,
    }
}

/// An AudioUnit instance; stopped, uninitialized and disposed on drop.
struct Unit {
    au: AudioUnit,
    initialized: bool,
    started: bool,
}

// SAFETY: AudioUnit API calls are thread-safe for the property/start/stop calls made here; the
// unit is only driven from behind the backend's mutex.
unsafe impl Send for Unit {}

impl Unit {
    fn new(subtype: u32) -> Result<Self> {
        let desc = AudioComponentDescription {
            componentType: kAudioUnitType_Output,
            componentSubType: subtype,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0,
        };
        // SAFETY: plain component lookup/instantiation with valid pointers.
        unsafe {
            let comp = AudioComponentFindNext(ptr::null_mut(), &desc);
            if comp.is_null() {
                return Err(Error::UnexpectedAudioBackendError(
                    "no output audio component".into(),
                ));
            }
            let mut au = ptr::null_mut();
            check(
                AudioComponentInstanceNew(comp, &mut au),
                "AudioComponentInstanceNew",
            )?;
            Ok(Self {
                au,
                initialized: false,
                started: false,
            })
        }
    }

    fn set<T>(&self, id: u32, scope: u32, element: u32, value: &T, what: &str) -> Result<()> {
        // SAFETY: value is a valid T for the duration of the call.
        check(
            unsafe {
                AudioUnitSetProperty(
                    self.au,
                    id,
                    scope,
                    element,
                    value as *const T as *const c_void,
                    std::mem::size_of::<T>() as u32,
                )
            },
            what,
        )
    }

    fn get<T: Copy + Default>(&self, id: u32, scope: u32, element: u32) -> Option<T> {
        let mut value = T::default();
        let mut size = std::mem::size_of::<T>() as u32;
        // SAFETY: value is a T-sized buffer.
        let status = unsafe {
            AudioUnitGetProperty(
                self.au,
                id,
                scope,
                element,
                &mut value as *mut T as *mut c_void,
                &mut size,
            )
        };
        (status == 0).then_some(value)
    }

    /// Output gain of a HAL output unit (also the default-output unit).
    fn set_volume(&self, gain: f32) {
        // SAFETY: valid unit; a plain parameter set.
        let status = unsafe {
            AudioUnitSetParameter(
                self.au,
                kHALOutputParam_Volume,
                kAudioUnitScope_Global,
                0,
                gain,
                0,
            )
        };
        if status != 0 {
            log::warn!("snd: setting the output volume failed: OSStatus {status}");
        }
    }

    fn current_device(&self) -> AudioObjectID {
        self.get(
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
        )
        .unwrap_or(0)
    }

    fn initialize(&mut self) -> Result<()> {
        if !self.initialized {
            // SAFETY: valid unit.
            check(
                unsafe { AudioUnitInitialize(self.au) },
                "AudioUnitInitialize",
            )?;
            self.initialized = true;
        }
        Ok(())
    }

    fn uninitialize(&mut self) {
        self.stop();
        if self.initialized {
            // SAFETY: valid unit.
            unsafe { AudioUnitUninitialize(self.au) };
            self.initialized = false;
        }
    }

    fn start(&mut self) -> Result<()> {
        if !self.started {
            // SAFETY: valid, initialized unit.
            check(
                unsafe { AudioOutputUnitStart(self.au) },
                "AudioOutputUnitStart",
            )?;
            self.started = true;
        }
        Ok(())
    }

    /// Returns once the IO thread no longer runs the unit's callbacks.
    fn stop(&mut self) {
        if self.started {
            // SAFETY: valid unit.
            unsafe { AudioOutputUnitStop(self.au) };
            self.started = false;
        }
    }
}

impl Drop for Unit {
    fn drop(&mut self) {
        self.uninitialize();
        // SAFETY: valid unit, disposed once.
        unsafe { AudioComponentInstanceDispose(self.au) };
    }
}

/// Capture rate/format converter (device Float32 → guest format and rate).
struct Converter {
    conv: AudioConverterRef,
    in_frame_bytes: usize,
    /// Frames handed to the converter by the last input proc call; must stay valid until the
    /// next call.
    staging: Vec<u8>,
}

// SAFETY: only used from behind the backend's mutex.
unsafe impl Send for Converter {}

impl Drop for Converter {
    fn drop(&mut self) {
        // SAFETY: created by AudioConverterNew, disposed once.
        unsafe { AudioConverterDispose(self.conv) };
    }
}

struct ConverterInput<'a> {
    ring: &'a Ring,
    staging: &'a mut Vec<u8>,
    frame_bytes: usize,
    channels: u32,
}

extern "C" fn converter_input(
    _conv: AudioConverterRef,
    packets: *mut u32,
    data: *mut AudioBufferList,
    _desc: *mut *mut c_void,
    user: *mut c_void,
) -> OSStatus {
    // SAFETY: user is the ConverterInput passed to AudioConverterFillComplexBuffer, alive for
    // the duration of that call; packets/data are the converter's out parameters.
    let (input, packets, data) = unsafe {
        (
            &mut *(user as *mut ConverterInput),
            &mut *packets,
            &mut *data,
        )
    };
    let fb = input.frame_bytes;
    let want = (*packets as usize) * fb;
    let n = input.ring.len().min(want) / fb * fb;
    input.staging.clear();
    // SAFETY: the pump thread is the capture ring's only consumer.
    unsafe {
        input.ring.consume(n, |src| {
            input.staging.extend_from_slice(src);
            src.len()
        })
    };
    *packets = (n / fb) as u32;
    data.mNumberBuffers = 1;
    data.mBuffers[0] = AudioBuffer {
        mNumberChannels: input.channels,
        mDataByteSize: n as u32,
        mData: input.staging.as_mut_ptr() as *mut c_void,
    };
    if n == 0 {
        NO_MORE_INPUT
    } else {
        0
    }
}

/// WAV file receiving the playback dump.
struct Dump {
    file: File,
    data_len: u32,
}

impl Dump {
    fn create(path: &Path, fmt: &Format) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let tag: u16 = if fmt.format == VIRTIO_SND_PCM_FMT_FLOAT {
            3
        } else {
            1
        };
        let mut h = Vec::with_capacity(44);
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&36u32.to_le_bytes());
        h.extend_from_slice(b"WAVEfmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&tag.to_le_bytes());
        h.extend_from_slice(&(fmt.channels as u16).to_le_bytes());
        h.extend_from_slice(&fmt.rate.to_le_bytes());
        h.extend_from_slice(&(fmt.rate * fmt.frame_bytes()).to_le_bytes());
        h.extend_from_slice(&(fmt.frame_bytes() as u16).to_le_bytes());
        h.extend_from_slice(&(fmt.sample_bytes() as u16 * 8).to_le_bytes());
        h.extend_from_slice(b"data");
        h.extend_from_slice(&0u32.to_le_bytes());
        file.write_all_at(&h, 0)?;
        Ok(Self { file, data_len: 0 })
    }

    fn append(&mut self, data: &[u8]) -> std::io::Result<()> {
        self.file.write_all_at(data, 44 + self.data_len as u64)?;
        self.data_len = self.data_len.saturating_add(data.len() as u32);
        self.file
            .write_all_at(&(36 + self.data_len).to_le_bytes(), 4)?;
        self.file.write_all_at(&self.data_len.to_le_bytes(), 40)
    }
}

/// Host side of one virtio-snd PCM stream.
struct HostStream {
    id: usize,
    direction: Direction,
    shared: Arc<Shared>,
    unit: Option<Unit>,
    /// Guest format of the last PREPARE.
    guest_format: Option<Format>,
    /// Playback: guest buffer and period bytes of the last PREPARE.
    guest_buffer: Option<(u64, u64)>,
    /// Guest format the unit is configured for.
    format: Option<Format>,
    /// Device the unit was last seen on, and its nominal rate then.
    device: AudioObjectID,
    device_rate: f64,
    converter: Option<Converter>,
    /// Playback messages fully copied into the ring, completed once the device has played up
    /// to `fifo_bytes` before the stream offset they end at.
    inflight: VecDeque<(u64, Buffer)>,
    /// Playback: ring position up to which messages have been completed (the guest's hw_ptr).
    completed: u64,
    /// Playback: host FIFO the guest's hw_ptr runs ahead of the device by (one device IO
    /// buffer plus the SndControl buffer).
    fifo_bytes: u64,
    latency_bytes: u32,
    reported_xruns: u64,
    dump_path: Option<PathBuf>,
    dump: Option<(Format, Dump)>,
    dump_files: u32,
    control: Arc<SndControl>,
    /// Control generation last applied.
    seen_generation: u64,
    /// Playback: device the unit was created for (None = the default-output unit).
    output_route: Option<OutputRoute>,
    output_restart_pending: bool,
    /// The unit is stopped because the VM is paused (SndControl::paused) and starts again on
    /// resume.
    suspended: bool,
    /// STEAMAC_SND_TRACE: callback count at the last heartbeat.
    traced_callbacks: u64,
}

impl HostStream {
    fn new(
        id: usize,
        direction: Direction,
        wake: Arc<Semaphore>,
        control: Arc<SndControl>,
    ) -> Self {
        let dump_path = (direction == Direction::Output)
            .then(|| std::env::var_os("STEAMAC_SND_DUMP").map(PathBuf::from))
            .flatten();
        Self {
            id,
            direction,
            shared: Arc::new(Shared::new(direction, dump_path.is_some(), wake)),
            unit: None,
            guest_format: None,
            guest_buffer: None,
            format: None,
            device: 0,
            device_rate: 0.0,
            converter: None,
            inflight: VecDeque::new(),
            completed: 0,
            fifo_bytes: 0,
            latency_bytes: 0,
            reported_xruns: 0,
            dump_path,
            dump: None,
            dump_files: 0,
            control,
            seen_generation: u64::MAX,
            output_route: None,
            output_restart_pending: false,
            suspended: false,
            traced_callbacks: 0,
        }
    }

    /// STEAMAC_SND_TRACE line with the host side of the stream (and the guest's, if given).
    fn trace_state(&self, what: &str, guest: Option<&Stream>) {
        if !*TRACE {
            return;
        }
        let sh = &self.shared;
        let (time, consumed, chunk) = sh.clock.read();
        // SAFETY: plain clock read.
        let now = unsafe { mach_absolute_time() };
        let ago = |t: u64| match t {
            0 => f64::NAN,
            t => ticks_to_nanos(now.saturating_sub(t)) as f64 / 1e6,
        };
        let limit = sh.limit.load(Ordering::Acquire);
        trace!(
            "stream {} {what}: guest {} | unit {} running {} written {} read {} limit {} \
             completed {} inflight {} (first end {}) | clock {} chunk {} {:.1} ms ago, \
             play {} | callbacks {} last {:.1} ms ago, silence {} | fifo {} latency {} \
             xruns {}",
            self.id,
            guest.map_or("-".into(), |g| format!(
                "{:?} queued {}",
                g.state,
                g.buffers.len()
            )),
            match &self.unit {
                None => "none",
                Some(u) if u.started => "started",
                Some(u) if u.initialized => "stopped",
                Some(_) => "uninitialized",
            },
            sh.running.load(Ordering::Acquire),
            sh.ring.written(),
            sh.ring.consumed(),
            if limit == u64::MAX {
                "max".to_string()
            } else {
                limit.to_string()
            },
            self.completed,
            self.inflight.len(),
            self.inflight.front().map_or(0, |(end, _)| *end),
            consumed,
            chunk,
            ago(time),
            self.play_position(),
            sh.callbacks.load(Ordering::Relaxed),
            ago(sh.last_callback.load(Ordering::Relaxed)),
            sh.silence_bytes.load(Ordering::Relaxed),
            self.fifo_bytes,
            self.latency_bytes,
            sh.xruns.load(Ordering::Relaxed),
        );
    }

    /// Apply changed runtime settings (krun_snd_set_*); with `recheck`, also notice that a
    /// selected output device appeared or went away.
    fn apply_control(&mut self, recheck: bool) {
        self.follow_pause();
        let generation = self.control.generation();
        let changed = generation != self.seen_generation;
        if self.direction != Direction::Output || !(changed || recheck) {
            return;
        }
        self.seen_generation = generation;
        let Some(fmt) = self.format else {
            return; // applied by configure()
        };
        let route = output_route(&self.control);
        if route != self.output_route || self.unit.as_ref().is_none_or(|u| !u.initialized) {
            // Dispose first: no callback may observe the mixer while it is being replaced.
            self.output_restart_pending |= self.unit.as_ref().is_some_and(|u| u.started);
            self.unit = None;
            self.output_route = None;
            if route.is_some() {
                let result = self.configure(fmt).and_then(|()| {
                    if self.output_restart_pending && !self.control.paused() {
                        self.start_unit()
                    } else {
                        Ok(())
                    }
                });
                if let Err(err) = result {
                    // Retry at the next housekeeping pass; retain queued guest PCM and clock.
                    self.unit = None;
                    log::error!("snd: stream {}: switching output device: {err}", self.id);
                } else {
                    self.output_restart_pending = false;
                    self.trace_state("output switched", None);
                }
            }
        }
        if let Some(unit) = self.unit.as_ref() {
            unit.set_volume(self.control.volume());
        }
        if changed {
            self.update_latency(); // buffer_ms
        }
    }

    /// Stop the unit while the VM is paused and start it again on resume. The render
    /// callback stops consuming the ring, so the guest's hw_ptr stops (after at most the host
    /// FIFO) and playback carries on from the same sample without an underrun.
    fn follow_pause(&mut self) {
        if self.control.paused() {
            self.suspended |= self.output_restart_pending;
            if self.unit.as_ref().is_some_and(|u| u.started) {
                self.stop_unit();
                self.suspended = true;
                self.trace_state("suspended (VM paused)", None);
            }
        } else if self.suspended && self.unit.as_ref().is_some_and(|u| u.initialized) {
            self.suspended = false;
            if let Err(err) = self.start_unit() {
                log::error!("snd: stream {}: restarting after VM pause: {err}", self.id);
            }
            self.trace_state("resumed (VM resumed)", None);
        }
    }

    fn refcon(&self) -> *mut c_void {
        Arc::as_ptr(&self.shared) as *mut c_void
    }

    fn create_unit(&mut self) -> Result<()> {
        if self.unit.is_some() {
            return Ok(());
        }
        let unit = match self.direction {
            Direction::Output => {
                let route = output_route(&self.control).ok_or_else(|| {
                    Error::UnexpectedAudioBackendError("no usable output device".into())
                })?;
                let unit = Unit::new(kAudioUnitSubType_HALOutput)?;
                unit.set(
                    kAudioOutputUnitProperty_CurrentDevice,
                    kAudioUnitScope_Global,
                    0,
                    &route.device,
                    "CurrentDevice(output)",
                )?;
                self.output_route = Some(route);
                let cb = AURenderCallbackStruct {
                    inputProc: render_output,
                    inputProcRefCon: self.refcon(),
                };
                unit.set(
                    kAudioUnitProperty_SetRenderCallback,
                    kAudioUnitScope_Input,
                    0,
                    &cb,
                    "SetRenderCallback",
                )?;
                unit
            }
            Direction::Input => {
                let unit = Unit::new(kAudioUnitSubType_HALOutput)?;
                unit.set(
                    kAudioOutputUnitProperty_EnableIO,
                    kAudioUnitScope_Input,
                    1,
                    &1u32,
                    "EnableIO(input)",
                )?;
                unit.set(
                    kAudioOutputUnitProperty_EnableIO,
                    kAudioUnitScope_Output,
                    0,
                    &0u32,
                    "EnableIO(output)",
                )?;
                let cb = AURenderCallbackStruct {
                    inputProc: render_input,
                    inputProcRefCon: self.refcon(),
                };
                unit.set(
                    kAudioOutputUnitProperty_SetInputCallback,
                    kAudioUnitScope_Global,
                    0,
                    &cb,
                    "SetInputCallback",
                )?;
                self.shared.unit.store(unit.au, Ordering::Release);
                unit
            }
        };
        self.unit = Some(unit);
        Ok(())
    }

    /// (Re)configure the unit for `fmt`. The unit is left initialized and stopped.
    fn configure(&mut self, fmt: Format) -> Result<()> {
        self.create_unit()?;
        let unit = self.unit.as_mut().unwrap();
        unit.uninitialize();
        match self.direction {
            Direction::Output => {
                let route = self.output_route.as_ref().unwrap();
                let source = self
                    .control
                    .playback_maps()
                    .get(fmt.channels as usize - 1)
                    .ok_or(Error::UnexpectedAudioBackendConfiguration)?;
                let mixer = Mixer::new(fmt.format, source, &route.positions);
                unit.set(
                    kAudioUnitProperty_StreamFormat,
                    kAudioUnitScope_Input,
                    0,
                    &float_asbd(fmt.rate as f64, mixer.output_channels as u32),
                    "StreamFormat(output)",
                )?;
                // Identity map over every physical channel. A device wider than the mixer
                // (stereo route) keeps the HAL's default mapping instead.
                if route.device_channels == mixer.output_channels {
                    let map: Vec<i32> = (0..mixer.output_channels as i32).collect();
                    // SAFETY: map is a live array of physical output-channel indices.
                    check(
                        unsafe {
                            AudioUnitSetProperty(
                                unit.au,
                                kAudioOutputUnitProperty_ChannelMap,
                                kAudioUnitScope_Input,
                                0,
                                map.as_ptr().cast(),
                                (map.len() * 4) as u32,
                            )
                        },
                        "ChannelMap(output)",
                    )?;
                }
                // SAFETY: configure uninitialized the unit, so no render callback can read it.
                unsafe {
                    *self.shared.mixer.get() = mixer;
                }
                log::info!(
                    "snd: stream {} routes guest {:?} to '{}' speakers {:?}",
                    self.id,
                    source,
                    device_name(route.device),
                    route.positions
                );
                self.shared
                    .frame_bytes
                    .store(fmt.frame_bytes(), Ordering::Relaxed);
                // fifo_bytes: update_latency() below, once the unit knows its device.
            }
            Direction::Input => {
                let device: AudioObjectID = object_property(
                    kAudioObjectSystemObject,
                    kAudioHardwarePropertyDefaultInputDevice,
                    kAudioObjectPropertyScopeGlobal,
                )
                .filter(|d| *d != 0)
                .ok_or_else(|| Error::UnexpectedAudioBackendError("no input device".into()))?;
                unit.set(
                    kAudioOutputUnitProperty_CurrentDevice,
                    kAudioUnitScope_Global,
                    0,
                    &device,
                    "CurrentDevice(input)",
                )?;
                // The HAL unit does not resample input: take Float32 at the device rate and
                // convert in the pump thread.
                let rate: f64 = object_property(
                    device,
                    kAudioDevicePropertyNominalSampleRate,
                    kAudioObjectPropertyScopeGlobal,
                )
                .unwrap_or(48000.0);
                let dev_fmt = float_asbd(rate, fmt.channels);
                unit.set(
                    kAudioUnitProperty_StreamFormat,
                    kAudioUnitScope_Output,
                    1,
                    &dev_fmt,
                    "StreamFormat(input)",
                )?;
                let mut conv = ptr::null_mut();
                // SAFETY: valid formats and out pointer.
                check(
                    unsafe { AudioConverterNew(&dev_fmt, &fmt.asbd(), &mut conv) },
                    "AudioConverterNew",
                )?;
                self.converter = Some(Converter {
                    conv,
                    in_frame_bytes: dev_fmt.mBytesPerFrame as usize,
                    staging: Vec::new(),
                });
                self.shared
                    .frame_bytes
                    .store(dev_fmt.mBytesPerFrame, Ordering::Relaxed);
            }
        }
        self.shared.channels.store(fmt.channels, Ordering::Relaxed);
        self.shared.silence.store(fmt.silence(), Ordering::Relaxed);
        unit.initialize()?;
        if self.direction == Direction::Output {
            unit.set_volume(self.control.volume());
        }
        self.format = Some(fmt);
        self.update_latency();
        Ok(())
    }

    fn start_unit(&mut self) -> Result<()> {
        match self.unit.as_mut() {
            // Started by follow_pause() on resume.
            Some(unit) if unit.initialized && self.control.paused() => {
                self.suspended = true;
                Ok(())
            }
            Some(unit) if unit.initialized => unit.start(),
            _ => Err(Error::UnexpectedAudioBackendError(
                "stream not prepared".into(),
            )),
        }
    }

    fn stop_unit(&mut self) {
        self.suspended = false;
        if let Some(unit) = self.unit.as_mut() {
            unit.stop();
        }
    }

    /// Recompute the host FIFO (playback) and latency_bytes from the current device; logs
    /// device and device rate changes.
    fn update_latency(&mut self) {
        let Some(fmt) = self.format else {
            return;
        };
        let device = self.unit.as_ref().map_or(0, |u| u.current_device());
        let scope = match self.direction {
            Direction::Output => kAudioObjectPropertyScopeOutput,
            Direction::Input => kAudioObjectPropertyScopeInput,
        };
        let timing = (device != 0).then(|| device_timing(device, scope));
        if self.direction == Direction::Output {
            // The render callback takes a whole IO buffer at once, so the FIFO holds one (plus
            // the safety offset) for a guest that writes just in time; buffer_ms on top is
            // how long the guest may stall without an underrun.
            let io_frames = timing.as_ref().map_or(0, |t| {
                (t.io as f64 * fmt.rate as f64 / t.rate).ceil() as u64
            });
            let headroom = self.control.buffer_ms() as u64 * fmt.rate as u64 / 1000;
            let mut fifo = (io_frames + headroom) * fmt.frame_bytes() as u64;
            // The guest must keep part of its own buffer pending: a FIFO as large as that
            // buffer would complete everything it queues at once (to the guest, an underrun
            // right after START, after which PipeWire stops writing).
            if let Some((buffer, period)) = self.guest_buffer {
                fifo = fifo.min(buffer.saturating_sub(2 * period));
            }
            self.fifo_bytes = fifo / fmt.frame_bytes() as u64 * fmt.frame_bytes() as u64;
            let bytes_per_sec = fmt.rate as u64 * fmt.frame_bytes() as u64;
            self.shared
                .prime_bytes
                .store(self.fifo_bytes, Ordering::Relaxed);
            self.shared.prime_nanos.store(
                self.fifo_bytes * 1_000_000_000 / bytes_per_sec,
                Ordering::Relaxed,
            );
        }
        let Some(timing) = timing else {
            return;
        };
        let unit_secs: f64 = self
            .unit
            .as_ref()
            .and_then(|u| u.get(kAudioUnitProperty_Latency, kAudioUnitScope_Global, 0))
            .unwrap_or(0.0);
        let fifo_secs = self.fifo_bytes as f64 / (fmt.rate * fmt.frame_bytes()) as f64;
        let device_secs = (timing.latency + timing.io) as f64 / timing.rate;
        let secs = device_secs + unit_secs + fifo_secs;
        let guest_frames = (secs * fmt.rate as f64).round() as u32;
        let latency_bytes = guest_frames * fmt.frame_bytes();
        if device != self.device
            || timing.rate != self.device_rate
            || latency_bytes != self.latency_bytes
        {
            log::info!(
                "snd: stream {} {} '{}' at {} Hz, host latency {} frames ({:.1} ms{})",
                self.id,
                match self.direction {
                    Direction::Output => "playing on",
                    Direction::Input => "recording from",
                },
                device_name(device),
                timing.rate,
                guest_frames,
                secs * 1000.0,
                if fifo_secs > 0.0 {
                    format!(", {:.1} ms of it in the host FIFO", fifo_secs * 1000.0)
                } else {
                    String::new()
                },
            );
        }
        self.device = device;
        self.device_rate = timing.rate;
        self.latency_bytes = latency_bytes;
    }

    /// Capture: follow the default input device while recording (the HAL unit does not do it
    /// by itself; START always picks the current default).
    fn follow_input_device(&mut self) {
        let (Some(unit), Some(fmt)) = (self.unit.as_ref(), self.format) else {
            return;
        };
        let default: AudioObjectID = object_property(
            kAudioObjectSystemObject,
            kAudioHardwarePropertyDefaultInputDevice,
            kAudioObjectPropertyScopeGlobal,
        )
        .unwrap_or(0);
        if !unit.started || default == 0 || default == unit.current_device() {
            return;
        }
        self.stop_unit();
        self.shared.ring.clear();
        if let Err(err) = self.configure(fmt).and_then(|()| self.start_unit()) {
            log::error!("snd: stream {}: switching input device: {err}", self.id);
        }
    }

    /// Playback: let the device play what the guest already saw completed (the host FIFO)
    /// before the unit stops, so a drained stream keeps its tail.
    fn play_out(&self) {
        if self.unit.as_ref().is_none_or(|u| !u.started) {
            return;
        }
        let ring = &self.shared.ring;
        let end = self
            .shared
            .limit
            .load(Ordering::Acquire)
            .min(ring.written());
        let deadline =
            Instant::now() + Duration::from_millis(SndControl::MAX_BUFFER_MS as u64 + 100);
        while ring.consumed() < end && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// Complete everything the guest has queued on this stream and drop buffered audio.
    /// The unit must be stopped.
    fn flush(&mut self, guest: &mut Stream) {
        self.shared.running.store(false, Ordering::Release);
        self.shared.ring.clear();
        self.completed = self.shared.ring.written();
        self.shared.limit.store(self.completed, Ordering::Release);
        for (_, buf) in self.inflight.drain(..) {
            buf.message.latency_bytes.store(0, Ordering::Relaxed);
        }
        for buf in guest.buffers.drain(..) {
            buf.message.latency_bytes.store(0, Ordering::Relaxed);
        }
        self.drain_dump();
    }

    /// Move data between the guest's queued buffers and the ring; complete what is done.
    fn pump(&mut self, streams: &RwLock<Vec<Stream>>) {
        match self.direction {
            Direction::Output => self.pump_output(streams),
            Direction::Input => self.pump_input(streams),
        }
        self.drain_dump();
    }

    /// Ring position the device has played up to now: the last callback's chunk is spread over
    /// the time it takes to play, so completions follow the device clock instead of arriving in
    /// bursts. Never beyond what the callback has consumed.
    fn play_position(&self) -> u64 {
        let (time, consumed, chunk) = self.shared.clock.read();
        let Some(fmt) = self.format else {
            return consumed;
        };
        // SAFETY: plain clock read.
        let now = unsafe { mach_absolute_time() };
        let elapsed = ticks_to_nanos(now.saturating_sub(time)) as u128;
        let played = elapsed * fmt.rate as u128 * fmt.frame_bytes() as u128 / 1_000_000_000;
        consumed - chunk + (played as u64).min(chunk)
    }

    /// Time until the oldest in-flight playback message is due, if the device has taken it.
    fn next_completion(&self) -> Option<Duration> {
        let (end, _) = self.inflight.front()?;
        let fmt = self.format?;
        let due_at = end.saturating_sub(self.fifo_bytes);
        if due_at > self.shared.clock.read().1 {
            return None; // the render callback signals when it takes more
        }
        let bytes = due_at.saturating_sub(self.play_position()) as u128;
        let nanos = bytes * 1_000_000_000 / (fmt.rate as u128 * fmt.frame_bytes() as u128);
        Some(Duration::from_nanos(nanos as u64).max(Duration::from_micros(500)))
    }

    fn pump_output(&mut self, streams: &RwLock<Vec<Stream>>) {
        // The guest's hw_ptr only moves between START and STOP.
        let hw_ptr = if self.shared.running.load(Ordering::Acquire) {
            self.play_position() + self.fifo_bytes
        } else {
            0
        };
        while self.inflight.front().is_some_and(|(end, _)| *end <= hw_ptr) {
            let (end, buf) = self.inflight.pop_front().unwrap();
            self.completed = end;
            buf.message
                .latency_bytes
                .store(self.latency_bytes, Ordering::Relaxed);
        }

        let mut streams = streams.write().unwrap();
        let guest = &mut streams[self.id];
        if !matches!(
            guest.state,
            PCMState::Prepare | PCMState::Start | PCMState::Stop
        ) {
            // I/O outside PREPARE..RELEASE has nothing to play: complete it at once. Held
            // back, it would stay pending in the guest, whose next prepare then fails with
            // "invalid I/O queue state" for good.
            if !guest.buffers.is_empty() {
                trace!(
                    "stream {}: completing {} messages queued in state {:?}",
                    self.id,
                    guest.buffers.len(),
                    guest.state
                );
            }
            for buf in guest.buffers.drain(..) {
                buf.message
                    .status
                    .store(VIRTIO_SND_S_BAD_MSG, Ordering::SeqCst);
            }
            return;
        }
        // Fill the ring even while the unit is not running (yet), so it starts primed.
        let ring = &self.shared.ring;
        while let Some(buf) = guest.buffers.front_mut() {
            let remaining = (buf.desc_len() as usize).saturating_sub(buf.pos);
            let mut failed = false;
            // SAFETY: the pump (under the backend mutex) is the playback ring's only producer.
            let n = unsafe {
                ring.produce(remaining, |dst| match buf.read_output(dst) {
                    Ok(len) => {
                        buf.pos += len as usize;
                        len as usize
                    }
                    Err(_) => {
                        failed = true;
                        0
                    }
                })
            };
            if failed {
                log::error!("snd: stream {}: unreadable playback buffer", self.id);
                buf.message
                    .status
                    .store(VIRTIO_SND_S_IO_ERR, Ordering::SeqCst);
            } else if n < remaining {
                break; // ring full
            }
            let buf = guest.buffers.pop_front().unwrap();
            self.inflight.push_back((ring.written(), buf));
        }
    }

    fn pump_input(&mut self, streams: &RwLock<Vec<Stream>>) {
        let (Some(fmt), Some(conv)) = (self.format, self.converter.as_mut()) else {
            return;
        };
        if !self.shared.running.load(Ordering::Acquire) {
            return;
        }
        let mut streams = streams.write().unwrap();
        let guest = &mut streams[self.id];
        let fb = fmt.frame_bytes() as usize;
        let mut out = Vec::new();
        while let Some(buf) = guest.buffers.front_mut() {
            let space = (buf.desc_len() as usize).saturating_sub(buf.pos) / fb * fb;
            if space > 0 {
                out.resize(space, 0);
                let mut packets = (space / fb) as u32;
                let mut abl = AudioBufferList {
                    mNumberBuffers: 1,
                    mBuffers: [AudioBuffer {
                        mNumberChannels: fmt.channels,
                        mDataByteSize: space as u32,
                        mData: out.as_mut_ptr() as *mut c_void,
                    }],
                };
                let mut input = ConverterInput {
                    ring: &self.shared.ring,
                    staging: &mut conv.staging,
                    frame_bytes: conv.in_frame_bytes,
                    channels: fmt.channels,
                };
                // SAFETY: abl describes `out`; `input` outlives the call.
                let status = unsafe {
                    AudioConverterFillComplexBuffer(
                        conv.conv,
                        converter_input,
                        &mut input as *mut ConverterInput as *mut c_void,
                        &mut packets,
                        &mut abl,
                        ptr::null_mut(),
                    )
                };
                if status != 0 && status != NO_MORE_INPUT {
                    log::error!("snd: stream {}: AudioConverter: {status}", self.id);
                }
                let produced = packets as usize * fb;
                if produced == 0 {
                    break;
                }
                if buf.write_input(&out[..produced]).is_err() {
                    log::error!("snd: stream {}: unwritable capture buffer", self.id);
                    buf.message
                        .status
                        .store(VIRTIO_SND_S_IO_ERR, Ordering::SeqCst);
                    buf.pos = buf.desc_len() as usize;
                }
                if buf.pos + fb <= buf.desc_len() as usize {
                    continue;
                }
            }
            let buf = guest.buffers.pop_front().unwrap();
            buf.message
                .latency_bytes
                .store(self.latency_bytes, Ordering::Relaxed);
        }
    }

    fn drain_dump(&mut self) {
        let Some(ring) = &self.shared.dump else {
            return;
        };
        let mut failed = false;
        let mut dump = self.dump.as_mut().map(|(_, d)| d);
        // SAFETY: the pump (under the backend mutex) is the dump ring's only consumer.
        unsafe {
            ring.consume(usize::MAX, |src| {
                if let Some(d) = dump.as_deref_mut() {
                    failed |= d.append(src).is_err();
                }
                src.len()
            })
        };
        if failed {
            log::error!("snd: writing the playback dump failed, stopping it");
            self.dump = None;
            self.dump_path = None;
        }
    }

    /// Start a new dump file when the format changes (the first keeps the given name).
    fn open_dump(&mut self, fmt: Format) {
        let Some(path) = &self.dump_path else {
            return;
        };
        if self.dump.as_ref().is_some_and(|(f, _)| *f == fmt) {
            return;
        }
        let path = if self.dump_files == 0 {
            path.clone()
        } else {
            let stem = path.file_stem().unwrap_or_default().to_string_lossy();
            path.with_file_name(format!("{stem}-{}.wav", self.dump_files))
        };
        match Dump::create(&path, &fmt) {
            Ok(d) => {
                log::info!("snd: dumping playback ({fmt:?}) to {}", path.display());
                self.dump_files += 1;
                self.dump = Some((fmt, d));
            }
            Err(err) => log::error!("snd: cannot create {}: {err}", path.display()),
        }
    }

    fn housekeeping(&mut self, streams: &RwLock<Vec<Stream>>) {
        if self.direction == Direction::Input {
            self.follow_input_device();
        }
        if self.unit.as_ref().is_some_and(|u| u.started) {
            self.update_latency();
        }
        let xruns = self.shared.xruns.load(Ordering::Relaxed);
        if xruns != self.reported_xruns {
            log::warn!(
                "snd: stream {}: {} {} (total {xruns})",
                self.id,
                xruns - self.reported_xruns,
                match self.direction {
                    Direction::Output => "underruns",
                    Direction::Input => "overruns",
                },
            );
            self.reported_xruns = xruns;
        }
        if *TRACE {
            let callbacks = self.shared.callbacks.load(Ordering::Relaxed);
            let guest = &streams.read().unwrap()[self.id];
            if callbacks != self.traced_callbacks
                || !self.inflight.is_empty()
                || !guest.buffers.is_empty()
            {
                self.traced_callbacks = callbacks;
                self.trace_state("heartbeat", Some(guest));
            }
        }
    }
}

impl Drop for HostStream {
    fn drop(&mut self) {
        // The unit references `shared`: stop and dispose it first.
        self.unit = None;
    }
}

struct Inner {
    streams: Arc<RwLock<Vec<Stream>>>,
    hosts: Mutex<Vec<HostStream>>,
    wake: Arc<Semaphore>,
    quit: AtomicBool,
    /// Set by `route_listener`: recheck the output route on the next pass.
    route_changed: AtomicBool,
}

/// System properties after which the output route is rechecked at once: the default output
/// (AirPods connecting, Control Center) and the device list (a selected device unplugged).
const ROUTE_SELECTORS: [u32; 2] = [
    kAudioHardwarePropertyDefaultOutputDevice,
    kAudioHardwarePropertyDevices,
];

extern "C" fn route_listener(
    _object: AudioObjectID,
    _count: u32,
    _addresses: *const AudioObjectPropertyAddress,
    data: *mut c_void,
) -> OSStatus {
    // SAFETY: data is the Inner of a CoreAudioBackend, which removes this listener before it
    // releases its reference.
    let inner = unsafe { &*(data as *const Inner) };
    inner.route_changed.store(true, Ordering::Release);
    inner.wake.signal();
    0
}

fn pump_thread(inner: Arc<Inner>) {
    utils::macos::set_thread_qos_user_interactive();
    let mut last = Instant::now();
    let mut timeout = HOUSEKEEPING;
    while !inner.quit.load(Ordering::Acquire) {
        inner.wake.wait(timeout);
        timeout = HOUSEKEEPING;
        let mut hosts = inner.hosts.lock().unwrap();
        let recheck =
            inner.route_changed.swap(false, Ordering::AcqRel) || last.elapsed() >= HOUSEKEEPING;
        for host in hosts.iter_mut() {
            host.apply_control(recheck);
            host.pump(&inner.streams);
            if let Some(due) = host.next_completion() {
                timeout = timeout.min(due);
            }
        }
        if last.elapsed() >= HOUSEKEEPING {
            last = Instant::now();
            for host in hosts.iter_mut() {
                host.housekeeping(&inner.streams);
            }
        }
    }
}

pub struct CoreAudioBackend {
    inner: Arc<Inner>,
    pump: Option<JoinHandle<()>>,
}

impl CoreAudioBackend {
    pub fn new(streams: Arc<RwLock<Vec<Stream>>>, control: Arc<SndControl>) -> Result<Self> {
        let wake = Arc::new(Semaphore::new()?);
        let waker = wake.clone();
        control.set_waker(Box::new(move || waker.signal()));
        let hosts = streams
            .read()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(id, s)| HostStream::new(id, s.direction, wake.clone(), control.clone()))
            .collect();
        let inner = Arc::new(Inner {
            streams,
            hosts: Mutex::new(hosts),
            wake,
            quit: AtomicBool::new(false),
            route_changed: AtomicBool::new(false),
        });
        let pump_inner = inner.clone();
        let pump = std::thread::Builder::new()
            .name("virtio-snd coreaudio".into())
            .spawn(move || pump_thread(pump_inner))
            .map_err(|e| Error::UnexpectedAudioBackendError(e.to_string()))?;
        for selector in ROUTE_SELECTORS {
            let addr = property_address(selector, kAudioObjectPropertyScopeGlobal);
            // SAFETY: `inner` outlives the registration; Drop removes it first.
            let status = unsafe {
                AudioObjectAddPropertyListener(
                    kAudioObjectSystemObject,
                    &addr,
                    route_listener,
                    Arc::as_ptr(&inner) as *mut c_void,
                )
            };
            if status != 0 {
                log::warn!(
                    "snd: output route listener {selector:#x}: OSStatus {status}; the output \
                     follows device changes once a second"
                );
            }
        }
        Ok(Self {
            inner,
            pump: Some(pump),
        })
    }

    /// Move the stream to the state of a PCM command. Linux sends STOP, RELEASE, PREPARE,
    /// START in the order the spec allows, but a guest that does not (or a resume after
    /// suspend) must not end up with the device stopped while it runs and waits forever for
    /// completions: the device always follows the guest and only logs an unexpected order.
    fn transition(
        &self,
        stream_id: u32,
        what: &str,
        target: PCMState,
        f: impl FnOnce(&mut PCMState) -> std::result::Result<(), StreamError>,
    ) -> Result<()> {
        let mut streams = self.inner.streams.write().unwrap();
        let stream = streams
            .get_mut(stream_id as usize)
            .ok_or(Error::StreamWithIdNotFound(stream_id))?;
        if let Err(err) = f(&mut stream.state) {
            log::debug!("snd: stream {stream_id} {what}: {err}");
            trace!("stream {stream_id} {what} from {:?}: {err}", stream.state);
            stream.state = target;
        }
        Ok(())
    }

    fn trace(&self, host: &HostStream, what: &str) {
        if *TRACE {
            host.trace_state(what, Some(&self.inner.streams.read().unwrap()[host.id]));
        }
    }
}

impl Drop for CoreAudioBackend {
    fn drop(&mut self) {
        for selector in ROUTE_SELECTORS {
            let addr = property_address(selector, kAudioObjectPropertyScopeGlobal);
            // SAFETY: same address, proc and client data as registered in new().
            unsafe {
                AudioObjectRemovePropertyListener(
                    kAudioObjectSystemObject,
                    &addr,
                    route_listener,
                    Arc::as_ptr(&self.inner) as *mut c_void,
                )
            };
        }
        self.inner.quit.store(true, Ordering::Release);
        self.inner.wake.signal();
        if let Some(pump) = self.pump.take() {
            let _ = pump.join();
        }
    }
}

impl AudioBackend for CoreAudioBackend {
    fn write(&self, _stream_id: u32) -> Result<()> {
        self.inner.wake.signal();
        Ok(())
    }

    fn read(&self, _stream_id: u32) -> Result<()> {
        self.inner.wake.signal();
        Ok(())
    }

    fn set_parameters(&self, stream_id: u32, request: VirtioSndPcmSetParams) -> Result<()> {
        let mut streams = self.inner.streams.write().unwrap();
        let st = streams
            .get_mut(stream_id as usize)
            .ok_or(Error::StreamWithIdNotFound(stream_id))?;
        if !st.supports_format(request.format)
            || !st.supports_rate(request.rate)
            || !(st.channels_min..=st.channels_max).contains(&request.channels)
        {
            return Err(Error::UnexpectedAudioBackendConfiguration);
        }
        st.state.set_parameters().map_err(|err| {
            log::error!("snd: stream {stream_id} set_parameters: {err}");
            Error::Stream(err)
        })?;
        st.params.features = request.features;
        st.params.buffer_bytes = request.buffer_bytes;
        st.params.period_bytes = request.period_bytes;
        st.params.channels = request.channels;
        st.params.format = request.format;
        st.params.rate = request.rate;
        Ok(())
    }

    fn prepare(&self, stream_id: u32) -> Result<()> {
        self.transition(stream_id, "prepare", PCMState::Prepare, PCMState::prepare)?;
        let (fmt, buffer) = {
            let params = &self.inner.streams.read().unwrap()[stream_id as usize].params;
            let fmt =
                Format::from_params(params).ok_or(Error::UnexpectedAudioBackendConfiguration)?;
            let buffer = (
                u32::from(params.buffer_bytes) as u64,
                u32::from(params.period_bytes) as u64,
            );
            (fmt, buffer)
        };
        let mut hosts = self.inner.hosts.lock().unwrap();
        let host = &mut hosts[stream_id as usize];
        self.trace(host, "PREPARE");
        host.stop_unit();
        host.flush(&mut self.inner.streams.write().unwrap()[stream_id as usize]);
        host.guest_format = Some(fmt);
        if host.direction == Direction::Output {
            host.guest_buffer = Some(buffer);
            if host.format != Some(fmt) || host.unit.is_none() {
                host.configure(fmt)?;
            } else {
                host.update_latency(); // the FIFO depends on the guest buffer
            }
            host.open_dump(fmt);
            // Run from PREPARE so START does not wait for the device; the callback plays
            // silence until then.
            host.start_unit()?;
        }
        self.trace(host, "PREPARE done");
        Ok(())
    }

    fn release(&self, stream_id: u32) -> Result<()> {
        self.transition(stream_id, "release", PCMState::Release, PCMState::release)?;
        let mut hosts = self.inner.hosts.lock().unwrap();
        let host = &mut hosts[stream_id as usize];
        self.trace(host, "RELEASE");
        if host.direction == Direction::Output {
            host.play_out();
        }
        host.stop_unit();
        host.flush(&mut self.inner.streams.write().unwrap()[stream_id as usize]);
        self.trace(host, "RELEASE done");
        Ok(())
    }

    fn start(&self, stream_id: u32) -> Result<()> {
        self.transition(stream_id, "start", PCMState::Start, PCMState::start)?;
        let mut hosts = self.inner.hosts.lock().unwrap();
        let host = &mut hosts[stream_id as usize];
        host.shared.primed.store(false, Ordering::Relaxed);
        host.shared.starved.store(false, Ordering::Relaxed);
        // SAFETY: plain clock read.
        host.shared
            .prime_since
            .store(unsafe { mach_absolute_time() }, Ordering::Relaxed);
        if host.direction == Direction::Input {
            // The microphone (and its permission prompt) is only touched while the guest
            // records: set the unit up for the current default input device now.
            let fmt = host
                .guest_format
                .ok_or(Error::UnexpectedAudioBackendConfiguration)?;
            host.stop_unit();
            host.shared.ring.clear();
            host.configure(fmt)?;
            host.shared.running.store(true, Ordering::Release);
            if let Err(err) = host.start_unit() {
                host.shared.running.store(false, Ordering::Release);
                return Err(err);
            }
        } else {
            // A guest starting without a PREPARE after RELEASE (which stopped the unit), or
            // a unit that failed to start at PREPARE: make sure the device is playing.
            if host.unit.as_ref().is_none_or(|u| !u.started) {
                let fmt = match host.guest_format {
                    Some(fmt) => fmt,
                    None => Format::from_params(
                        &self.inner.streams.read().unwrap()[stream_id as usize].params,
                    )
                    .ok_or(Error::UnexpectedAudioBackendConfiguration)?,
                };
                if host.format != Some(fmt) || host.unit.is_none() {
                    host.configure(fmt)?;
                }
                host.start_unit()?;
            }
            host.shared.limit.store(u64::MAX, Ordering::Release);
            host.shared.running.store(true, Ordering::Release);
        }
        self.trace(host, "START");
        self.inner.wake.signal();
        Ok(())
    }

    fn stop(&self, stream_id: u32) -> Result<()> {
        self.transition(stream_id, "stop", PCMState::Stop, PCMState::stop)?;
        let mut hosts = self.inner.hosts.lock().unwrap();
        let host = &mut hosts[stream_id as usize];
        host.shared.running.store(false, Ordering::Release);
        match host.direction {
            // Play out what the guest already saw completed (drain), keep the rest (pause).
            Direction::Output => host.shared.limit.store(host.completed, Ordering::Release),
            Direction::Input => host.stop_unit(),
        }
        self.trace(host, "STOP");
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
