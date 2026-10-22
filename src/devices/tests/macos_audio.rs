#![cfg(all(target_os = "macos", feature = "snd"))]
#![allow(dead_code)]
// Integration target avoids unrelated, stale upstream virtio-snd unit tests.
mod virtio_sound {
    include!("../src/virtio/snd/virtio_sound.rs");
}
mod audio_backends {
    mod channel_layout {
        include!("../src/virtio/snd/audio_backends/channel_layout.rs");
    }
}
use krun_devices::virtio::{snd::Snd, SndControl, VirtioDevice};
use std::sync::Arc;

#[test]
fn detected_maps_match_config_and_stay_stable_across_output_changes() {
    let control = Arc::new(SndControl::default());
    let device = Snd::new(control.clone()).unwrap();
    let mut config = [0; 12];
    device.read_config(0, &mut config);
    let maps = control.playback_maps().clone();
    assert!((1..=12).contains(&maps.len()));
    for (i, map) in maps.iter().enumerate() {
        assert_eq!(map.len(), i + 1);
    }
    assert_eq!(
        u32::from_le_bytes(config[8..12].try_into().unwrap()),
        maps.len() as u32 + 2
    );
    eprintln!("Detected guest playback maps: {maps:?}");
    // A missing selected UID falls back to default; cached guest maps never change midstream.
    control.set_output_device(Some("steamac-test-nonexistent-device".into()));
    assert_eq!(*control.playback_maps(), maps);
    device.read_config(0, &mut config);
    assert_eq!(
        u32::from_le_bytes(config[8..12].try_into().unwrap()),
        maps.len() as u32 + 2
    );
}
