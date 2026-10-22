// Speaker positions and allocation-free PCM routing for the macOS backend.
use super::super::virtio_sound::*;

pub const MAX_CHANNELS: usize = 12;
pub const MAX_DEVICE_CHANNELS: usize = 32;

pub fn apple_position(label: u32) -> u8 {
    match label {
        1 | 301 => VIRTIO_SND_CHMAP_FL,
        2 | 302 => VIRTIO_SND_CHMAP_FR,
        3 => VIRTIO_SND_CHMAP_FC,
        4 => VIRTIO_SND_CHMAP_LFE,
        5 | 10 | 55 => VIRTIO_SND_CHMAP_SL,
        6 | 11 | 56 => VIRTIO_SND_CHMAP_SR,
        7 => VIRTIO_SND_CHMAP_FLC,
        8 => VIRTIO_SND_CHMAP_FRC,
        9 | 44 => VIRTIO_SND_CHMAP_RC,
        12 => VIRTIO_SND_CHMAP_TC,
        13 => VIRTIO_SND_CHMAP_TFL,
        14 => VIRTIO_SND_CHMAP_TFC,
        15 => VIRTIO_SND_CHMAP_TFR,
        16 | 52 => VIRTIO_SND_CHMAP_TRL,
        17 | 53 => VIRTIO_SND_CHMAP_TRC,
        18 | 54 => VIRTIO_SND_CHMAP_TRR,
        33 | 63 => VIRTIO_SND_CHMAP_RL,
        34 | 64 => VIRTIO_SND_CHMAP_RR,
        42 => VIRTIO_SND_CHMAP_MONO,
        49 | 60 => VIRTIO_SND_CHMAP_TSL,
        51 | 61 => VIRTIO_SND_CHMAP_TSR,
        _ => VIRTIO_SND_CHMAP_NONE,
    }
}

/// Decode expanded AudioChannelLayout descriptions; never infer speakers from count.
pub fn descriptions(bytes: &[u8], channels: usize) -> Option<Vec<u8>> {
    let word = |i| {
        bytes
            .get(i..i + 4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
    };
    if word(0)? != 0
        || word(8)? as usize != channels
        || channels == 0
        || channels > MAX_DEVICE_CHANNELS
        || bytes.len() < 12 + channels * 20
    {
        return None;
    }
    let positions: Vec<_> = (0..channels)
        .map(|i| apple_position(word(12 + i * 20).unwrap()))
        .collect();
    // Discrete/coordinate/duplicate labels cannot identify a surround speaker layout.
    if positions.contains(&VIRTIO_SND_CHMAP_NONE)
        || positions
            .iter()
            .enumerate()
            .any(|(i, p)| positions[..i].contains(p))
    {
        return None;
    }
    Some(positions)
}

/// ALSA order, plus the detected height positions. One fixed map per channel count.
pub fn guest_maps(device: &[u8]) -> Vec<Vec<u8>> {
    let order = [
        VIRTIO_SND_CHMAP_FL,
        VIRTIO_SND_CHMAP_FR,
        VIRTIO_SND_CHMAP_RL,
        VIRTIO_SND_CHMAP_RR,
        VIRTIO_SND_CHMAP_FC,
        VIRTIO_SND_CHMAP_LFE,
        VIRTIO_SND_CHMAP_SL,
        VIRTIO_SND_CHMAP_SR,
        VIRTIO_SND_CHMAP_RC,
        VIRTIO_SND_CHMAP_FLC,
        VIRTIO_SND_CHMAP_FRC,
        VIRTIO_SND_CHMAP_TC,
        VIRTIO_SND_CHMAP_TFL,
        VIRTIO_SND_CHMAP_TFR,
        VIRTIO_SND_CHMAP_TFC,
        VIRTIO_SND_CHMAP_TRL,
        VIRTIO_SND_CHMAP_TRR,
        VIRTIO_SND_CHMAP_TRC,
        VIRTIO_SND_CHMAP_TSL,
        VIRTIO_SND_CHMAP_TSR,
    ];
    let ordered: Vec<_> = order
        .into_iter()
        .filter(|p| device.contains(p))
        .take(MAX_CHANNELS)
        .collect();
    if !ordered.starts_with(&[VIRTIO_SND_CHMAP_FL, VIRTIO_SND_CHMAP_FR]) {
        return vec![vec![VIRTIO_SND_CHMAP_MONO]];
    }
    let mut maps = vec![vec![VIRTIO_SND_CHMAP_MONO]];
    for n in 2..=ordered.len() {
        maps.push(ordered[..n].to_vec());
    }
    maps
}

#[derive(Clone)]
pub struct Mixer {
    pub format: u8,
    pub source_channels: usize,
    pub output_channels: usize,
    weights: [[f32; MAX_CHANNELS]; MAX_DEVICE_CHANNELS],
}

impl Default for Mixer {
    fn default() -> Self {
        Self::new(
            VIRTIO_SND_PCM_FMT_S16,
            &[VIRTIO_SND_CHMAP_FL, VIRTIO_SND_CHMAP_FR],
            &[VIRTIO_SND_CHMAP_FL, VIRTIO_SND_CHMAP_FR],
        )
    }
}

impl Mixer {
    pub fn new(format: u8, source: &[u8], output: &[u8]) -> Self {
        assert!(!source.is_empty() && source.len() <= MAX_CHANNELS);
        assert!(!output.is_empty() && output.len() <= MAX_DEVICE_CHANNELS);
        let mut weights = [[0.0; MAX_CHANNELS]; MAX_DEVICE_CHANNELS];
        for (s, position) in source.iter().copied().enumerate() {
            if let Some(d) = output.iter().position(|p| *p == position) {
                weights[d][s] = 1.0;
                continue;
            }
            let counterpart = match position {
                VIRTIO_SND_CHMAP_SL => VIRTIO_SND_CHMAP_RL,
                VIRTIO_SND_CHMAP_SR => VIRTIO_SND_CHMAP_RR,
                VIRTIO_SND_CHMAP_RL => VIRTIO_SND_CHMAP_SL,
                VIRTIO_SND_CHMAP_RR => VIRTIO_SND_CHMAP_SR,
                _ => VIRTIO_SND_CHMAP_NONE,
            };
            if counterpart != VIRTIO_SND_CHMAP_NONE {
                if let Some(d) = output.iter().position(|p| *p == counterpart) {
                    weights[d][s] = 1.0;
                    continue;
                }
            }
            let left = matches!(
                position,
                VIRTIO_SND_CHMAP_FL
                    | VIRTIO_SND_CHMAP_RL
                    | VIRTIO_SND_CHMAP_SL
                    | VIRTIO_SND_CHMAP_FLC
                    | VIRTIO_SND_CHMAP_TFL
                    | VIRTIO_SND_CHMAP_TRL
                    | VIRTIO_SND_CHMAP_TSL
            );
            let right = matches!(
                position,
                VIRTIO_SND_CHMAP_FR
                    | VIRTIO_SND_CHMAP_RR
                    | VIRTIO_SND_CHMAP_SR
                    | VIRTIO_SND_CHMAP_FRC
                    | VIRTIO_SND_CHMAP_TFR
                    | VIRTIO_SND_CHMAP_TRR
                    | VIRTIO_SND_CHMAP_TSR
            );
            for (d, target) in output.iter().copied().enumerate() {
                if (target == VIRTIO_SND_CHMAP_FL && !right)
                    || (target == VIRTIO_SND_CHMAP_FR && !left)
                    || target == VIRTIO_SND_CHMAP_MONO
                    || (output.len() == 1 && target != VIRTIO_SND_CHMAP_NONE)
                {
                    weights[d][s] = if position == VIRTIO_SND_CHMAP_MONO {
                        1.0
                    } else if position == VIRTIO_SND_CHMAP_LFE {
                        0.5
                    } else {
                        std::f32::consts::FRAC_1_SQRT_2
                    };
                }
            }
        }
        // Headroom for a full-scale correlated downmix; direct routing remains unity gain.
        for row in &mut weights {
            let sum: f32 = row.iter().sum();
            if sum > 1.0 {
                for w in row {
                    *w /= sum;
                }
            }
        }
        Self {
            format,
            source_channels: source.len(),
            output_channels: output.len(),
            weights,
        }
    }

    pub fn sample_bytes(&self) -> usize {
        match self.format {
            VIRTIO_SND_PCM_FMT_U8 => 1,
            VIRTIO_SND_PCM_FMT_S16 => 2,
            _ => 4,
        }
    }

    /// Guest interleaved PCM -> device interleaved Float32. No heap work on the IO thread.
    pub fn render(&self, source: &[u8], output: &mut [u8]) {
        output.fill(0);
        let bytes = self.sample_bytes();
        for (frame, dest) in source
            .chunks_exact(bytes * self.source_channels)
            .zip(output.chunks_exact_mut(4 * self.output_channels))
        {
            let mut samples = [0.0; MAX_CHANNELS];
            for (s, value) in frame.chunks_exact(bytes).enumerate() {
                samples[s] = match self.format {
                    VIRTIO_SND_PCM_FMT_U8 => (value[0] as f32 - 128.0) / 128.0,
                    VIRTIO_SND_PCM_FMT_S16 => {
                        i16::from_le_bytes(value.try_into().unwrap()) as f32 / 32768.0
                    }
                    VIRTIO_SND_PCM_FMT_S32 => {
                        i32::from_le_bytes(value.try_into().unwrap()) as f32 / 2147483648.0
                    }
                    _ => {
                        let v = f32::from_le_bytes(value.try_into().unwrap());
                        if v.is_finite() {
                            v
                        } else {
                            0.0
                        }
                    }
                };
            }
            for (d, sample) in dest.chunks_exact_mut(4).enumerate() {
                let value: f32 = samples
                    .iter()
                    .zip(self.weights[d])
                    .map(|(v, w)| v * w)
                    .sum();
                sample.copy_from_slice(&value.clamp(-1.0, 1.0).to_le_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn route(source: &[u8], output: &[u8], values: &[f32]) -> Vec<f32> {
        let mixer = Mixer::new(VIRTIO_SND_PCM_FMT_FLOAT, source, output);
        let input: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut data = vec![0; output.len() * 4];
        mixer.render(&input, &mut data);
        data.chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()
    }
    #[test]
    fn dolby_top_middle_uses_linux_uapi_numbers() {
        let speakers: Vec<_> = [1, 2, 3, 4, 5, 6, 49, 51]
            .iter().map(|&label| apple_position(label)).collect();
        assert_eq!(guest_maps(&speakers)[7], [3, 4, 7, 8, 9, 10, 30, 31]);
        assert_eq!([VIRTIO_SND_CHMAP_LLFE, VIRTIO_SND_CHMAP_RLFE,
            VIRTIO_SND_CHMAP_BC, VIRTIO_SND_CHMAP_BLC, VIRTIO_SND_CHMAP_BRC],
            [32, 33, 34, 35, 36]);
    }
    #[test]
    fn maps_preserve_height_and_alsa_order() {
        let speakers = [3, 4, 7, 8, 9, 10, 22, 23];
        let maps = guest_maps(&speakers);
        assert_eq!(maps.len(), 8);
        assert_eq!(maps[7], [3, 4, 7, 8, 9, 10, 22, 23]);
        assert!(!maps[7].contains(&VIRTIO_SND_CHMAP_RL));
        assert_eq!(
            guest_maps(&[3, 4, 7, 8, 9, 10, 5, 6])[7],
            [3, 4, 5, 6, 7, 8, 9, 10]
        );
    }
    #[test]
    fn permutation_and_height_impulses() {
        let source = [3, 4, 7, 8, 9, 10, 22, 23];
        let device = [22, 23, 9, 10, 8, 7, 4, 3];
        for channel in 0..8 {
            let mut input = [0.0; 8];
            input[channel] = 0.5;
            let out = route(&source, &device, &input);
            for d in 0..8 {
                assert_eq!(
                    out[d],
                    if source[channel] == device[d] {
                        0.5
                    } else {
                        0.0
                    }
                );
            }
        }
    }
    #[test]
    fn switching_to_stereo_keeps_center_lfe_and_height_audible() {
        let source = [3, 4, 7, 8, 9, 10, 22, 23];
        for channel in 0..8 {
            let mut input = [0.0; 8];
            input[channel] = 1.0;
            let out = route(&source, &[3, 4], &input);
            assert!(out.iter().any(|v| *v > 0.0));
            if [0, 4, 6].contains(&channel) {
                assert_eq!(out[1], 0.0);
            }
            if [1, 5, 7].contains(&channel) {
                assert_eq!(out[0], 0.0);
            }
        }
        assert!(route(&source, &[3, 4], &[1.0; 8]).iter().all(|v| *v <= 1.0));
    }
    #[test]
    fn stereo_does_not_fill_height_speakers() {
        assert_eq!(
            route(&[3, 4], &[3, 4, 22, 23], &[0.25, -0.5]),
            [0.25, -0.5, 0.0, 0.0]
        );
        assert_eq!(route(&[2], &[3, 4], &[0.5]), [0.5, 0.5]);
    }
    #[test]
    fn mono_center_device_is_audible_and_surround_aliases_route() {
        assert_eq!(route(&[2], &[7], &[0.5]), [0.5]);
        let out = route(&[3, 4, 7, 8, 9, 10], &[7], &[1.0; 6]);
        assert!(out[0] > 0.0 && out[0] <= 1.0);
        assert_eq!(route(&[9, 10], &[5, 6], &[0.25, -0.5]), [0.25, -0.5]);
    }
    #[test]
    fn unknown_physical_channels_remain_silent_and_guest_maps_are_bounded() {
        assert_eq!(
            route(&[3, 4], &[0, 3, 0, 4], &[0.25, -0.5]),
            [0.0, 0.25, 0.0, -0.5]
        );
        assert_eq!(guest_maps(&[0, 3, 0, 4]), [vec![2], vec![3, 4]]);
        assert_eq!(guest_maps(&[3, 4, 5, 6, 7, 8, 9, 10, 22, 23]).len(), 10);
        assert_eq!(guest_maps(&[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 30, 31]).len(), 12);
    }
    #[test]
    fn seven_one_four_preserves_all_speakers_and_routes_each_channel() {
        // Physical order: front pair, center/LFE, sides, rears, top front/rear.
        let speakers = [3, 4, 7, 8, 9, 10, 5, 6, 22, 23, 25, 26];
        let maps = guest_maps(&speakers);
        let source = [3, 4, 5, 6, 7, 8, 9, 10, 22, 23, 25, 26];
        assert_eq!(maps.len(), 12);
        assert_eq!(maps[11], source);
        let output = [26, 25, 23, 22, 6, 5, 10, 9, 8, 7, 4, 3];
        for channel in 0..12 {
            let mut input = [0.0; 12];
            input[channel] = 0.5;
            let rendered = route(&source, &output, &input);
            for (index, speaker) in output.iter().enumerate() {
                assert_eq!(rendered[index], if *speaker == source[channel] { 0.5 } else { 0.0 });
            }
            let stereo = route(&source, &[3, 4], &input);
            assert!(stereo.iter().any(|sample| *sample > 0.0));
        }
        assert!(route(&source, &[3, 4], &[1.0; 12]).iter().all(|sample| *sample <= 1.0));
        let mut bytes = vec![0; 12 + 12 * 20];
        bytes[8..12].copy_from_slice(&12u32.to_ne_bytes());
        let labels = [1u32, 2, 3, 4, 5, 6, 33, 34, 13, 15, 16, 18];
        for (index, label) in labels.iter().enumerate() {
            bytes[12 + index * 20..16 + index * 20].copy_from_slice(&label.to_ne_bytes());
        }
        assert_eq!(descriptions(&bytes, 12).unwrap(), speakers);
    }
    #[test]
    fn parses_labels_but_rejects_unknown_truncated_and_duplicate_layouts() {
        let mut bytes = vec![0; 12 + 8 * 20];
        bytes[8..12].copy_from_slice(&8u32.to_ne_bytes());
        for (i, label) in [1u32, 2, 3, 4, 5, 6, 13, 15].iter().enumerate() {
            bytes[12 + i * 20..16 + i * 20].copy_from_slice(&label.to_ne_bytes());
        }
        assert_eq!(
            descriptions(&bytes, 8).unwrap(),
            [3, 4, 7, 8, 9, 10, 22, 23]
        );
        assert!(descriptions(&bytes[..20], 8).is_none());
        bytes[12..16].copy_from_slice(&0x10000u32.to_ne_bytes());
        assert!(descriptions(&bytes, 8).is_none());
        bytes[12..16].copy_from_slice(&2u32.to_ne_bytes());
        assert!(descriptions(&bytes, 8).is_none());
    }
    #[test]
    fn decodes_all_advertised_sample_formats() {
        for (format, bytes) in [
            (VIRTIO_SND_PCM_FMT_U8, vec![192]),
            (VIRTIO_SND_PCM_FMT_S16, 16384i16.to_le_bytes().to_vec()),
            (VIRTIO_SND_PCM_FMT_S32, 1073741824i32.to_le_bytes().to_vec()),
            (VIRTIO_SND_PCM_FMT_FLOAT, 0.5f32.to_le_bytes().to_vec()),
        ] {
            let mut out = [0; 4];
            Mixer::new(format, &[2], &[2]).render(&bytes, &mut out);
            assert_eq!(f32::from_le_bytes(out), 0.5);
        }
    }
}
