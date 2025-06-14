use std::collections::HashMap;

#[derive(Debug, serde::Deserialize)]
pub struct Config {
    #[serde(default)]
    pub bert_model_path: Option<String>,

    #[serde(default)]
    pub g2pw_model_path: Option<String>,

    pub ssl_model_path: String,
    pub mini_bart_g2p_path: String,

    #[serde(default)]
    pub device: RuntimeDevice,

    pub speaker: Vec<SpeakerConfig>,
}

#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub enum Version {
    V2,
    V2_1,
    V3,
    V4,
    V2Pro,
}

impl Default for Version {
    fn default() -> Self {
        Version::V2
    }
}

#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub enum RuntimeDevice {
    None,
    Cpu,
    Cuda,
    Mps,
}

impl Default for RuntimeDevice {
    fn default() -> Self {
        RuntimeDevice::None
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct SpeakerConfig {
    pub name: String,
    pub gpt_sovits_path: String,
    pub ref_audio_path: String,
    pub ref_text: String,
    #[serde(default)]
    pub version: Version,
}

pub struct GPTSovitsRuntime {
    gpt_sovits: gpt_sovits_rs::GPTSovits,
    pub speakers: HashMap<String, Version>,
    pub output_wav: Vec<u8>,
}

impl GPTSovitsRuntime {
    pub fn new_by_env() -> anyhow::Result<Self> {
        let config_path =
            std::env::var("GPT_SOVITS_CONFIG_PATH").unwrap_or("config.json".to_string());
        let config = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config)?;
        Self::new(&config)
    }

    pub fn new(config: &Config) -> anyhow::Result<Self> {
        let device = match config.device {
            RuntimeDevice::None => gpt_sovits_rs::Device::cuda_if_available(),
            RuntimeDevice::Cpu => gpt_sovits_rs::Device::Cpu,
            RuntimeDevice::Cuda => gpt_sovits_rs::Device::Cuda(0),
            RuntimeDevice::Mps => gpt_sovits_rs::Device::Mps,
        };
        let mut gpt_sovits_config = gpt_sovits_rs::GPTSovitsConfig::new(
            config.ssl_model_path.clone(),
            config.mini_bart_g2p_path.clone(),
        );

        match (
            config.g2pw_model_path.clone(),
            config.bert_model_path.clone(),
        ) {
            (Some(g2pw_model_path), Some(cn_bert_path)) => {
                gpt_sovits_config = gpt_sovits_config.with_chinese(g2pw_model_path, cn_bert_path);
            }
            _ => {}
        }

        let mut gpt_sovits = gpt_sovits_config.build(device)?;
        let mut speakers = HashMap::with_capacity(config.speaker.len());

        for speaker in &config.speaker {
            let (load_ref_audio, ref_audio_sr) = load_ref_audio(&speaker.ref_audio_path)?;
            match speaker.version {
                Version::V2 => gpt_sovits.create_speaker(
                    &speaker.name,
                    &speaker.gpt_sovits_path,
                    &load_ref_audio,
                    ref_audio_sr as usize,
                    &speaker.ref_text,
                )?,
                Version::V2_1 => gpt_sovits.create_speaker_v2_1(
                    &speaker.name,
                    &speaker.gpt_sovits_path,
                    &load_ref_audio,
                    ref_audio_sr as usize,
                    &speaker.ref_text,
                    None,
                )?,
                Version::V2Pro => gpt_sovits.create_speaker_v2_pro(
                    &speaker.name,
                    &speaker.gpt_sovits_path,
                    &load_ref_audio,
                    ref_audio_sr as usize,
                    &speaker.ref_text,
                    None,
                )?,
                Version::V3 => gpt_sovits.create_speaker_v3(
                    &speaker.name,
                    &speaker.gpt_sovits_path,
                    &load_ref_audio,
                    ref_audio_sr as usize,
                    &speaker.ref_text,
                    None,
                    None,
                )?,
                Version::V4 => gpt_sovits.create_speaker_v4(
                    &speaker.name,
                    &speaker.gpt_sovits_path,
                    &load_ref_audio,
                    ref_audio_sr as usize,
                    &speaker.ref_text,
                    None,
                    None,
                )?,
            }
            speakers.insert(speaker.name.to_string(), speaker.version);
        }

        Ok(Self {
            gpt_sovits,
            speakers,
            output_wav: Vec::new(),
        })
    }

    pub fn infer(&self, speaker: &str, text: &str) -> anyhow::Result<Vec<u8>> {
        let split_chunk_size = if text.chars().all(|c| c.is_ascii()) {
            100 // ASCII text
        } else {
            50 // Non-ASCII text
        };

        let mut audio = self
            .gpt_sovits
            .segment_infer(speaker, text, split_chunk_size)?;
        match self.speakers.get(speaker) {
            Some(Version::V3) => {
                audio = self.gpt_sovits.resample(&audio, 24000, 32000)?;
            }
            Some(Version::V4) => {
                audio = self.gpt_sovits.resample(&audio, 48000, 32000)?;
            }
            _ => {}
        }

        let audio_size = audio.size1()? as usize;
        let mut samples = vec![0f32; audio_size];
        audio.f_copy_data(&mut samples, audio_size)?;
        let header = wav_io::new_header(32000, 16, false, true);
        let out_put = wav_io::write_to_bytes(&header, &samples)
            .map_err(|e| anyhow::anyhow!("write wav error: {e}"))?;

        Ok(out_put)
    }
}

pub fn load_ref_audio(ref_audio_path: &str) -> anyhow::Result<(Vec<f32>, u32)> {
    let ref_audio_file = std::fs::File::open(ref_audio_path)?;
    let (head, samples) =
        wav_io::read_from_file(ref_audio_file).map_err(|e| anyhow::anyhow!("{e}"))?;
    let samples = if head.channels != 1 {
        // to mono
        let mut samples_ = Vec::with_capacity(samples.len() / head.channels as usize);
        for (i, sample) in samples.iter().enumerate() {
            if i % head.channels as usize == 0 {
                samples_.push(*sample);
            }
        }
        samples_
    } else {
        samples
    };
    Ok((samples, head.sample_rate))
}
