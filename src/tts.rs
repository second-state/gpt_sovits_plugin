#[derive(Debug, serde::Deserialize)]
pub struct Config {
    #[serde(default)]
    pub bert_model_path: Option<String>,

    #[serde(default)]
    pub g2pw_model_path: Option<String>,

    pub ssl_model_path: String,
    pub mini_bart_g2p_path: String,

    pub speaker: Vec<SpeakerConfig>,
}

#[derive(Debug, serde::Deserialize)]
pub struct SpeakerConfig {
    pub name: String,
    pub gpt_sovits_path: String,
    pub ref_audio_path: String,
    pub ref_text: String,
}

pub struct GPTSovitsRuntime {
    gpt_sovits: gpt_sovits_rs::GPTSovits,
    pub output_wav: Vec<u8>,
}

impl GPTSovitsRuntime {
    pub fn new_by_env() -> anyhow::Result<Self> {
        let config_path = std::env::var("GPT_SOVITS_CONFIG_PATH")?;
        let config = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config)?;
        Self::new(&config)
    }

    pub fn new(config: &Config) -> anyhow::Result<Self> {
        let device = gpt_sovits_rs::Device::cuda_if_available();

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

        for speaker in &config.speaker {
            let (load_ref_audio, ref_audio_sr) = load_ref_audio(&speaker.ref_audio_path)?;
            gpt_sovits.create_speaker(
                &speaker.name,
                &speaker.gpt_sovits_path,
                &load_ref_audio,
                ref_audio_sr as usize,
                &speaker.ref_text,
            )?;
        }

        Ok(Self {
            gpt_sovits,
            output_wav: Vec::new(),
        })
    }

    pub fn infer(&self, speaker: &str, text: &str) -> anyhow::Result<Vec<u8>> {
        let audio = self.gpt_sovits.segment_infer(speaker, text, 50)?;

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
    Ok((samples, head.sample_rate))
}
