#[derive(Debug, serde::Deserialize)]
pub struct Config {
    #[serde(default)]
    pub bert_model_path: Option<String>,
    #[serde(default)]
    pub tokenizer_path: Option<String>,

    pub ssl_model_path: String,
    pub gpt_sovits_path: String,

    pub ref_audio_path: String,
    pub ref_text: String,
}

pub struct GPTSovitsRuntime {
    ref_audio_simple: Vec<f32>,
    ref_audio_sr: u32,
    ref_text: String,
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
        Self::new_(config)
    }

    fn new_(config: &Config) -> anyhow::Result<Self> {
        let device = gpt_sovits_rs::Device::cuda_if_available();

        let (load_ref_audio, ref_audio_sr) = load_ref_audio(&config.ref_audio_path)?;
        let ref_text = config.ref_text.clone();

        let mut gpt_sovits_config = gpt_sovits_rs::GPTSovitsConfig::new(
            config.gpt_sovits_path.clone(),
            config.ssl_model_path.clone(),
        );

        match (
            config.bert_model_path.clone(),
            config.tokenizer_path.clone(),
        ) {
            (Some(cn_bert_path), Some(tokenizer_path)) => {
                gpt_sovits_config =
                    gpt_sovits_config.with_cn_bert_path(cn_bert_path, tokenizer_path);
            }
            _ => {}
        }

        let gpt_sovits = gpt_sovits_config.build(device)?;

        Ok(Self {
            gpt_sovits,
            ref_audio_simple: load_ref_audio,
            ref_audio_sr,
            ref_text,
            output_wav: Vec::new(),
        })
    }

    pub fn infer(&self, text: &str) -> anyhow::Result<Vec<u8>> {
        let audio = self.gpt_sovits.infer(
            &self.ref_audio_simple,
            self.ref_audio_sr as usize,
            &self.ref_text,
            text,
        )?;

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
