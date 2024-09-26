use std::collections::HashMap;

use tch::{IValue, Tensor};
use tokenizers::Tokenizer;

#[derive(Debug, serde::Deserialize)]
pub struct Config {
    pub bert_model_path: String,
    pub ssl_model_path: String,
    pub gpt_sovits_path: String,
    pub tokenizer_path: String,
    pub symbols_path: String,

    pub ref_audio_path: String,
    pub ref_text: String,
}

fn split_zh_ph(ph: &str) -> (&str, &str) {
    if ph.starts_with("zh") || ph.starts_with("ch") || ph.starts_with("sh") {
        ph.split_at(2)
    } else {
        ph.split_at(1)
    }
}

fn phones(symbels: &HashMap<String, i64>, text: &str, device: tch::Device) -> (Tensor, Tensor) {
    use pinyin::ToPinyin;

    let mut word2ph = Vec::new();

    let mut sequence = Vec::new();

    for c in text.chars() {
        if let Some(p) = c.to_pinyin() {
            let (y, s) = split_zh_ph(p.with_tone_num_end());
            sequence.push(symbels.get(y).map(|id| *id).unwrap_or(0));
            sequence.push(symbels.get(s).map(|id| *id).unwrap_or(0));
            word2ph.push(2);
        } else {
            let s = c.to_string();
            sequence.push(symbels.get(&s).map(|id| *id).unwrap_or(0));
            word2ph.push(1);
        }
    }

    let word2ph = Tensor::from_slice(&word2ph);
    let t = Tensor::from_slice(&sequence).to_device(device);
    (t.unsqueeze(0), word2ph)
}

pub struct GPTSovitsRuntime {
    tokenizer: Tokenizer,
    symbels: HashMap<String, i64>,
    device: tch::Device,
    bert: tch::CModule,
    gpt_sovits: tch::CModule,
    ref_data: RefData,
    pub output_wav: Vec<u8>,
}

pub struct RefData {
    ssl_content: Tensor,
    ref_audio_sr: Tensor,

    ref_seq: Tensor,
    ref_bert: Tensor,
}

fn resample(
    ssl: &tch::CModule,
    ref_audio: &IValue,
    src_sr: u32,
    target_sr: u32,
) -> anyhow::Result<Tensor> {
    let ref_audio_new = ssl.method_is(
        "resample",
        &[
            ref_audio,
            &IValue::Int(src_sr as i64),
            &IValue::Int(target_sr as i64),
        ],
    )?;
    let ref_audio_16k = match ref_audio_new {
        IValue::Tensor(ref_audio_16k) => ref_audio_16k,
        _ => unreachable!(),
    };
    Ok(ref_audio_16k)
}

fn get_ref_data(
    ref_text: &str,
    ref_audio: Tensor,
    ref_audio_sr: u32,
    tokenizer: &Tokenizer,
    symbels: &HashMap<String, i64>,
    ssl: &tch::CModule,
    bert: &tch::CModule,
    device: tch::Device,
) -> anyhow::Result<RefData> {
    let ref_audio = IValue::Tensor(ref_audio);

    let (ref_text_ids, ref_text_mask, ref_text_token_type_ids) =
        encode_text(ref_text, &tokenizer, device)?;
    let (ref_seq, ref_text_word2ph) = phones(&symbels, ref_text, device);

    let ref_bert = bert.forward_ts(&[
        ref_text_ids,
        ref_text_mask,
        ref_text_token_type_ids,
        ref_text_word2ph,
    ])?;

    let ref_audio_16k = resample(ssl, &ref_audio, ref_audio_sr, 16000)?;
    let ref_audio_sr = resample(ssl, &ref_audio, ref_audio_sr, 32000)?;

    let ssl_content = ssl.forward_ts(&[&ref_audio_16k])?;

    Ok(RefData {
        ssl_content,
        ref_audio_sr,
        ref_seq,
        ref_bert,
    })
}

impl GPTSovitsRuntime {
    pub fn new_by_env() -> anyhow::Result<Self> {
        let config_path = std::env::var("GPT_SOVITS_CONFIG_PATH")?;
        let config = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config)?;
        Self::new(&config)
    }

    pub fn new(config: &Config) -> anyhow::Result<Self> {
        tch::no_grad(|| Self::new_(config))
    }

    fn new_(config: &Config) -> anyhow::Result<Self> {
        let tokenizer =
            Tokenizer::from_file(&config.tokenizer_path).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let symbels = load_symbel(&config.symbols_path)?;
        let device = tch::Device::cuda_if_available();
        let (bert, ssl, gpt_sovits) = load_model(config, device)?;

        let (load_ref_audio, ref_audio_sr) = load_ref_audio(&config.ref_audio_path, device)?;

        let ref_data = get_ref_data(
            &config.ref_text,
            load_ref_audio,
            ref_audio_sr,
            &tokenizer,
            &symbels,
            &ssl,
            &bert,
            device,
        )?;

        Ok(Self {
            tokenizer,
            symbels,
            device,
            bert,
            gpt_sovits,
            ref_data,
            output_wav: Vec::new(),
        })
    }

    pub fn infer(&self, text: &str) -> anyhow::Result<Vec<u8>> {
        let (text_ids, text_mask, text_token_type_ids) =
            encode_text(text, &self.tokenizer, self.device)?;

        let (text_seq, text_word2ph) = phones(&self.symbels, text, self.device);

        let text_bert =
            self.bert
                .forward_ts(&[text_ids, text_mask, text_token_type_ids, text_word2ph])?;

        let audio = self.gpt_sovits.forward_ts(&[
            &self.ref_data.ssl_content,
            &self.ref_data.ref_audio_sr,
            &self.ref_data.ref_seq,
            &text_seq,
            &self.ref_data.ref_bert,
            &text_bert,
        ])?;

        let audio_size = audio.size1()? as usize;
        let mut samples = vec![0f32; audio_size];
        audio.f_copy_data(&mut samples, audio_size)?;
        let header = wav_io::new_header(32000, 16, false, true);
        let out_put = wav_io::write_to_bytes(&header, &samples)
            .map_err(|e| anyhow::anyhow!("write wav error: {e}"))?;

        Ok(out_put)
    }
}

fn load_symbel(path: &str) -> anyhow::Result<HashMap<String, i64>> {
    let f = std::fs::File::open(path)?;
    Ok(serde_json::from_reader(f)?)
}

fn encode_text(
    text: &str,
    tokenizer: &Tokenizer,
    device: tch::Device,
) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let ids = encoding
        .get_ids()
        .into_iter()
        .map(|x| (*x) as i64)
        .collect::<Vec<i64>>();
    let text_ids = Tensor::from_slice(&ids);
    let text_ids = text_ids.unsqueeze(0).to_device(device);

    let mask = encoding
        .get_attention_mask()
        .into_iter()
        .map(|x| (*x) as i64)
        .collect::<Vec<i64>>();
    let text_mask = Tensor::from_slice(&mask);
    let text_mask = text_mask.unsqueeze(0).to_device(device);

    let token_type_ids = encoding
        .get_type_ids()
        .into_iter()
        .map(|x| (*x) as i64)
        .collect::<Vec<i64>>();
    let text_token_type_ids = Tensor::from_slice(&token_type_ids);
    let text_token_type_ids = text_token_type_ids.unsqueeze(0).to_device(device);
    Ok((text_ids, text_mask, text_token_type_ids))
}

fn load_model(
    config: &Config,
    device: tch::Device,
) -> anyhow::Result<(tch::CModule, tch::CModule, tch::CModule)> {
    let mut bert = tch::CModule::load_on_device(&config.bert_model_path, device)
        .map_err(|e| anyhow::anyhow!("load bert_model error: {}", e))?;
    bert.set_eval();

    let mut ssl = tch::CModule::load_on_device(&config.ssl_model_path, device)
        .map_err(|e| anyhow::anyhow!("load ssl_model error: {}", e))?;
    ssl.set_eval();

    let mut gpt_sovits = tch::CModule::load_on_device(&config.gpt_sovits_path, device)
        .map_err(|e| anyhow::anyhow!("load vits_model error: {}", e))?;
    gpt_sovits.set_eval();

    Ok((bert, ssl, gpt_sovits))
}

pub fn load_ref_audio(ref_audio_path: &str, device: tch::Device) -> anyhow::Result<(Tensor, u32)> {
    let ref_audio_file = std::fs::File::open(ref_audio_path)?;
    let (head, samples) =
        wav_io::read_from_file(ref_audio_file).map_err(|e| anyhow::anyhow!("{e}"))?;
    let t = Tensor::from_slice(&samples).to_device(device);
    Ok((t.unsqueeze(0), head.sample_rate))
}
