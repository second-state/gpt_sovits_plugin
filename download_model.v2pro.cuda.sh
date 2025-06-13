#!/bin/bash

echo "Make sure that all required tools are installed"
unzip -v
if [ $? -ne 0 ]; then
    echo "unzip is not installed"
    exit 1
fi

echo "Download model files"

echo "Download GPT_Sovits common model files"
curl -LO https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/resource.zip
unzip resource.zip
rm resource.zip

echo "Download the GPT_Sovits g2pw model"
curl -LO https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/g2pw.pt

echo "Download the GPT_Sovits mini-bart-g2p model"
curl -LO https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/mini-bart-g2p.pt

echo "Download the GPT_Sovits V2Pro"
curl -LO https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/gpt_sovits_v2pro.cuda.pt

echo "Download the GPT_Sovits voice actors model"
curl -LO https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/cooper.zip
unzip cooper.zip
rm cooper.zip
curl -LO https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/kelly.zip
unzip kelly.zip
rm kelly.zip

echo "Download tts-api-server.wasm"
curl -LO https://github.com/LlamaEdge/tts-api-server/releases/download/0.2.0/tts-api-server_gpt-sovits.wasm

echo "Generate GPT_Sovits config.json"
cat > config.json << EOF
{
    "bert_model_path": "bert_model.pt",
    "g2pw_model_path": "g2pw.pt",
    "ssl_model_path": "ssl_model.pt",
    "mini_bart_g2p_path": "mini-bart-g2p.pt",
    "device":"Cuda",
    "speaker": [
        {
            "name": "cooper",
            "gpt_sovits_path": "gpt_sovits_v2pro.cuda.pt",
            "version":"V2Pro",
            "ref_audio_path": "cooper/ref.wav",
            "ref_text": "The birds are singing again. The sky is clearing."
        },
        {
            "name": "kelly",
            "gpt_sovits_path": "gpt_sovits_v2pro.cuda.pt",
            "version":"V2Pro",
            "ref_audio_path": "kelly/ref.wav",
            "ref_text": "The whole case boils down to the same alleged scheme."
        }
    ]
}
EOF

echo "If the WasmEdge GPT-SoVITS plugin is not installed, please install it manually. Goto https://github.com/second-state/gpt_sovits_plugin"
echo "# run tts-api-server"
echo "wasmedge --dir .:. tts-api-server_gpt-sovits.wasm --model-name gpt_sovits --model gpt_sovits --config NA --espeak-ng-dir NA"
