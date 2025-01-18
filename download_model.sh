#!/bin/bash

echo "All required tools are installed"
wget --version
if [ $? -ne 0 ]; then
    echo "wget is not installed"
    exit 1
fi

unzip -v
if [ $? -ne 0 ]; then
    echo "unzip is not installed"
    exit 1
fi

echo "Download model files"

echo "Download GPT_Sovits common model files"
wget https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/resource.zip
unzip resource.zip

echo "Download GPT_Sovits g2pw model"
wget https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/g2pw.pt
mv g2pw.pt resource

echo "Download GPT_Sovits mini-bart-g2p model"
wget https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/mini-bart-g2p.pt
mv mini-bart-g2p.pt resource

echo "Download GPT_Sovits voice model"
wget https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/cooper.zip
unzip cooper.zip
wget https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/kelly.zip
unzip kelly.zip

if [ -f "tts-api-server.wasm" ]; then
    echo "Download tts-api-server.wasm"
    wget https://github.com/LlamaEdge/tts-api-server/releases/download/0.2.0/tts-api-server_gpt-sovits.wasm -O tts-api-server.wasm
fi

echo "If plugin is not installed, please install it manually. Goto https://github.com/second-state/gpt_sovits_plugin"

echo "Generate GPT_Sovits config.json"
cat > config.json << EOF
{
    "bert_model_path": "resource/bert_model.pt",
    "g2pw_model_path": "resource/g2pw_model.pt",
    "ssl_model_path": "resource/ssl_model.pt",
    "mini_bart_g2p_path": "resource/mini-bart-g2p.pt",
    "speaker": [
        {
            "name": "cooper",
            "gpt_sovits_path": "cooper/gpt_sovits_model.pt",
            "ref_audio_path": "cooper/ref.wav",
            "ref_text": "The birds are singing again. The sky is clearing."
        },
        {
            "name": "kelly",
            "gpt_sovits_path": "kelly/gpt_sovits_model.pt",
            "ref_audio_path": "kelly/ref.wav",
            "ref_text": "The whole case boils down to the same alleged scheme."
        }
    ]
}
EOF

echo "# run tts-api-server"
echo "wasmedge --dir .:. tts-api-server.wasm --model-name gpt_sovits --model gpt_sovits --config NA --espeak-ng-dir NA"
