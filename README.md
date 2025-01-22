# Run an OpenAI compatible TTS service

## Install libtorch dependencies

Regular Linux CPU

```
# download libtorch
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip

unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cu124.zip

# Add to ~/.zprofile or ~/.bash_profile
export LD_LIBRARY_PATH=$(pwd)/libtorch/lib:$LD_LIBRARY_PATH
export LIBTORCH=$(pwd)/libtorch 
```

Or, on the Mac

```
# download libtorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.4.0.zip

unzip libtorch-macos-arm64-2.4.0.zip

# Add to ~/.zprofile or ~/.bash_profile
export DYLD_LIBRARY_PATH=$(pwd)/libtorch/lib:$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/libtorch/lib:$LD_LIBRARY_PATH
export LIBTORCH=$(pwd)/libtorch 
```

## Install LlamaEdge

```
# Install wasmedge
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s
```

Download the plugin for your platform

```
# MacOS
curl -LO https://github.com/second-state/gpt_sovits_plugin/releases/download/0.0.3/libgpt_sovits_plugin.macos.cpu.dylib

# Linux + CUDA 12
curl -LO https://github.com/second-state/gpt_sovits_plugin/releases/download/0.0.3/libgpt_sovits_plugin.ubuntu-2004.cuda12.so

# Linux CPU
curl -LO https://github.com/second-state/gpt_sovits_plugin/releases/download/0.0.3/libgpt_sovits_plugin.ubuntu-2004.cpu.so
```

Copy the plugin to the library path

```
# Copy the dylib for the Mac
cp target/release/libgpt_sovits_plugin.dylib ~/.wasmedge/plugin

# Or, copy so file into the plugin directory
cp target/release/libgpt_sovits_plugin.so ~/.wasmedge/plugin
```

Optional: You can build the plugin for your own platform.

```
# build plugin
git clone https://github.com/second-state/gpt_sovits_plugin.git
cd gpt_sovits_plugin
cargo build —release
```


## Download  model files for the tts api server

Use the script

```
curl -sSf https://raw.githubusercontent.com/second-state/gpt_sovits_plugin/main/download_model.sh   | bash -s
```

## Start the tts api server

```
wasmedge --dir .:. tts-api-server.wasm \
  --model-name gpt_sovits \
  --model gpt_sovits \
  --config NA \
  --espeak-ng-dir NA \
  --socket-addr 0.0.0.0:8080
```

If you are running on a CPU, you will also need the following environment variables.

```
export ONEDNN_PRIMITIVE_CACHE_CAPACITY=0 
export LRU_CACHE_CAPACITY=1
```

## Access the server

```
curl --location 'http://localhost:8080/v1/audio/speech' --output audio.wav \
  --header 'Content-Type: application/json' \
  --data '{
    "speaker": "cooper",
    "input": "When a big event happens, people turn on to CNN, not only because they know there will be people there covering an event on the ground, but because they know we''re going to cover it in a way that''s non-partisan, that''s not left or right."
  }'
```

## Bonus: use your own dictionary

You can put the definition files in the server directory. Use the following environment variable to tell wasmedge that it is in the same directory where you start wasmedge

```
export GPT_SOVITS_DICT_PATH=. 
```

### For English

Create the following file `en_word_dict.json`

```
{
    "github": [
        "G",
        "IH1",
        "T",
        "-",
        "HH",
        "AH2",
        "B"
    ]
}
```

### For Chinese

Create the following file `zh_word_dict.json`

```
{
    "听不见": [
        "ting1",
        "bu5",
        "jian4"
    ],
    "一声": [
        "yi4",
        "sheng1"
    ],
    "一起": [
        "yi4",
        "qi3"
    ],
    "铠甲": [
        "kai2",
        "jia3"
    ],
    "哪个": [
        "na3",
        "ge5"
    ],
    "耳朵": [
        "er3",
        "duo1"
    ],
    "将士": [
        "jiang4",
        "shi4"
    ],
    "要求": [
        "yao1",
        "qiu2"
    ],
    "动弹": [
        "dong4",
        "tan2"
    ],
    "两只": [
        "liang3",
        "zhi1"
    ],
    "回来": [
        "hui2",
        "lai5"
    ],
    "弟弟": [
        "di4",
        "di5"
    ],
    "姐姐": [
        "jie3",
        "jie5"
    ],
    "父亲": [
        "fu4",
        "qin5"
    ],
    "惦记": [
        "dian4",
        "ji4"
    ]
}
```

