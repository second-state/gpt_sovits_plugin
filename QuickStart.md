Step 1: Install Torch 2.4.0

Step 2: Build plugin & Wasm

```bash
cd gpt_sovits_plugin

export LIBTORCH=/home/csh/torch_2.4/libtorch
cargo build --release

cd gpt_sovits_wasm
cargo build --target wasm32-wasi --release

cd ..
mkdir plugins
cp target/release/libgpt_sovits_plugin.so plugins

export WASMEDGE_PLUGIN_PATH=plugins
export LD_LIBRARY_PATH=~/torch_2.4/libtorch/lib
```

Step 3: Download resource.zip and unzip

Step 4: Run
```bash
# run
export GPT_SOVITS_CONFIG_PATH='./config.json' 
wasmedge --dir=".:." target/wasm32-wasi/debug/gpt_sovits_wasm.wasm '这是一段语音示例'
```
