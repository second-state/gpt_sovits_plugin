use wasmedge_plugin_sdk::{
    error::{CoreError, CoreExecutionError},
    memory::{Memory, WasmPtr},
    module::{PluginModule, SyncInstanceRef},
    types::{ValType, WasmVal},
};

mod tts;

pub struct HostData(anyhow::Result<tts::GPTSovitsRuntime>);

fn infer<'a>(
    _inst_ref: &'a mut SyncInstanceRef,
    mem: &'a mut Memory,
    data: &'a mut HostData,
    args: Vec<WasmVal>,
) -> Result<Vec<WasmVal>, CoreError> {
    if let Some([WasmVal::I32(text_ptr), WasmVal::I32(text_len)]) = args.get(0..2) {
        match &mut data.0 {
            Ok(runtime) => {
                let text_ptr = *text_ptr as usize;
                let text_len = *text_len as usize;
                if let Some(text) = mem.get_slice(WasmPtr::<u8>::from(text_ptr), text_len) {
                    if let Ok(text) = std::str::from_utf8(text) {
                        match runtime.infer(text) {
                            Ok(wav) => {
                                let len = wav.len();
                                runtime.output_wav = wav;
                                Ok(vec![WasmVal::I32(len as i32)])
                            }
                            Err(e) => {
                                eprintln!("infer error: {:?}", e);
                                Ok(vec![WasmVal::I32(-1)])
                            }
                        }
                    } else {
                        Ok(vec![WasmVal::I32(-1)])
                    }
                } else {
                    Err(CoreError::Execution(CoreExecutionError::MemoryOutOfBounds))
                }
            }
            Err(_) => Ok(vec![WasmVal::I32(-2)]),
        }
    } else {
        Err(CoreError::Execution(CoreExecutionError::FuncTypeMismatch))
    }
}

fn get_output(
    _inst_ref: &mut SyncInstanceRef,
    mem: &mut Memory,
    data: &mut HostData,
    args: Vec<WasmVal>,
) -> Result<Vec<WasmVal>, CoreError> {
    if let Some([WasmVal::I32(data_ptr), WasmVal::I32(len)]) = args.get(0..2) {
        match &mut data.0 {
            Ok(runtime) => {
                let data_ptr = *data_ptr as usize;
                let len = *len as usize;
                if let Some(output_buff) = mem.mut_slice(WasmPtr::from(data_ptr), len) {
                    output_buff.copy_from_slice(&runtime.output_wav);
                    Ok(vec![WasmVal::I32(0)])
                } else {
                    Err(CoreError::Execution(CoreExecutionError::MemoryOutOfBounds))
                }
            }
            Err(_) => Ok(vec![WasmVal::I32(-2)]),
        }
    } else {
        Err(CoreError::Execution(CoreExecutionError::FuncTypeMismatch))
    }
}

fn is_ok<'a>(
    _inst_ref: &'a mut SyncInstanceRef,
    _main_memory: &'a mut Memory,
    data: &'a mut HostData,
    _args: Vec<WasmVal>,
) -> Result<Vec<WasmVal>, CoreError> {
    match data.0 {
        Ok(_) => Ok(vec![WasmVal::I32(1)]),
        Err(_) => Ok(vec![WasmVal::I32(0)]),
    }
}

pub fn create_module() -> PluginModule<HostData> {
    let runtime = tts::GPTSovitsRuntime::new_by_env();

    let mut module = PluginModule::create("gpt_sovits", HostData(runtime)).unwrap();

    module
        .add_func(
            "infer",
            (vec![ValType::I32, ValType::I32], vec![ValType::I32]),
            infer,
        )
        .unwrap();
    module
        .add_func(
            "get_output",
            (vec![ValType::I32, ValType::I32], vec![ValType::I32]),
            get_output,
        )
        .unwrap();
    module
        .add_func("is_ok", (vec![], vec![ValType::I32]), is_ok)
        .unwrap();

    module
}

wasmedge_plugin_sdk::plugin::register_plugin!(
    plugin_name="gpt_sovits",
    plugin_description="a tts plugin based on gpt-sovits",
    version=(0,0,0,1),
    modules=[
        {"gpt_sovits","a tts plugin based on gpt-sovits",create_module}
    ]
);
