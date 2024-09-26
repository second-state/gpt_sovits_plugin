mod gpt_sovits {
    mod ffi {
        #[link(wasm_import_module = "gpt_sovits")]
        extern "C" {
            pub fn infer(text_ptr: *const u8, text_len: usize) -> i32;
            pub fn get_output(output_buf: *mut u8, output_len: usize) -> i32;
        }
    }

    pub fn infer(text: &str) -> Result<Vec<u8>, &'static str> {
        unsafe {
            let i = ffi::infer(text.as_ptr(), text.len());
            match i {
                -1 => Err("infer error"),
                -2 => Err("runtime error"),
                _ => {
                    let mut buf = vec![0u8; i as usize];
                    let o = ffi::get_output(buf.as_mut_ptr(), i as usize);
                    match o {
                        -2 => Err("runtime error"),
                        _ => Ok(buf),
                    }
                }
            }
        }
    }
}
fn main() {
    let text = std::env::args().nth(1).unwrap();
    println!("infer {text} -> out.wav");
    match gpt_sovits::infer(&text) {
        Ok(buf) => {
            std::fs::write("out.wav", buf).unwrap();
            println!("done");
        }
        Err(e) => eprintln!("error: {e}"),
    }
}
