#![allow(clippy::new_without_default)]

use alloc::string::String;
use js_sys::Array;

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

use crate::vae::Model;
use crate::state::{Backend, build_and_load_model};

use burn::tensor::Tensor;

#[cfg_attr(target_family = "wasm", wasm_bindgen(start))]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Mnist structure that corresponds to JavaScript class.
/// See:[exporting-rust-struct](https://rustwasm.github.io/wasm-bindgen/contributing/design/exporting-rust-struct.html)
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct Mnist {
    model: Option<Model<Backend>>,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl Mnist {
    /// Constructor called by JavaScripts with the new keyword.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self { model: None }
    }

    /// Returns the inference results.
    ///
    /// This method is called from JavaScript via generated wrapper code by wasm-bindgen.
    ///
    /// # Arguments
    ///
    /// * `input` - A f32 slice of input 28x28 image
    ///
    /// See bindgen support types for passing and returning arrays:
    /// * [number-slices](https://rustwasm.github.io/wasm-bindgen/reference/types/number-slices.html)
    /// * [boxed-number-slices](https://rustwasm.github.io/wasm-bindgen/reference/types/boxed-number-slices.html)
    ///
    pub async fn inference(&mut self, input: &[f32]) -> Result<Array, String> {
        if self.model.is_none() {
            self.model = Some(build_and_load_model().await);
        }

        let model = self.model.as_ref().unwrap();

        let device = Default::default();
        // Reshape from the 1D array to 3d tensor [batch, height, width]
        let input = Tensor::<Backend, 1>::from_floats(input, &device).reshape([1, 1, 256, 256]);

        // Run the tensor input through the model
        let (output_4d, _, _, _) = model.forward(input);

        let output = output_4d.into_data_async().await;

        let array = Array::new();
        for value in output.iter::<f32>() {
            array.push(&value.into());
        }

        Ok(array)
    }
}
