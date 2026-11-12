/// NeuroFlow SIMD Vector Engine (VU)
/// - 512-bit vector registers (matching one 16-element FP32 row of a Systolic Tile).
/// - Non-linear element-wise activations (GELU, SiLU, Softmax) and normalizations (RMSNorm).

use std::f32::consts::PI;

pub const VECTOR_BYTES: usize = 64; // 512 bits = 64 bytes = 16 x FP32 elements
pub const VECTOR_LANES_FP32: usize = 16;

#[derive(Debug, Clone, Copy)]
pub struct VectorRegister {
    pub raw: [u8; VECTOR_BYTES],
}

impl Default for VectorRegister {
    fn default() -> Self {
        Self {
            raw: [0u8; VECTOR_BYTES],
        }
    }
}

impl VectorRegister {
    pub fn read_fp32(&self) -> [f32; VECTOR_LANES_FP32] {
        let mut out = [0.0f32; VECTOR_LANES_FP32];
        for (i, item) in out.iter_mut().enumerate() {
            let offset = i * 4;
            let bytes = [
                self.raw[offset],
                self.raw[offset + 1],
                self.raw[offset + 2],
                self.raw[offset + 3],
            ];
            *item = f32::from_le_bytes(bytes);
        }
        out
    }

    pub fn write_fp32(&mut self, data: &[f32; VECTOR_LANES_FP32]) {
        for (i, &val) in data.iter().enumerate() {
            let offset = i * 4;
            self.raw[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        }
    }
}

#[derive(Debug, Default)]
pub struct VectorEngine {
    pub registers: [VectorRegister; 8],
}

impl VectorEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// VGELU: Gaussian Error Linear Unit
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn vgelu(&mut self, vd: usize, vs: usize) -> Result<u64, &'static str> {
        let in_vec = self.registers.get(vs).ok_or("Invalid source vector register")?.read_fp32();
        let mut out_vec = [0.0f32; VECTOR_LANES_FP32];

        let sqrt_2_over_pi = (2.0f32 / PI).sqrt();

        for i in 0..VECTOR_LANES_FP32 {
            let x = in_vec[i];
            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            out_vec[i] = 0.5 * x * (1.0 + inner.tanh());
        }

        self.registers.get_mut(vd).ok_or("Invalid destination vector register")?.write_fp32(&out_vec);
        Ok(1) // 1 clock cycle
    }

    /// VRMSNORM: Root Mean Square Layer Normalization
    /// RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight
    pub fn vrmsnorm(&mut self, vd: usize, vs: usize, v_weight: usize) -> Result<u64, &'static str> {
        let in_vec = self.registers.get(vs).ok_or("Invalid source vector")?.read_fp32();
        let weight_vec = self.registers.get(v_weight).ok_or("Invalid weight vector")?.read_fp32();
        let mut out_vec = [0.0f32; VECTOR_LANES_FP32];

        let mut sum_sq = 0.0f32;
        for &x in &in_vec {
            sum_sq += x * x;
        }
        let rms = ((sum_sq / (VECTOR_LANES_FP32 as f32)) + 1e-5).sqrt();

        for i in 0..VECTOR_LANES_FP32 {
            out_vec[i] = (in_vec[i] / rms) * weight_vec[i];
        }

        self.registers.get_mut(vd).ok_or("Invalid destination vector")?.write_fp32(&out_vec);
        Ok(2) // 2 clock cycles
    }

    /// VSOFTMAX: Numerically stable Vector Softmax
    pub fn vsoftmax(&mut self, vd: usize, vs: usize) -> Result<u64, &'static str> {
        let in_vec = self.registers.get(vs).ok_or("Invalid source vector")?.read_fp32();
        let mut out_vec = [0.0f32; VECTOR_LANES_FP32];

        // Find max
        let mut max_val = f32::NEG_INFINITY;
        for &x in &in_vec {
            if x > max_val { max_val = x; }
        }

        // Exponentiate & sum
        let mut sum = 0.0f32;
        for i in 0..VECTOR_LANES_FP32 {
            out_vec[i] = (in_vec[i] - max_val).exp();
            sum += out_vec[i];
        }

        // Normalize
        for item in out_vec.iter_mut() {
            *item /= sum;
        }

        self.registers.get_mut(vd).ok_or("Invalid destination vector")?.write_fp32(&out_vec);
        Ok(2) // 2 clock cycles
    }

    /// VADD: Element-wise vector addition
    pub fn vadd(&mut self, vd: usize, vs1: usize, vs2: usize) -> Result<u64, &'static str> {
        let a = self.registers.get(vs1).ok_or("Invalid vs1")?.read_fp32();
        let b = self.registers.get(vs2).ok_or("Invalid vs2")?.read_fp32();
        let mut out = [0.0f32; VECTOR_LANES_FP32];

        for i in 0..VECTOR_LANES_FP32 {
            out[i] = a[i] + b[i];
        }

        self.registers.get_mut(vd).ok_or("Invalid vd")?.write_fp32(&out);
        Ok(1)
    }

    /// VMUL: Element-wise vector multiplication
    pub fn vmul(&mut self, vd: usize, vs1: usize, vs2: usize) -> Result<u64, &'static str> {
        let a = self.registers.get(vs1).ok_or("Invalid vs1")?.read_fp32();
        let b = self.registers.get(vs2).ok_or("Invalid vs2")?.read_fp32();
        let mut out = [0.0f32; VECTOR_LANES_FP32];

        for i in 0..VECTOR_LANES_FP32 {
            out[i] = a[i] * b[i];
        }

        self.registers.get_mut(vd).ok_or("Invalid vd")?.write_fp32(&out);
        Ok(1)
    }
}
