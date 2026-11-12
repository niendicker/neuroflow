/// AI Numeric Datatypes and conversions for NeuroFlow.
/// Supports FP32, BF16 (Bfloat16), FP8 (E4M3, E5M2), and INT8/INT32.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataType {
    Fp8E4M3 = 0,
    Fp8E5M2 = 1,
    Bf16 = 2,
    Int8 = 3,
    Fp32 = 4,
    Int32 = 5,
}

impl DataType {
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(DataType::Fp8E4M3),
            1 => Some(DataType::Fp8E5M2),
            2 => Some(DataType::Bf16),
            3 => Some(DataType::Int8),
            4 => Some(DataType::Fp32),
            5 => Some(DataType::Int32),
            _ => None,
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        match self {
            DataType::Fp8E4M3 | DataType::Fp8E5M2 | DataType::Int8 => 1,
            DataType::Bf16 => 2,
            DataType::Fp32 | DataType::Int32 => 4,
        }
    }
}

/// 16-bit Brain Floating Point (BF16):
/// 1 sign bit, 8 exponent bits, 7 mantissa bits.
/// Matches the dynamic range of FP32 with reduced precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Bf16(pub u16);

impl Bf16 {
    pub const ZERO: Self = Bf16(0);

    #[inline]
    pub fn from_f32(val: f32) -> Self {
        let bits = val.to_bits();
        // Truncate bottom 16 bits with round-to-nearest-even
        let lsb = (bits >> 16) & 1;
        let rounding_bias = 0x7FFF + lsb;
        let rounded = bits.wrapping_add(rounding_bias);
        Bf16((rounded >> 16) as u16)
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        let bits = (self.0 as u32) << 16;
        f32::from_bits(bits)
    }
}

impl From<f32> for Bf16 {
    fn from(val: f32) -> Self {
        Bf16::from_f32(val)
    }
}

impl From<Bf16> for f32 {
    fn from(val: Bf16) -> Self {
        val.to_f32()
    }
}

/// 8-bit Float E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits (bias = 7).
/// Standard format for forward pass GEMMs in modern Transformer architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fp8E4M3(pub u8);

impl Fp8E4M3 {
    pub const ZERO: Self = Fp8E4M3(0);

    pub fn to_f32(self) -> f32 {
        let sign = (self.0 >> 7) & 1;
        let exp = (self.0 >> 3) & 0x0F;
        let mant = self.0 & 0x07;

        if exp == 0 {
            // Subnormal or zero
            let val = (mant as f32) / 8.0 * (2.0f32).powi(-6);
            if sign == 1 { -val } else { val }
        } else if exp == 0x0F && mant == 0x07 {
            // NaN in E4M3 (0x7F / 0xFF)
            f32::NAN
        } else {
            // Normalized
            let val = (1.0 + (mant as f32) / 8.0) * (2.0f32).powi((exp as i32) - 7);
            if sign == 1 { -val } else { val }
        }
    }

    pub fn from_f32(val: f32) -> Self {
        if val.is_nan() {
            return Fp8E4M3(0x7F);
        }
        let sign = if val.is_sign_negative() { 1u8 } else { 0u8 };
        let abs_val = val.abs();

        if abs_val == 0.0 {
            return Fp8E4M3(sign << 7);
        }

        // Clamp to max representable number in E4M3 (448.0)
        let clamped = abs_val.min(448.0);
        let bits = clamped.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32;
        let f32_mant = bits & 0x7FFFFF;

        let e4m3_exp = f32_exp - 127 + 7;

        if e4m3_exp <= 0 {
            // Subnormal
            let shift = 1 - e4m3_exp;
            let full_mant = (1 << 23) | f32_mant;
            let mant = (full_mant >> (20 + shift)) as u8;
            Fp8E4M3((sign << 7) | (mant & 0x07))
        } else if e4m3_exp >= 15 {
            // Saturate to max value
            Fp8E4M3((sign << 7) | 0x7E)
        } else {
            // Normalized
            let mant = (f32_mant >> 20) as u8;
            Fp8E4M3((sign << 7) | ((e4m3_exp as u8) << 3) | (mant & 0x07))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_conversion() {
        let original = 3.141592f32;
        let bf = Bf16::from_f32(original);
        let recovered = bf.to_f32();
        assert!((original - recovered).abs() < 0.02);
    }

    #[test]
    fn test_fp8_e4m3_conversion() {
        let original = 1.5f32;
        let fp8 = Fp8E4M3::from_f32(original);
        let recovered = fp8.to_f32();
        assert_eq!(recovered, 1.5f32);
    }
}
