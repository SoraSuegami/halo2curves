use core::convert::TryInto;
use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};

use ff::PrimeField;
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use crate::arithmetic::{adc, mac, sbb};

use pasta_curves::arithmetic::{FieldExt, Group, SqrtRatio};

/// This represents an element of $\mathbb{F}_q$ where
///
/// `q = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed`
///
/// is the scalar field of the curve25519 curve.
// The internal representation of this type is four 64-bit unsigned
// integers in little-endian order. `Fq` values are always in
// Montgomery form; i.e., Fq(a) = aR mod q, with R = 2^256.
#[derive(Clone, Copy, Eq)]
pub struct Fq(pub(crate) [u64; 4]);

/// Constant representing the modulus
/// q = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed
const MODULUS: Fq = Fq([
    0x5812631a5cf5d3ed,
    0x14def9dea2f79cd6,
    0x0000000000000000,
    0x1000000000000000
]);

/// The modulus as u32 limbs.
#[cfg(not(target_pointer_width = "64"))]
const MODULUS_LIMBS_32: [u32; 8] = [
    0xd036_4141,
    0xbfd2_5e8c,
    0xaf48_a03b,
    0xbaae_dce6,
    0xffff_fffe,
    0xffff_ffff,
    0xffff_ffff,
    0xffff_ffff,
];

///Constant representing the modulus as static str
const MODULUS_STR: &str = "0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed";

/// INV = -(q^{-1} mod 2^64) mod 2^64
const INV: u64 = 0xd2b51da312547e1b;

/// R = 2^256 mod q
/// 0xffffffffffffffffffffffffffffffec6ef5bf4737dcf70d6ec31748d98951d
const R: Fq = Fq([0xd6ec31748d98951d, 0xc6ef5bf4737dcf70, 0xfffffffffffffffe, 0xfffffffffffffff]);

/// R^2 = 2^512 mod q
/// 0x399411b7c309a3dceec73d217f5be65d00e1ba768859347a40611e3449c0f01
const R2: Fq = Fq([
    0xa40611e3449c0f01,
    0xd00e1ba768859347,
    0xceec73d217f5be65,
    0x399411b7c309a3d,
]);

/// R^3 = 2^768 mod q
/// 0xe530b773599cec78065dc6c04ec5b65278324e6aef7f3ec2a9e49687b83a2db
const R3: Fq = Fq([
    0x2a9e49687b83a2db,
    0x278324e6aef7f3ec,
    0x8065dc6c04ec5b65,
    0xe530b773599cec7,
]);

/// `GENERATOR = 7 mod r` is a generator of the `q - 1` order multiplicative
/// subgroup, or in other words a primitive root of the field.
//const GENERATOR: Fq = Fq::from_raw([0x07, 0x00, 0x00, 0x00]);

/// GENERATOR^t where t * 2^s + 1 = r
/// with t odd. In other words, this
/// is a 2^s root of unity.
/// `0xc1dc060e7a91986df9879a3fbc483a898bdeab680756045992f4b5402b052f2`
/*const ROOT_OF_UNITY: Fq = Fq::from_raw([
    0x992f4b5402b052f2,
    0x98bdeab680756045,
    0xdf9879a3fbc483a8,
    0xc1dc060e7a91986,
]);*/

/// 1 / ROOT_OF_UNITY mod q
/*const ROOT_OF_UNITY_INV: Fq = Fq::from_raw([
    0xb6fb30a0884f0d1c,
    0x77a275910aa413c3,
    0xefc7b0c75b8cbb72,
    0xfd3ae181f12d7096,
]);*/

/// 1 / 2 mod q
/// 0x80000000000000000000000000000000a6f7cef517bce6b2c09318d2e7ae9f7
const TWO_INV: Fq = Fq::from_raw([
    0x2c09318d2e7ae9f7,
    0x0a6f7cef517bce6b,
    0x0000000000000000,
    0x800000000000000,
]);

// 2^((q-1)/4)
// 0x94a7310e07981e77d3d6d60abc1c27a0ef0565342ce83febe8775dfebbe07d4
const SQRT_TWO_POW:Fq = Fq::from_raw([
    0xbe8775dfebbe07d4,
    0x0ef0565342ce83fe,
    0x7d3d6d60abc1c27a,
    0x94a7310e07981e7
]);


const ZETA: Fq = Fq::zero();
const DELTA: Fq = Fq::zero();
const ROOT_OF_UNITY_INV: Fq = Fq::zero();

use crate::{
    field_arithmetic, field_common, field_specific, impl_add_binop_specify_output,
    impl_binops_additive, impl_binops_additive_specify_output, impl_binops_multiplicative,
    impl_binops_multiplicative_mixed, impl_sub_binop_specify_output,
};
impl_binops_additive!(Fq, Fq);
impl_binops_multiplicative!(Fq, Fq);
field_common!(
    Fq,
    MODULUS,
    INV,
    MODULUS_STR,
    TWO_INV,
    ROOT_OF_UNITY_INV,
    DELTA,
    ZETA
);
field_arithmetic!(Fq, dense);

impl Fq {
    pub const fn size() -> usize {
        32
    }
}

impl ff::Field for Fq {
    fn random(mut rng: impl RngCore) -> Self {
        Self::from_u512([
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ])
    }

    fn zero() -> Self {
        Self::zero()
    }

    fn one() -> Self {
        Self::one()
    }

    fn double(&self) -> Self {
        self.double()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        self.square()
    }

    /// Computes the square root of this element, if it exists.
    fn sqrt(&self) -> CtOption<Self> {
        // (q+3)//8
        // 0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7e
        let tmp = self.pow(&[
            0xcb024c634b9eba7e,
            0x029bdf3bd45ef39a,
            0x0000000000000000,
            0x200000000000000,
        ]);
        let mut value = tmp * SQRT_TWO_POW;
        value.conditional_assign(&tmp, tmp.square().ct_eq(self));
        CtOption::new(value, value.square().ct_eq(self))
        //crate::arithmetic::sqrt_tonelli_shanks(self, &<Self as SqrtRatio>::T_MINUS1_OVER2)
    }

    /// Computes the multiplicative inverse of this element,
    /// failing if the element is zero.
    fn invert(&self) -> CtOption<Self> {
        // q-2
        let tmp = self.pow_vartime(&[
            0x5812631a5cf5d3eb,
            0x14def9dea2f79cd6,
            0x0000000000000000,
            0x1000000000000000
        ]);

        CtOption::new(tmp, !self.ct_eq(&Self::zero()))
    }

    fn pow_vartime<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::one();
        let mut found_one = false;
        for e in exp.as_ref().iter().rev() {
            for i in (0..64).rev() {
                if found_one {
                    res = res.square();
                }

                if ((*e >> i) & 1) == 1 {
                    found_one = true;
                    res *= self;
                }
            }
        }
        res
    }
}

impl ff::PrimeField for Fq {
    type Repr = [u8; 32];

    const NUM_BITS: u32 = 256;
    const CAPACITY: u32 = 255;
    const S: u32 = 6;

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        let mut tmp = Fq([0, 0, 0, 0]);

        tmp.0[0] = u64::from_le_bytes(repr[0..8].try_into().unwrap());
        tmp.0[1] = u64::from_le_bytes(repr[8..16].try_into().unwrap());
        tmp.0[2] = u64::from_le_bytes(repr[16..24].try_into().unwrap());
        tmp.0[3] = u64::from_le_bytes(repr[24..32].try_into().unwrap());

        // Try to subtract the modulus
        let (_, borrow) = sbb(tmp.0[0], MODULUS.0[0], 0);
        let (_, borrow) = sbb(tmp.0[1], MODULUS.0[1], borrow);
        let (_, borrow) = sbb(tmp.0[2], MODULUS.0[2], borrow);
        let (_, borrow) = sbb(tmp.0[3], MODULUS.0[3], borrow);

        // If the element is smaller than MODULUS then the
        // subtraction will underflow, producing a borrow value
        // of 0xffff...ffff. Otherwise, it'll be zero.
        let is_some = (borrow as u8) & 1;

        // Convert to Montgomery form by computing
        // (a.R^0 * R^2) / R = a.R
        tmp *= &R2;

        CtOption::new(tmp, Choice::from(is_some))
    }

    fn to_repr(&self) -> Self::Repr {
        // Turn into canonical form by computing
        // (a.R) / R = a
        let tmp = Fq::montgomery_reduce(self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0);

        let mut res = [0; 32];
        res[0..8].copy_from_slice(&tmp.0[0].to_le_bytes());
        res[8..16].copy_from_slice(&tmp.0[1].to_le_bytes());
        res[16..24].copy_from_slice(&tmp.0[2].to_le_bytes());
        res[24..32].copy_from_slice(&tmp.0[3].to_le_bytes());

        res
    }

    fn is_odd(&self) -> Choice {
        Choice::from(self.to_repr()[0] & 1)
    }

    fn multiplicative_generator() -> Self {
        unimplemented!()
    }

    fn root_of_unity() -> Self {
        unimplemented!()
    }
}

impl SqrtRatio for Fq {
    /*const T_MINUS1_OVER2: [u64; 4] = [
        0x777fa4bd19a06c82,
        0xfd755db9cd5e9140,
        0xffffffffffffffff,
        0x01ffffffffffffff,
    ];*/
    const T_MINUS1_OVER2: [u64; 4] = [0, 0, 0, 0];

    fn get_lower_32(&self) -> u32 {
        let tmp = Fq::montgomery_reduce(self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0);
        tmp.0[0] as u32
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ff::Field;
    use rand_core::OsRng;

    #[test]
    fn test_sqrt() {
        // NB: TWO_INV is standing in as a "random" field element
        let v = (Fq::TWO_INV).square().sqrt().unwrap();
        assert!(v == Fq::TWO_INV || (-v) == Fq::TWO_INV);

        for _ in 0..10000 {
            let a = Fq::random(OsRng);
            let mut b = a;
            b = b.square();

            let b = b.sqrt().unwrap();
            let mut negb = b;
            negb = negb.neg();

            assert!(a == b || a == negb);
        }
    }

    /*#[test]
    fn test_root_of_unity() {
        assert_eq!(
            Fq::root_of_unity().pow_vartime(&[1 << Fq::S, 0, 0, 0]),
            Fq::one()
        );
    }

    #[test]
    fn test_inv_root_of_unity() {
        assert_eq!(Fq::ROOT_OF_UNITY_INV, Fq::root_of_unity().invert().unwrap());
    }*/

    #[test]
    fn test_field() {
        crate::tests::field::random_field_tests::<Fq>("curve25519 scalar".to_string());
    }
}
