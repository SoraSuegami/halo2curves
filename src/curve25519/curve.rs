use crate::curve25519::Fp;
use crate::curve25519::Fq;
use crate::{Coordinates, CurveAffine, CurveAffineExt, CurveExt, Group};
use core::cmp;
use core::fmt::Debug;
use core::iter::Sum;
use core::ops::{Add, Mul, Neg, Sub};
use ff::{Field, PrimeField};
use group::Curve;
use group::{prime::PrimeCurveAffine, Group as _, GroupEncoding};

use pasta_curves::arithmetic::FieldExt;
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

pub trait MontgomeryCurve {
    type WeiForm;
    type Coordinate;

    fn to_wei_form(&self) -> Self::WeiForm;
    fn coordinate(&self) -> Self::Coordinate;
}

/// Curve25519
#[derive(Copy, Clone, Debug)]
pub struct Curve25519(Wei25519);

#[derive(Copy, Clone)]
pub struct Curve25519Affine(Wei25519Affine);

impl MontgomeryCurve for Curve25519 {
    type WeiForm = Wei25519;
    type Coordinate = (Fp, Fp, Fp);

    fn to_wei_form(&self) -> Self::WeiForm {
        self.0
    }

    fn coordinate(&self) -> Self::Coordinate {
        let wei_affine = self.0.to_affine();
        let is_identity = wei_affine.is_identity();
        let curve_affine = Curve25519Affine(wei_affine);
        let (x, y) = curve_affine.coordinate();
        let z = Fp::conditional_select(&Fp::one(), &Fp::zero(), is_identity);
        (x, y, z)
    }
}

impl From<Wei25519> for Curve25519 {
    fn from(wei: Wei25519) -> Self {
        Self(wei)
    }
}

impl Curve25519 {
    pub fn new(x: Fp, y: Fp, z: Fp) -> Self {
        let zinv = z.invert().unwrap_or(Fp::zero());
        let zinv2 = zinv.square();
        let x = x * zinv2;
        let zinv3 = zinv2 * zinv;
        let y = y * zinv3;
        let wei_affine = Curve25519Affine::new(x, y).0;
        let wei_curve = wei_affine.to_curve();
        Self(wei_curve)
    }
}

impl MontgomeryCurve for Curve25519Affine {
    type WeiForm = Wei25519Affine;
    type Coordinate = (Fp, Fp);

    fn to_wei_form(&self) -> Self::WeiForm {
        self.0
    }

    fn coordinate(&self) -> Self::Coordinate {
        // [TODO] Precompute b_inv.
        let b_inv = Wei25519Affine::curve_constant_b().invert().unwrap();
        let a = Wei25519Affine::curve_constant_a();
        let wei = self.0;
        let x = wei.x * b_inv - a * b_inv;
        let y = wei.y * b_inv;
        (x, y)
    }
}

impl From<Wei25519Affine> for Curve25519Affine {
    fn from(wei: Wei25519Affine) -> Self {
        Self(wei)
    }
}

impl Curve25519Affine {
    /// A: 0x76d06 [https://neuromancer.sk/std/other/Curve25519]
    /// B: 0x01 [https://neuromancer.sk/std/other/Curve25519]
    pub const A: Fp = Fp::from_raw([0x76d06, 0, 0, 0]);
    pub const B: Fp = Fp::from_raw([1, 0, 0, 0]);

    pub fn new(x: Fp, y: Fp) -> Self {
        // [TODO] Precompute b_inv and three_inv.
        let b_inv = Self::B.invert().unwrap();
        let three_inv = Fp::from_u128(3).invert().unwrap();
        let u = x * b_inv + Self::A * b_inv * three_inv;
        let v = y * b_inv;
        let wei_affine = Wei25519Affine { x: u, y: v };
        Self(wei_affine)
    }
}

/// Wei25519
impl Wei25519 {
    const COFACTOR: Fq = Fq::from_raw([8, 0, 0, 0]);
    fn endomorphism_base(&self) -> Self {
        unimplemented!();
    }
}

impl group::cofactor::CofactorGroup for Wei25519 {
    type Subgroup = Wei25519;

    fn clear_cofactor(&self) -> Self {
        self * Self::COFACTOR
    }

    fn into_subgroup(self) -> CtOption<Self::Subgroup> {
        CtOption::new(self, 1.into())
    }

    fn is_torsion_free(&self) -> Choice {
        1.into()
    }
}

const SECP_GENERATOR_X: Fp = Fp::from_raw([
    0xaaaaaaaaaaad245a,
    0xaaaaaaaaaaaaaaaa,
    0xaaaaaaaaaaaaaaaa,
    0x2aaaaaaaaaaaaaaa,
]);
const SECP_GENERATOR_Y: Fp = Fp::from_raw([
    0x29e9c5a27eced3d9,
    0x923d4d7e6d7c61b2,
    0xe01edd2c7748d14c,
    0x20ae19a1b8a086b4,
]);
const SECP_A: Fp = Fp::from_raw([
    0xaaaaaa984914a144,
    0xaaaaaaaaaaaaaaaa,
    0xaaaaaaaaaaaaaaaa,
    0x2aaaaaaaaaaaaaaa,
]);
const SECP_B: Fp = Fp::from_raw([
    0x260b5e9c7710c864,
    0xed097b425ed097b4,
    0x097b425ed097b425,
    0x7b425ed097b425ed,
]);

use crate::{
    batch_add, impl_add_binop_specify_output, impl_binops_additive,
    impl_binops_additive_specify_output, impl_binops_multiplicative,
    impl_binops_multiplicative_mixed, impl_sub_binop_specify_output, new_a_b_curve_impl,
};

new_a_b_curve_impl!(
    (pub),
    Wei25519,
    Wei25519Affine,
    Wei25519Compressed,
    Fp,
    Fq,
    (SECP_GENERATOR_X,SECP_GENERATOR_Y),
    SECP_A,
    SECP_B,
    Fp::TWO_INV,
    "wei25519",
);

/*impl CurveAffineExt for Secp256k1Affine {
    batch_add!();
}*/

#[test]
fn test_curve() {
    crate::tests::curve::curve_tests::<Wei25519>();
}

#[test]
fn ecdsa_example() {
    use crate::group::Curve;
    use crate::{CurveAffine, FieldExt};
    use rand_core::OsRng;

    fn mod_n(x: Fp) -> Fq {
        let mut x_repr = [0u8; 32];
        x_repr.copy_from_slice(x.to_repr().as_ref());
        let mut x_bytes = [0u8; 64];
        x_bytes[..32].copy_from_slice(&x_repr[..]);
        Fq::from_bytes_wide(&x_bytes)
    }

    let g = Wei25519::generator();

    for _ in 0..1000 {
        // Generate a key pair
        let sk = Fq::random(OsRng);
        let pk = (g * sk).to_affine();

        // Generate a valid signature
        // Suppose `m_hash` is the message hash
        let msg_hash = Fq::random(OsRng);

        let (r, s) = {
            // Draw arandomness
            let k = Fq::random(OsRng);
            let k_inv = k.invert().unwrap();

            // Calculate `r`
            let r_point = (g * k).to_affine().coordinates().unwrap();
            let x = r_point.x();
            let r = mod_n(*x);

            // Calculate `s`
            let s = k_inv * (msg_hash + (r * sk));

            (r, s)
        };

        {
            // Verify
            let s_inv = s.invert().unwrap();
            let u_1 = msg_hash * s_inv;
            let u_2 = r * s_inv;

            let v_1 = g * u_1;
            let v_2 = pk * u_2;

            let r_point = (v_1 + v_2).to_affine().coordinates().unwrap();
            let x_candidate = r_point.x();
            let r_candidate = mod_n(*x_candidate);

            assert_eq!(r, r_candidate);
        }
    }
}
