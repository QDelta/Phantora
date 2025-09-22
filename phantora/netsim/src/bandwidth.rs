use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum BandwidthUnit {
    Kbps = 1000,
    Mbps = 1_000_000,
    Gbps = 1_000_000_000,
}

pub const MAX: Bandwidth = Bandwidth {
    val: u64::MAX,
    unit: BandwidthUnit::Gbps,
};
pub const MIN: Bandwidth = Bandwidth {
    val: u64::MIN,
    unit: BandwidthUnit::Gbps,
};

impl std::fmt::Display for BandwidthUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use BandwidthUnit::*;
        match self {
            Kbps => write!(f, "Kb/s"),
            Mbps => write!(f, "Mb/s"),
            Gbps => write!(f, "Gb/s"),
        }
    }
}

pub trait BandwidthTrait {
    fn kbps(self) -> Bandwidth;
    fn mbps(self) -> Bandwidth;
    fn gbps(self) -> Bandwidth;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bandwidth {
    val: u64,
    unit: BandwidthUnit,
}

impl Bandwidth {
    #[inline]
    pub fn val(&self) -> u64 {
        self.val
    }
}

impl BandwidthTrait for Bandwidth {
    fn kbps(self) -> Bandwidth {
        Bandwidth {
            val: self.val(),
            unit: BandwidthUnit::Kbps,
        }
    }
    fn mbps(self) -> Bandwidth {
        Bandwidth {
            val: self.val(),
            unit: BandwidthUnit::Mbps,
        }
    }
    fn gbps(self) -> Bandwidth {
        Bandwidth {
            val: self.val(),
            unit: BandwidthUnit::Gbps,
        }
    }
}

impl std::fmt::Display for Bandwidth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {}",
            self.val as f64 / self.unit as u64 as f64,
            self.unit
        )
    }
}

impl std::iter::Sum for Bandwidth {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut s = 0.gbps();
        for b in iter {
            s = s + b;
        }
        s
    }
}

impl std::cmp::PartialEq for Bandwidth {
    fn eq(&self, other: &Self) -> bool {
        self.val().eq(&other.val())
    }
}

impl Eq for Bandwidth {}

impl std::cmp::PartialOrd for Bandwidth {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.val().partial_cmp(&other.val())
    }
}

impl std::cmp::Ord for Bandwidth {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.val().cmp(&other.val())
    }
}

macro_rules! impl_bandwidth_div_mul_for {
    ($($ty:ty),+ $(,)?) => (
        $(impl std::ops::Div<$ty> for Bandwidth {
            type Output = Self;
            fn div(self, rhs: $ty) -> Self::Output {
                Bandwidth {
                    val: ((self.val as f64) / rhs as f64) as u64,
                    unit: self.unit,
                }
            }
        }
        impl std::ops::Mul<$ty> for Bandwidth {
            type Output = Self;
            fn mul(self, rhs: $ty) -> Self::Output {
                Bandwidth {
                    val: ((self.val as f64) * rhs as f64) as u64,
                    unit: self.unit,
                }
            }
        })+
    )
}

impl_bandwidth_div_mul_for!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, isize, usize);

impl std::ops::Sub for Bandwidth {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Bandwidth {
            val: self.val - rhs.val,
            unit: self.unit,
        }
    }
}

impl std::ops::Add for Bandwidth {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Bandwidth {
            val: self.val + rhs.val,
            unit: self.unit,
        }
    }
}

macro_rules! impl_bandwidth_trait_for {
    ($($ty:ty),+ $(,)?) => (
        $(impl BandwidthTrait for $ty
        {
            fn kbps(self) -> Bandwidth {
                let unit = BandwidthUnit::Kbps;
                Bandwidth {
                    val: (self as f64 * unit as u64 as f64) as u64,
                    unit,
                }
            }
            fn mbps(self) -> Bandwidth {
                let unit = BandwidthUnit::Mbps;
                Bandwidth {
                    val: (self as f64 * unit as u64 as f64) as u64,
                    unit,
                }
            }
            fn gbps(self) -> Bandwidth {
                let unit = BandwidthUnit::Gbps;
                Bandwidth {
                    val: (self as f64 * unit as u64 as f64) as u64,
                    unit,
                }
            }
        })+
    )
}

impl_bandwidth_trait_for!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, isize, usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it() {
        let a: Bandwidth = 1000.kbps();
        let b: Bandwidth = (10.1555).mbps();
        let c: Bandwidth = (9.5).gbps();
        assert_eq!(format!("{:?}", a), "Bandwidth { val: 1000000, unit: Kbps }");
        assert_eq!(format!("{}", b.val()), "10155500");
        assert_eq!(format!("{}", c.kbps()), "9500000 Kb/s");
        assert_eq!(format!("{}", (c - b).mbps()), "9489.8445 Mb/s");
        assert_eq!(format!("{}", (c + b).mbps() / 1000), "9.510155 Mb/s");
        assert_eq!(
            format!("{}", (c - b).gbps() * 1000 + a * 1000),
            "9490.8445 Gb/s"
        );

        let d: Bandwidth = c / 1000;
        assert_eq!(format!("{}", d.mbps()), "9.5 Mb/s");
        assert_eq!(format!("{}", ((c / 1000) as Bandwidth).mbps()), "9.5 Mb/s");
    }
}
