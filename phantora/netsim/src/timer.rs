use std::any::Any;

use crate::simulator::TimerId;
use crate::{Duration, Timestamp, Token};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimerKind {
    Repeat, // used by nethint sampler
    Once,   // registered by application
}

pub trait Timer: std::any::Any {
    fn next_alert(&self) -> Timestamp;
    fn kind(&self) -> TimerKind;
    fn as_any(&self) -> &dyn Any;
    fn as_box_any(self: Box<Self>) -> Box<dyn Any>;
}

/// A timer triggered every dura nanoseconds since registration.
#[derive(Debug)]
pub struct RepeatTimer {
    kind: TimerKind,
    next_ready: Timestamp,
    dura: Duration,
}

impl RepeatTimer {
    pub fn new(next_ready: Timestamp, dura: Duration) -> Self {
        RepeatTimer {
            kind: TimerKind::Repeat,
            next_ready,
            dura,
        }
    }

    pub fn reset(&mut self) {
        self.next_ready += self.dura;
    }
}

use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};

/// A timer triggered event accroding to a possion distribution since registration.
#[derive(Debug)]
pub struct PoissonTimer {
    kind: TimerKind,
    next_ready: Timestamp,
    #[allow(unused)]
    lambda: f64,
    poi: Poisson<f64>,
    rng: rand::rngs::StdRng,
}

impl PoissonTimer {
    pub fn new(next_ready: Timestamp, lambda: f64) -> Self {
        PoissonTimer {
            kind: TimerKind::Repeat,
            next_ready,
            lambda,
            poi: Poisson::new(lambda).unwrap(),
            rng: rand::rngs::StdRng::seed_from_u64(0),
        }
    }

    pub fn reset(&mut self) {
        let v = self.poi.sample(&mut self.rng);
        self.next_ready += v as Duration;
    }
}

/// A timer triggered after dura only once since registration.
#[derive(Debug)]
pub struct OnceTimer {
    kind: TimerKind,
    next_ready: Timestamp,
    pub token: Option<Token>,
    pub timer_id: Option<TimerId>,
}

impl OnceTimer {
    pub fn new(next_ready: Timestamp, token: Option<Token>, timer_id: Option<TimerId>) -> Self {
        OnceTimer {
            kind: TimerKind::Once,
            next_ready,
            token,
            timer_id,
        }
    }
}

macro_rules! impl_timer_for {
    ($($ty:ty),+ $(,)?) => (
        $(
            impl Timer for $ty {
                fn next_alert(&self) -> Timestamp {
                    self.next_ready
                }
                fn kind(&self) -> TimerKind {
                    self.kind
                    // TODO(cjr): use downcast()
                }
                fn as_any(&self) -> &dyn Any {
                    self
                }
                fn as_box_any(self: Box<Self>) -> Box<dyn Any> {
                    self
                }
            }
            impl PartialEq for $ty {
                fn eq(&self, other: &$ty) -> bool {
                    self.next_alert().eq(&other.next_alert())
                }
            }

            impl Eq for $ty {}

            impl PartialOrd for $ty {
                fn partial_cmp(&self, other: &$ty) -> Option<std::cmp::Ordering> {
                    other.next_alert().partial_cmp(&self.next_alert())
                }
            }

            impl Ord for $ty {
                fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                    other.next_alert().cmp(&self.next_alert())
                }
            }
        )+
    )
}

impl PartialEq for Box<dyn Timer> {
    fn eq(&self, other: &Box<dyn Timer>) -> bool {
        self.next_alert().eq(&other.next_alert())
    }
}

impl Eq for Box<dyn Timer> {}

impl PartialOrd for Box<dyn Timer> {
    fn partial_cmp(&self, other: &Box<dyn Timer>) -> Option<std::cmp::Ordering> {
        other.next_alert().partial_cmp(&self.next_alert())
    }
}

impl Ord for Box<dyn Timer> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.next_alert().cmp(&self.next_alert())
    }
}

impl_timer_for!(RepeatTimer, PoissonTimer, OnceTimer);
