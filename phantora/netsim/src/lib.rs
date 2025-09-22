use std::cell::RefCell;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use lazy_static::lazy_static;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use strum_macros::EnumString;

const RAND_SEED: u64 = 0;
thread_local! {
    pub static RNG: Rc<RefCell<StdRng>> = Rc::new(RefCell::new(StdRng::seed_from_u64(RAND_SEED)));
}

lazy_static! {
    static ref FLOW_ID: AtomicUsize = AtomicUsize::new(0);
    static ref TIMER_ID: AtomicUsize = AtomicUsize::new(0);
}

pub type TenantId = usize;

pub mod app;
pub mod bandwidth;
pub mod cluster;
pub mod simulator;
pub use simulator::{Event, Events};
pub mod architecture;
pub mod background_flow;
pub mod config;
pub mod timer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FairnessModel {
    PerFlowMaxMin,
    PerVmPairMaxMin,
    TenantFlowMaxMin,
}

impl std::default::Default for FairnessModel {
    fn default() -> Self {
        Self::PerFlowMaxMin
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SharingMode {
    RateLimited,
    Guaranteed,
}

impl std::default::Default for SharingMode {
    fn default() -> Self {
        Self::Guaranteed
    }
}

// nanoseconds
pub type Timestamp = u64;
pub type Duration = u64;

pub trait ToStdDuration {
    fn to_dura(self) -> std::time::Duration;
}

impl ToStdDuration for u64 {
    #[inline]
    fn to_dura(self) -> std::time::Duration {
        std::time::Duration::new(self / 1_000_000_000, (self % 1_000_000_000) as u32)
    }
}

/// A flow Trace is a table of flow record.
#[derive(Debug, Clone)]
pub struct Trace {
    pub recs: Vec<TraceRecord>,
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

impl Trace {
    pub fn new() -> Self {
        Trace { recs: Vec::new() }
    }

    #[inline]
    pub fn add_record(&mut self, rec: TraceRecord) {
        self.recs.push(rec);
    }
}

#[derive(Clone, Hash, PartialEq)]
pub struct TraceRecord {
    /// The start timestamp of the flow.
    pub ts: Timestamp,
    pub flow: Flow,
    pub dura: Option<Duration>, // this is calculated by the simulator
}

impl TraceRecord {
    #[inline]
    pub fn new(ts: Timestamp, flow: Flow, dura: Option<Duration>) -> Self {
        TraceRecord { ts, flow, dura }
    }
}

impl std::fmt::Debug for TraceRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dura = self.dura.map(|x| x.to_dura());
        f.debug_struct("TraceRecord")
            .field("ts", &self.ts)
            .field("flow", &self.flow)
            .field("dura", &dura)
            .finish()
    }
}

#[derive(Debug, Clone, Eq, Serialize, Deserialize)]
pub struct Flow {
    /// an identifier
    id: usize,
    pub bytes: usize,
    pub src: String,
    pub dst: String,
    /// an optional tag for application use (e.g. identify the flow in application)
    pub token: Option<Token>,
    /// this field is to explicitly support tenant based fairness
    tenant_id: Option<TenantId>,
    /// TODO(cjr): This is a hack to let the simulator be able to
    /// be aware of the virtual address.
    /// A better way to do this is to encapsulate the inner header
    /// inside the new header instead of replacing fields in place.
    vsrc: Option<String>,
    vdst: Option<String>,
    /// Patched field. Used when combined with network load balancer to determine the path among
    /// multiple network links.
    pub udp_src_port: Option<u16>,
}

impl Hash for Flow {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl PartialEq for Flow {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Flow {
    #[inline]
    pub fn new(bytes: usize, src: &str, dst: &str, token: Option<Token>) -> Self {
        Flow {
            id: FLOW_ID.fetch_add(1, SeqCst),
            bytes,
            src: src.to_owned(),
            dst: dst.to_owned(),
            tenant_id: None,
            token,
            vsrc: None,
            vdst: None,
            udp_src_port: None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize)]
pub struct Token(pub usize);

impl From<usize> for Token {
    fn from(val: usize) -> Token {
        Token(val)
    }
}

impl From<Token> for usize {
    fn from(val: Token) -> usize {
        val.0
    }
}

pub trait LoadBalancer: std::fmt::Debug + Send + Sync + 'static {
    fn compute_hash(&mut self, flow: &Flow) -> u64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, EnumString)]
pub enum LoadBalancerType {
    EcmpEverything,
    EcmpSourcePort,
}

#[derive(Clone, Debug)]
pub struct EcmpFlowHasher {
    // We manually add this fake information to a flow because we don't have source port inside the
    // `Flow` structure.
    udp_src_port: u16,
    rng: StdRng,
}

impl EcmpFlowHasher {
    pub fn new(udp_src_port: u16, seed: u64) -> Self {
        Self {
            udp_src_port,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl LoadBalancer for EcmpFlowHasher {
    fn compute_hash(&mut self, flow: &Flow) -> u64 {
        // EcmpFlowHasher
        let mut hasher = DefaultHasher::new();
        flow.id.hash(&mut hasher);
        flow.src.hash(&mut hasher);
        flow.dst.hash(&mut hasher);
        let random_step = self.rng.gen_range(1..=100);
        self.udp_src_port = self.udp_src_port.wrapping_add(random_step);
        self.udp_src_port.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Clone, Debug, Default)]
pub struct EcmpSourcePortHasher;

impl LoadBalancer for EcmpSourcePortHasher {
    fn compute_hash(&mut self, flow: &Flow) -> u64 {
        // A hasher that only considers UDP source port.
        // UDP source port is not used by RoCEv2. So our algorithm
        // leverages this field to realize the ability to pin a flow
        // to a desginated network path.
        // let mut hasher = DefaultHasher::new();
        // flow.udp_src_port.hash(&mut hasher);
        // hasher.finish()
        match flow.udp_src_port {
            Some(p) => p as _,
            None => panic!("You forget to set UDP source port"),
        }
    }
}
