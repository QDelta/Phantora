use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::rc::Rc;
use std::time;
use std::{cmp::Reverse, fmt::Debug};

use fnv::{FnvBuildHasher, FnvHashMap};
use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smallvec::{smallvec, SmallVec};
use thiserror::Error;

use crate::{
    app::{AppEvent, AppEventKind, Application, Replayer},
    bandwidth::{self, Bandwidth, BandwidthTrait},
    cluster::{Cluster, Link, LinkIx, Route, RouteHint, Topology},
    timer::{OnceTimer, Timer, TimerKind},
};
use crate::{
    Duration, FairnessModel, Flow, TenantId, Timestamp, ToStdDuration, Token, Trace, TraceRecord,
    TIMER_ID,
};

type HashMap<K, V> = IndexMap<K, V, FnvBuildHasher>;
type HashMapValues<'a, K, V> = indexmap::map::Values<'a, K, V>;
type HashMapValuesMut<'a, K, V> = indexmap::map::ValuesMut<'a, K, V>;

pub const LOOPBACK_SPEED_GBPS: u64 = 400; // 400Gbps

#[derive(Debug, Clone, PartialEq)]
pub enum OnEventResult {
    SimulationFinished,
    Ok,
    Rollback(Vec<TraceRecord>),
}

/// The simulator driver API
pub trait Executor<'a> {
    fn run_with_trace(&mut self, trace: Trace) -> Trace;
    fn run_with_application<T>(&mut self, app: Box<dyn Application<Output = T> + 'a>) -> T;
    /// Allow the simulator to take event from the outside world.
    fn on_event(&mut self, event: Event) -> OnEventResult;
    fn run_one_step(&mut self, until: Option<Timestamp>) -> AppEvent;
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SimulatorSetting {
    #[serde(serialize_with = "serialize_bandwidth")]
    #[serde(deserialize_with = "deserialize_bandwidth")]
    pub loopback_speed: Bandwidth,
    pub fairness: FairnessModel,
}

impl Default for SimulatorSetting {
    fn default() -> Self {
        Self {
            fairness: FairnessModel::default(),
            loopback_speed: LOOPBACK_SPEED_GBPS.gbps(),
        }
    }
}

fn serialize_bandwidth<S>(bw: &Bandwidth, se: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let s = bw.to_string();
    s.serialize(se)
}

fn deserialize_bandwidth<'de, D>(de: D) -> Result<Bandwidth, D::Error>
where
    D: Deserializer<'de>,
{
    let f: f64 = Deserialize::deserialize(de)?;
    Ok(f.gbps())
}

type HostMapping = FnvHashMap<String, String>;

#[derive(Debug, Clone)]
pub struct SimulatorBuilder {
    cluster: Option<Cluster>,
    setting: SimulatorSetting,
    // Real world hostname to "host_{i}" (the naming convention used in `cluster::Cluster`)
    host_mapping: Option<HostMapping>,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("Cluster not provided")]
    EmptyCluster,
}

impl Default for SimulatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulatorBuilder {
    pub fn new() -> Self {
        SimulatorBuilder {
            cluster: None,
            setting: Default::default(),
            host_mapping: None,
        }
    }

    pub fn with_setting(&mut self, setting: SimulatorSetting) -> &mut Self {
        self.setting = setting;
        self
    }

    pub fn host_mapping(&mut self, host_mapping: Vec<String>) -> &mut Self {
        self.host_mapping = Some(
            host_mapping
                .into_iter()
                .enumerate()
                .map(|(i, h)| (h, format!("host_{}", i)))
                .collect(),
        );
        self
    }

    pub fn cluster(&mut self, cluster: Cluster) -> &mut Self {
        self.cluster = Some(cluster);
        self
    }

    pub fn fairness(&mut self, fairness: FairnessModel) -> &mut Self {
        self.setting.fairness = fairness;
        self
    }

    pub fn loopback_speed(&mut self, loopback_speed: Bandwidth) -> &mut Self {
        self.setting.loopback_speed = loopback_speed;
        self
    }

    pub fn build(&mut self) -> Result<Simulator, Error> {
        if self.cluster.is_none() {
            return Err(Error::EmptyCluster);
        }

        Ok(Simulator {
            cluster: self.cluster.take().unwrap(),
            ts: 0,
            state: NetState::default(),
            timers: BinaryHeap::<Box<dyn Timer>>::new(),
            setting: self.setting,
            host_mapping: self.host_mapping.take(),
        })
    }
}

/// The flow-level simulator.
pub struct Simulator {
    cluster: Cluster,
    ts: Timestamp,
    state: NetState,
    timers: BinaryHeap<Box<dyn Timer>>,
    // setting
    setting: SimulatorSetting,
    host_mapping: Option<HostMapping>,
}

macro_rules! calc_delta_on_group {
    ($total_bw:expr, $m:expr) => {{
        let mut consumed_bw = 0.0;
        let mut num_active_objects = 0;
        let mut num_active_flows = Vec::new();
        let mut min_inc_to_max_rate = f64::MAX;
        for fs in $m.values() {
            let mut active_flows_per_object = 0;
            for f in fs {
                let f = f.borrow();
                consumed_bw += f.speed;
                if !f.converged {
                    active_flows_per_object += 1;
                    assert!(f.speed < f.max_rate.val() as f64 + 10.0, "flow: {:?}", f);
                    min_inc_to_max_rate =
                        min_inc_to_max_rate.min(f.max_rate.val() as f64 - f.speed);
                }
            }

            num_active_objects += (active_flows_per_object > 0) as usize;
            num_active_flows.push(active_flows_per_object);
        }

        assert_ne!(num_active_objects, 0);

        let bw_inc_per_object = if $total_bw < (consumed_bw / 1e9).gbps() {
            0.gbps()
        } else {
            ($total_bw - (consumed_bw / 1e9).gbps()) / num_active_objects as f64
        };

        let mut min_inc = bw_inc_per_object.min((min_inc_to_max_rate / 1e9).gbps());

        // set when a flow will converge
        for (i, fs) in $m.values_mut().enumerate() {
            if num_active_flows[i] == 0 {
                continue;
            }
            let speed_inc_per_object = bw_inc_per_object / num_active_flows[i] as f64;
            min_inc = min_inc.min(speed_inc_per_object);

            let speed_inc_per_object_f64 = speed_inc_per_object.val() as f64;
            for f in fs {
                let mut f = f.borrow_mut();
                f.speed_bound = f.speed_bound.min(f.speed + speed_inc_per_object_f64);
            }
        }

        min_inc
    }};
}

impl Simulator {
    pub fn new(cluster: Cluster) -> Self {
        Simulator {
            cluster,
            ts: 0,
            state: Default::default(),
            timers: BinaryHeap::new(),
            setting: SimulatorSetting::default(),
            host_mapping: None,
        }
    }

    pub fn suspend(&mut self, _path: &std::path::Path) {
        // dump all running states of the simulator
        unimplemented!();
    }

    pub fn resume(&mut self, _path: &std::path::Path) {
        // resume from the previous saved state
        unimplemented!();
    }

    #[inline]
    pub fn update_flow_start_ts(&mut self, flow: &Flow, new_ts_ns: Timestamp) {
        self.state.update_flow_start_ts(flow, new_ts_ns);
    }

    fn register_once(
        &mut self,
        next_ready: Timestamp,
        token: Option<Token>,
        timer_id: Option<TimerId>,
    ) {
        self.timers
            .push(Box::new(OnceTimer::new(next_ready, token, timer_id)));
    }

    fn calc_delta_per_flow(total_bw: Bandwidth, fs: &mut FlowSet) -> Bandwidth {
        let (num_active, consumed_bw, min_inc) = fs.iter().fold((0, 0.0, f64::MAX), |acc, f| {
            let f = f.borrow();
            assert!(f.speed <= f.max_rate.val() as f64 + 10.0, "f: {:?}", f);
            (
                acc.0 + !f.converged as usize,
                acc.1 + f.speed,
                acc.2.min(if f.converged {
                    f.max_rate.val() as f64 - f.speed
                } else {
                    f64::MAX
                }),
            )
        });
        // COMMENT(cjr): due to precision issue, here consumed_bw can be a little bigger than bw
        let mut bw_inc = if total_bw < (consumed_bw / 1e9).gbps() {
            0.gbps()
        } else {
            (total_bw - (consumed_bw / 1e9).gbps()) / num_active as f64
        };

        // some flows may reach its max_rate earlier
        bw_inc = bw_inc.min((min_inc / 1e9).gbps());

        // set when a flow will converge
        fs.iter_mut()
            .map(|f| f.borrow_mut())
            .for_each(|mut f| f.speed_bound = f.speed_bound.min(f.speed + bw_inc.val() as f64));
        bw_inc
    }

    fn calc_delta_groupped(total_bw: Bandwidth, fs: &mut FlowSet) -> Bandwidth {
        match fs {
            FlowSet::GroupByVmPair(m) => {
                calc_delta_on_group!(total_bw, m)
            }
            FlowSet::GroupByTenant(m) => {
                calc_delta_on_group!(total_bw, m)
            }
            _ => panic!("flows must be groupped by some label"),
        }
    }

    fn calc_delta(fairness: FairnessModel, l: &Link, fs: &mut FlowSet) -> Bandwidth {
        match fairness {
            FairnessModel::PerFlowMaxMin => Self::calc_delta_per_flow(l.bandwidth, fs),
            FairnessModel::PerVmPairMaxMin => Self::calc_delta_groupped(l.bandwidth, fs),
            FairnessModel::TenantFlowMaxMin => Self::calc_delta_groupped(l.bandwidth, fs),
        }
    }

    fn proceed(&mut self, ts_inc: Duration) -> Vec<TraceRecord> {
        // complete some flows
        let comp_flows = self.state.complete_flows(self.ts, ts_inc);

        // start new ready flows
        self.ts += ts_inc;
        self.state.emit_ready_flows(self.ts);

        comp_flows
    }

    fn max_min_fairness_converge(&mut self) {
        let mut converged = 0;
        let active_flows = self.state.running_flows.len();
        self.state.running_flows.iter().for_each(|f| {
            let mut f = f.borrow_mut();
            if f.is_loopback() {
                f.speed_bound = self.setting.loopback_speed.val() as f64;
                f.converged = true;
                f.speed = self.setting.loopback_speed.val() as f64;
                converged += 1;
            } else {
                f.speed_bound = f.max_rate.val() as f64;
                f.converged = false;
                f.speed = 0.0;
            }
        });

        let fairness = self.setting.fairness;
        while converged < active_flows {
            // find the bottleneck link
            let bw = self
                .state
                .link_flows
                .iter_mut()
                .filter(|(_, fs)| fs.iter().any(|f| !f.borrow().converged)) // this seems to be redundant
                .map(|(l, fs)| {
                    assert!(!fs.is_empty());
                    let link = &self.cluster[*l];
                    Self::calc_delta(fairness, link, fs)
                })
                .min()
                .unwrap_or_else(|| {
                    panic!(
                        "converged: {} vs active: {}\n=== running_flows: {:?}\n=== link_flows: {:?}",
                        converged, active_flows,
                        self.state.running_flows, self.state.link_flows
                    )
                });

            let speed_inc = bw.val() as f64;

            // increase the speed of all active flows
            for f in &self.state.running_flows {
                let mut f = f.borrow_mut();
                if !f.converged {
                    f.speed += speed_inc;
                    // TODO(cjr): be careful about this
                    if f.speed + 1e-10 >= f.speed_bound {
                        f.converged = true;
                        converged += 1;
                    } else {
                        f.speed_bound = f.max_rate.val() as f64;
                    }
                }
            }
        }
    }

    fn max_min_fairness(&mut self, until: Option<Timestamp>) -> AppEventKind {
        loop {
            if self.ts >= until.unwrap_or(Timestamp::max_value()) {
                break AppEventKind::FlowComplete(vec![]);
            }

            if self.state.running_flows.is_empty()
                && self.state.flow_bufs.is_empty()
                && self.timers.is_empty()
            {
                break AppEventKind::FlowComplete(vec![]);
            }

            // TODO(cjr): Optimization: if netstate hasn't been changed
            // (i.e. new newly added or completed flows), then skip max_min_fairness_converge.
            // compute a fair share of bandwidth allocation
            self.max_min_fairness_converge();
            log::trace!(
                "after max_min_fairness converged, ts: {:?}, running flows: {:#?}\nnumber of ready flows: {}",
                self.ts.to_dura(),
                self.state.running_flows,
                self.state.flow_bufs.len(),
            );
            // all FlowStates are converged

            // find the next flow to complete
            let first_complete_time = self
                .state
                .running_flows
                .iter()
                .map(|f| f.borrow().time_to_complete() + self.ts)
                .min();

            // find the next flow to start
            let first_ready_time = self.state.flow_bufs.peek().map(|(_, fs)| fs.0.borrow().ts);

            // get min from first_complete_time and first_ready_time, both could be None
            let ts_inc = first_complete_time
                .into_iter()
                .chain(first_ready_time)
                .chain(self.timers.peek().map(|x| x.next_alert()))
                .chain(until)
                .min()
                .expect("running flows, ready flows, and timers are all empty")
                - self.ts;

            log::trace!("self.ts: {}, ts_inc: {}", self.ts, ts_inc);
            assert!(
                !(first_complete_time.is_none()
                    && first_ready_time.is_none()
                    && self.timers.is_empty())
            );

            // if it is not due to until, it must be the timer
            // jianxing: not sure if it's still the case now
            if ts_inc == 0 {
                // the next event should be the timer event
                if let Some(timer) = self.timers.peek() {
                    // if timer.next_alert() != self.ts {
                    //     log::warn!(
                    //         "ts_inc = 0, while next alert ts: {}, self.ts: {}",
                    //         timer.next_alert(),
                    //         self.ts
                    //     );
                    // }
                    if timer.next_alert() == self.ts {
                        if timer.kind() == TimerKind::Once {
                            let timer = self.timers.pop().unwrap();
                            let once_timer = timer.as_any().downcast_ref::<OnceTimer>().unwrap();
                            log::trace!("{:?}", once_timer);
                            let token = once_timer.token;
                            let timer_id = once_timer.timer_id;
                            if timer_id.is_some() {
                                break AppEventKind::AdapterNotification(token, timer_id.unwrap());
                            } else {
                                break AppEventKind::UserNotification(token);
                            }
                        }
                    }
                }
            }

            // assert!(
            //     ts_inc > 0
            //         || (ts_inc == 0
            //             && (self
            //                 .timers
            //                 .peek()
            //                 .and_then(|timer| if timer.kind() == TimerKind::Repeat {
            //                     Some(())
            //                 } else {
            //                     None
            //                 })
            //                 .is_some()
            //                 || until.is_some())),
            //     "only Nethint timers can cause ts_inc == 0"
            // );

            // modify the network state to the time at ts + ts_inc
            let comp_flows = self.proceed(ts_inc);
            if !comp_flows.is_empty() {
                log::trace!(
                    "ts: {:?}, completed flows: {:?}",
                    self.ts.to_dura(),
                    comp_flows
                );
                break AppEventKind::FlowComplete(comp_flows);
            }
        }
    }
}

impl<'a> Executor<'a> for Simulator {
    fn run_with_trace(&mut self, trace: Trace) -> Trace {
        let app = Box::new(Replayer::new(trace));
        self.run_with_application(app)
    }

    fn run_one_step(&mut self, until: Option<Timestamp>) -> AppEvent {
        let app_event_kind = self.max_min_fairness(until);
        AppEvent::new(self.ts, app_event_kind)
    }

    #[inline]
    fn on_event(&mut self, event: Event) -> OnEventResult {
        log::trace!("simulator: on event {:?}", event);
        match event {
            Event::FlowArrive(recs) => {
                assert!(!recs.is_empty(), "No flow arrives.");
                // 1. find path for each flow and add to current net state
                for r in &recs {
                    self.state.add_flow(
                        r.clone(),
                        &self.cluster,
                        self.ts,
                        self.host_mapping.as_ref(),
                    );
                }
                let min_ts = recs.iter().map(|r| r.ts).min().unwrap_or(self.ts);
                if min_ts < self.ts / 1000 * 1000 {
                    let affected_flows = self.state.rollback_to(self.ts, min_ts);
                    self.ts = min_ts;
                    let mut ret = Vec::new();
                    for fs in affected_flows {
                        let f = fs.borrow();
                        ret.push(TraceRecord::new(f.ts, f.flow.clone(), None));
                    }
                    return OnEventResult::Rollback(ret);
                }
            }
            Event::AppFinish => {
                return OnEventResult::SimulationFinished;
            }
            Event::AdapterRegisterTimer(after_dura, token, timer_id) => {
                self.register_once(self.ts + after_dura, token, Some(timer_id));
            }
            Event::UserRegisterTimer(after_dura, token) => {
                // panic!("currently we do not support user app directly register this
                //         kind of timer, considering save and restore the token
                //         just like stored_flow_token");
                self.register_once(self.ts + after_dura, token, None);
            }
        }
        OnEventResult::Ok
    }

    fn run_with_application<T>(&mut self, mut app: Box<dyn Application<Output = T> + 'a>) -> T {
        macro_rules! app_event {
            ($kind:expr) => {{
                let kind = $kind;
                AppEvent::new(self.ts, kind)
            }};
        }

        let start = time::Instant::now();
        let mut events = app.on_event(app_event!(AppEventKind::AppStart));
        let mut new_events = Events::new();
        loop {
            let mut finished = false;
            events.reverse();
            log::trace!("simulator: events.len: {:?}", events.len());
            while let Some(event) = events.pop() {
                finished |= self.on_event(event) == OnEventResult::SimulationFinished;
            }

            if finished {
                break;
            }

            if !new_events.is_empty() {
                // NetHintResponse sent, new flows may arrive
                // must handle them first before computing max_min_fairness
                std::mem::swap(&mut events, &mut new_events);
                continue;
            }

            // 2. run max-min fairness to find the next completed flow
            let app_event_kind = self.max_min_fairness(None);

            // 3. nofity the application with this flow
            new_events.append(app.on_event(app_event!(app_event_kind)));
            std::mem::swap(&mut events, &mut new_events);
        }

        log::debug!("sim_time: {:?}", start.elapsed());
        // output
        app.answer()
    }
}

fn hash_vm_pair(f: &Flow) -> usize {
    let vsrc_id: usize = f
        .vsrc
        .as_ref()
        .unwrap()
        .strip_prefix("host_")
        .unwrap()
        .parse()
        .unwrap();
    let vdst_id: usize = f
        .vsrc
        .as_ref()
        .unwrap()
        .strip_prefix("host_")
        .unwrap()
        .parse()
        .unwrap();
    (vsrc_id << 32) | vdst_id
}

#[derive(Debug)]
pub(crate) enum FlowSet {
    Flat(Vec<FlowStateRef>),
    // src, dst -> vector of flows
    GroupByVmPair(HashMap<usize, Vec<FlowStateRef>>),
    GroupByTenant(HashMap<TenantId, Vec<FlowStateRef>>),
}

impl FlowSet {
    fn new(fairness: FairnessModel) -> FlowSet {
        match fairness {
            FairnessModel::PerFlowMaxMin => FlowSet::Flat(Vec::new()),
            FairnessModel::PerVmPairMaxMin => FlowSet::GroupByVmPair(HashMap::default()),
            FairnessModel::TenantFlowMaxMin => FlowSet::GroupByTenant(HashMap::default()),
        }
    }

    fn push(&mut self, fs: FlowStateRef) {
        match self {
            Self::Flat(v) => v.push(fs),
            Self::GroupByVmPair(m) => {
                // we can just unwrap here because it must have valid vsrc and vdst fields,
                // otherwise, it does not make sense to use PerVmPairMaxMin
                let key = hash_vm_pair(&fs.borrow().flow);
                m.entry(key).or_insert_with(Vec::new).push(fs);
            }
            Self::GroupByTenant(m) => {
                let key = fs
                    .borrow()
                    .flow
                    .tenant_id
                    .unwrap_or_else(|| panic!("flow: {:?}", fs.borrow().flow));
                m.entry(key).or_insert_with(Vec::new).push(fs);
            }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Flat(v) => v.is_empty(),
            Self::GroupByVmPair(m) => m.is_empty(),
            Self::GroupByTenant(m) => m.is_empty(),
        }
    }

    fn retain<F>(&mut self, f: F)
    where
        F: Clone + FnMut(&FlowStateRef) -> bool,
    {
        match self {
            Self::Flat(v) => v.retain(f),
            Self::GroupByVmPair(m) => {
                m.values_mut().for_each(|v| v.retain(f.clone()));
                m.retain(|_, v| !v.is_empty());
            }
            Self::GroupByTenant(m) => {
                m.values_mut().for_each(|v| v.retain(f.clone()));
                m.retain(|_, v| !v.is_empty());
            }
        }
    }

    pub(crate) fn iter(&self) -> FlowSetIter<'_> {
        match self {
            Self::Flat(v) => FlowSetIter::FlatIter(v.iter()),
            Self::GroupByVmPair(m) => {
                let mut iter1 = m.values();
                let iter2 = iter1.next().map(|v| v.iter());
                FlowSetIter::GroupByVmPairIter(iter1, iter2)
            }
            Self::GroupByTenant(m) => {
                let mut iter1 = m.values();
                let iter2 = iter1.next().map(|v| v.iter());
                FlowSetIter::GroupByTenantIter(iter1, iter2)
            }
        }
    }

    fn iter_mut(&mut self) -> FlowSetIterMut<'_> {
        match self {
            Self::Flat(v) => FlowSetIterMut::FlatIter(v.iter_mut()),
            Self::GroupByVmPair(m) => {
                let mut iter1 = m.values_mut();
                let iter2 = iter1.next().map(|v| v.iter_mut());
                FlowSetIterMut::GroupByVmPairIter(iter1, iter2)
            }
            Self::GroupByTenant(m) => {
                let mut iter1 = m.values_mut();
                let iter2 = iter1.next().map(|v| v.iter_mut());
                FlowSetIterMut::GroupByTenantIter(iter1, iter2)
            }
        }
    }
}

#[derive(Debug)]
pub(crate) enum FlowSetIter<'a> {
    FlatIter(std::slice::Iter<'a, FlowStateRef>),
    GroupByVmPairIter(
        HashMapValues<'a, usize, Vec<FlowStateRef>>,
        Option<std::slice::Iter<'a, FlowStateRef>>,
    ),
    GroupByTenantIter(
        HashMapValues<'a, TenantId, Vec<FlowStateRef>>,
        Option<std::slice::Iter<'a, FlowStateRef>>,
    ),
}

enum FlowSetIterMut<'a> {
    FlatIter(std::slice::IterMut<'a, FlowStateRef>),
    GroupByVmPairIter(
        HashMapValuesMut<'a, usize, Vec<FlowStateRef>>,
        Option<std::slice::IterMut<'a, FlowStateRef>>,
    ),
    GroupByTenantIter(
        HashMapValuesMut<'a, TenantId, Vec<FlowStateRef>>,
        Option<std::slice::IterMut<'a, FlowStateRef>>,
    ),
}

macro_rules! impl_groupped_iter {
    ($iter1:expr, $iter2:expr, $iter_func:ident) => {
        if $iter2.is_none() {
            None
        } else {
            match $iter2.as_mut().unwrap().next() {
                Some(ret) => Some(ret),
                None => {
                    *$iter2 = $iter1.next().map(|v| v.$iter_func());
                    if $iter2.is_none() {
                        None
                    } else {
                        $iter2.as_mut().unwrap().next()
                    }
                }
            }
        }
    };
}

macro_rules! impl_iter_for {
    ($name:ident, $item:ty, $iter_func:ident) => {
        impl<'a> Iterator for $name<'a> {
            type Item = $item;
            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    Self::FlatIter(iter) => iter.next(),

                    Self::GroupByVmPairIter(iter1, iter2) => {
                        impl_groupped_iter!(iter1, iter2, $iter_func)
                    }

                    Self::GroupByTenantIter(iter1, iter2) => {
                        impl_groupped_iter!(iter1, iter2, $iter_func)
                    }
                }
            }
        }
    };
}

impl_iter_for!(FlowSetIter, &'a FlowStateRef, iter);
impl_iter_for!(FlowSetIterMut, &'a mut FlowStateRef, iter_mut);

use priority_queue::PriorityQueue;

#[derive(Default)]
struct NetState {
    resolve_route_time: time::Duration,
    fairness: FairnessModel,

    /// buffered flows, flow.ts > sim.ts
    flow_bufs: PriorityQueue<Flow, Reverse<FlowStateRef>>,
    // emitted flows
    running_flows: Vec<FlowStateRef>,
    link_flows: HashMap<LinkIx, FlowSet>,
    loopback_flows: Vec<FlowStateRef>,

    // For rolling back, so these flows are not retired forever
    // The retired_flows should be monotonically increasing.
    retired_flows: Vec<FlowStateRef>,
}

impl NetState {
    fn update_flow_start_ts(&mut self, flow: &Flow, new_ts_ns: Timestamp) {
        let found = self.flow_bufs.change_priority_by(&flow, |Reverse(fs)| {
            let mut f = fs.borrow_mut();
            f.ts = f.ts.max(new_ts_ns);
        });
        assert!(found, "flow: {:?}", flow);
    }

    fn rollback_to(&mut self, sim_ts: Timestamp, past_ts: Timestamp) -> Vec<FlowStateRef> {
        let mut retired_to_running = Vec::new();
        // Revive retired flows, adding them back to running flows
        while let Some(fs) = self.retired_flows.last() {
            let f = fs.borrow();
            // This is a completed flow, so the speed_history must not be empty
            assert!(!f.speed_history.is_empty());
            // The last entry in the speed history is (end_time, 0.0).
            let last_entry = f.speed_history.last().unwrap();
            assert!(last_entry.speed == 0.0);
            let end_ts = last_entry.start;
            drop(f);
            // Ignore events at sub-microsecond level
            if past_ts < end_ts / 1000 * 1000 {
                // COMMENT(cjr): Only add to running_flows here because the code below will
                // handle the case where it should be further retired to flow_bufs.
                retired_to_running.push(Rc::clone(fs));
                self.retired_flows.pop();
            } else {
                // The complete time of retired_flows are always monotonically increasing
                // so we can early exit here.
                break;
            }
        }
        for fs in &retired_to_running {
            self.emit_flow(Rc::clone(fs));
        }
        // Then perform rollback to running flows
        let mut revived_flows = Vec::new();
        self.running_flows.retain(|fs| {
            let mut f = fs.borrow_mut();
            let mut last_ts = sim_ts;
            let mut reverted_bytes = 0.;
            log::trace!("ROLLING: {:?}", f.flow);
            // A flow's speed_history can be empty because this flow has just been ready and emitted
            while let Some(entry) = f.speed_history.last() {
                // entry0.start, entry1.start, ... entry_k.start, past_ts, entry_k+1.start, sim_ts
                log::trace!(
                    "ROLLING: {}, {}, {}, {}",
                    last_ts,
                    entry.start,
                    past_ts,
                    entry.speed
                );
                reverted_bytes +=
                    (last_ts - entry.start.max(past_ts)) as f64 * entry.speed / (8.0 * 1e9);
                last_ts = entry.start;
                if entry.start < past_ts {
                    break;
                }
                f.speed_history.pop();
            }

            // integer overflow is dangerous
            if reverted_bytes > f.bytes_sent {
                // the delta must be small enough
                assert!(
                    ((reverted_bytes - f.bytes_sent) as f64 / f.bytes_sent as f64) < 1e-6,
                    "reverted: {}, sent: {}",
                    reverted_bytes,
                    f.bytes_sent
                );
                reverted_bytes = f.bytes_sent;
            }
            f.bytes_sent -= reverted_bytes;
            if past_ts <= f.ts {
                assert!(
                    f.bytes_sent.abs() < 1e-10
                        || (f.bytes_sent as f64 / (f.bytes_sent + reverted_bytes) as f64) < 1e-6,
                    "f: {:?}, reverted_bytes: {}",
                    f,
                    reverted_bytes
                );
                assert!(f.speed_history.is_empty(), "f: {:?}", f);
                f.bytes_sent = 0.0;
                // This flow becomes an embryo, put it back to flow_bufs
                revived_flows.push(Rc::clone(fs));
            }
            past_ts > f.ts
        });

        // retire and filter link flows
        for (_, fs) in self.link_flows.iter_mut() {
            fs.retain(|f| past_ts > f.borrow().ts);
        }

        // retire all empty links
        self.link_flows.retain(|_, fs| !fs.is_empty());

        // retire and filter loopback_flows
        self.loopback_flows.retain(|f| past_ts > f.borrow().ts);

        // revive flows
        for fs in revived_flows {
            self.push_flow(fs, past_ts);
        }
        retired_to_running
    }

    fn emit_flow(&mut self, fs: FlowStateRef) {
        self.running_flows.push(Rc::clone(&fs));
        let fairness = self.fairness;
        for &l in &fs.borrow().route.path {
            self.link_flows
                .entry(l)
                .or_insert_with(|| FlowSet::new(fairness))
                .push(Rc::clone(&fs));
        }
        if fs.borrow().is_loopback() {
            self.loopback_flows.push(Rc::clone(&fs));
        }
    }

    fn add_flow(
        &mut self,
        mut r: TraceRecord,
        cluster: &Cluster,
        sim_ts: Timestamp,
        host_mapping: Option<&HostMapping>,
    ) {
        let start = time::Instant::now();
        if let Some(host_mapping) = host_mapping {
            r.flow.src = host_mapping
                .get(&r.flow.src)
                .unwrap_or_else(|| {
                    panic!("Hostname {} not found in {:?}", r.flow.src, host_mapping)
                })
                .clone();
            r.flow.dst = host_mapping
                .get(&r.flow.dst)
                .unwrap_or_else(|| {
                    panic!("Hostname {} not found in {:?}", r.flow.dst, host_mapping)
                })
                .clone();
        };
        let (max_rate, route) = {
            // only the physical cluster is what we have, no virtualization
            let hint = RouteHint::VirtAddr(r.flow.vsrc.as_deref(), r.flow.vdst.as_deref());
            let max_rate = bandwidth::MAX;
            (
                max_rate,
                cluster.resolve_route(&r.flow.src, &r.flow.dst, &hint),
            )
        };

        self.resolve_route_time += start.elapsed();
        self.push_flow(FlowState::new(r.ts, r.flow, max_rate, route), sim_ts);
    }

    /// Add flow to a buffer of ready flows (`flow_bufs`) or emit the flow immediately
    fn push_flow(&mut self, fs: FlowStateRef, sim_ts: Timestamp) {
        let flow_start_ts = fs.borrow().ts;
        if flow_start_ts > sim_ts {
            // add to buffered flows
            let flow = fs.borrow().flow.clone();
            self.flow_bufs.push(flow, Reverse(fs));
        } else {
            // add to current flow states, an invereted index
            self.emit_flow(fs);
        }
    }

    fn emit_ready_flows(&mut self, sim_ts: Timestamp) {
        while let Some((_, f)) = self.flow_bufs.peek() {
            let f = Rc::clone(&f.0);
            if sim_ts < f.borrow().ts {
                break;
            }
            self.flow_bufs.pop();
            assert_eq!(sim_ts, f.borrow().ts);
            self.emit_flow(f);
        }
    }

    fn complete_flows(&mut self, sim_ts: Timestamp, ts_inc: Duration) -> Vec<TraceRecord> {
        log::trace!(
            "complete_flows: sim_ts: {:?}, ts_inc: {:?}",
            sim_ts.to_dura(),
            ts_inc.to_dura()
        );
        let mut comp_flows = Vec::new();

        self.running_flows.iter().for_each(|fs| {
            let mut f = fs.borrow_mut();
            let speed = f.speed;
            f.speed_history.push(SpeedHistoryEntry::new(sim_ts, speed));
            assert!(
                f.speed <= f.max_rate.val() as f64 + 10.0,
                "flowstate: {:?}",
                f
            );
            // let delta = (speed / 1e9 * ts_inc as f64).round() as usize / 8;
            let delta = (speed / 1e9 * ts_inc as f64) / 8.0;
            f.bytes_sent += delta;

            if f.completed() {
                // log::debug!("ts: {}, comp f: {:?}", sim_ts + ts_inc, f);
                comp_flows.push(TraceRecord::new(
                    f.ts,
                    f.flow.clone(),
                    Some(sim_ts + ts_inc - f.ts),
                ));
                // Also add it to the retired flows
                self.retired_flows.push(Rc::clone(fs));
                // Add a termination entry
                f.speed_history
                    .push(SpeedHistoryEntry::new(sim_ts + ts_inc, 0.));
            }
        });

        self.running_flows.retain(|f| !f.borrow().completed());

        // finish and filter link flows
        for (_, fs) in self.link_flows.iter_mut() {
            fs.retain(|f| !f.borrow().completed());
        }

        // filter all empty links
        self.link_flows.retain(|_, fs| !fs.is_empty());

        // finish and filter loopback_flows
        self.loopback_flows.retain(|f| !f.borrow().completed());

        comp_flows
    }
}

type FlowStateRef = Rc<RefCell<FlowState>>;

#[derive(Debug, Clone, Copy, PartialEq)]
struct SpeedHistoryEntry {
    start: Timestamp,
    speed: f64,
}

impl SpeedHistoryEntry {
    fn new(start: Timestamp, speed: f64) -> Self {
        SpeedHistoryEntry { start, speed }
    }
}

/// A running flow
#[derive(Debug)]
pub(crate) struct FlowState {
    /// flow start time, could be greater than sim_ts, read only
    ts: Timestamp,
    /// read only flow property
    pub(crate) flow: Flow,
    /// maximal rate of this flow. Some flows are rate limited while others are limited by the host's speed
    max_rate: Bandwidth,
    /// below are states, mutated by the simulator
    ///
    /// upper bound to decide if a flow should converge
    speed_bound: f64,
    converged: bool,
    bytes_sent: f64,
    speed: f64, // bits/s
    pub(crate) route: Route,

    /// For rolling back
    speed_history: Vec<SpeedHistoryEntry>,
}

impl FlowState {
    fn new(ts: Timestamp, flow: Flow, max_rate: Bandwidth, route: Route) -> FlowStateRef {
        Rc::new(RefCell::new(FlowState {
            ts,
            flow,
            max_rate,
            speed_bound: 0.0,
            converged: false,
            bytes_sent: 0.0,
            speed: 0.0,
            route,
            speed_history: Vec::new(),
        }))
    }

    fn time_to_complete(&self) -> Duration {
        assert!(
            self.speed > 0.0,
            "speed: {}, speed_bound: {}",
            self.speed,
            self.speed_bound
        );
        let time_sec = (self.flow.bytes as f64 - self.bytes_sent) as f64 * 8.0 / self.speed;
        (time_sec * 1e9).ceil() as Duration
    }

    fn completed(&self) -> bool {
        // TODO(cjr): check the precision of this condition
        // self.bytes_sent >= self.flow.bytes
        self.bytes_sent >= self.flow.bytes as f64 || self.time_to_complete() == 0
    }

    fn is_loopback(&self) -> bool {
        self.route.path.is_empty()
    }
}

impl std::cmp::PartialEq for FlowState {
    fn eq(&self, other: &Self) -> bool {
        self.ts == other.ts
    }
}

impl Eq for FlowState {}

impl std::cmp::PartialOrd for FlowState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.ts.partial_cmp(&other.ts)
    }
}

impl std::cmp::Ord for FlowState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.ts.cmp(&other.ts)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize)]
pub struct TimerId(pub usize);

impl TimerId {
    pub fn new() -> TimerId {
        TimerId(TIMER_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

impl From<usize> for TimerId {
    fn from(val: usize) -> TimerId {
        TimerId(val)
    }
}

impl From<TimerId> for usize {
    fn from(val: TimerId) -> usize {
        val.0
    }
}

#[derive(Debug, Clone)]
pub enum Event {
    /// Application notifies the simulator with the arrival of a set of flows.
    FlowArrive(Vec<TraceRecord>),
    /// The application has completed all flows and has no more flows to send. This event should
    /// appear after certain FlowComplete event.
    AppFinish,
    /// A Timer event is registered by Application. It notifies the application after duration ns.
    /// Token is used to identify the timer.
    // RegisterTimer(Duration, Option<Token>, TimerId),
    AdapterRegisterTimer(Duration, Option<Token>, TimerId),
    /// This kind of events only faces to the user apps, and it will be
    /// translated into and from `AdapterRegisterTimer`.
    UserRegisterTimer(Duration, Option<Token>),
}

/// Iterator of Event
#[derive(Debug, Clone, Default)]
pub struct Events(SmallVec<[Event; 8]>);

impl Events {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn last(&self) -> Option<&Event> {
        self.0.last()
    }

    pub fn add(&mut self, e: Event) {
        self.0.push(e);
    }

    pub fn append(&mut self, mut e: Events) {
        self.0.append(&mut e.0);
    }

    pub fn reverse(&mut self) {
        self.0.reverse();
    }

    pub fn pop(&mut self) -> Option<Event> {
        self.0.pop()
    }
}

impl IntoIterator for Events {
    type Item = Event;
    type IntoIter = smallvec::IntoIter<[Self::Item; 8]>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl std::iter::FromIterator<Event> for Events {
    fn from_iter<T: IntoIterator<Item = Event>>(iter: T) -> Self {
        Events(iter.into_iter().collect())
    }
}

impl From<Event> for Events {
    fn from(e: Event) -> Self {
        Events(smallvec![e])
    }
}
