use crate::simulator::{CommMeta, ComputeMeta, NodeId};
use crate::timeline;
use netsim::simulator::{Executor, Simulator as NetSim};
use netsim::{Flow, TraceRecord};
use priority_queue::PriorityQueue;
use serde::{Deserialize, Serialize};
use std::cmp::{max, Reverse};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use timeline::Timeline;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct EventId(u64);

impl Debug for EventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Display for EventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

#[derive(PartialEq, Eq, Hash)]
enum Event {
    Start(EventId),
    End(EventId),
    Point(EventId),
}

enum PushEvent {
    Start(EventId, Action),
    End(EventId),
    Point(EventId),
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize, Deserialize)]
pub enum Action {
    Computation(NodeId, i64 /* computation time */, ComputeMeta),
    Communication(Flow, CommMeta),
}

pub struct StartedRecord {
    pub start_time: i64,
}

struct Pending {
    dependencies: HashSet<Option<EventId>>,
    arrive_time: i64,
    dep_end_time: i64,
    action: Option<Action>,
}

pub struct EventQueue {
    netsim: NetSim,
    event_counter: u64,

    event_to_flow: HashMap<EventId, Flow>,

    pending: HashMap<EventId, Pending>,
    in_queue_not_started: HashMap<EventId, Action>,
    started: HashMap<EventId, StartedRecord>,
    ended: HashMap<EventId, (i64, i64)>,
    rev_dependencies: HashMap<EventId, HashSet<EventId>>,

    event_queue: PriorityQueue<Event, Reverse<(i64, usize)>>,
    queue_counter: usize,

    timeline: Option<Timeline>,
}

pub enum QueueStep {
    /// Simulation ends
    EmptyQueue,
    /// An span event (event_id, event_time) started
    EventStarted(EventId, i64),
    /// An span event (event_id, event_time) ended
    EventEnded(EventId, i64),
    /// An point event (event_id, event_time) has reached
    ReachedPoint(EventId, i64),
}

impl EventQueue {
    pub fn new(netsim: NetSim) -> EventQueue {
        let timeline = crate::args::get_args().timeline_file.as_ref().map(|f| {
            Timeline::new(f).unwrap_or_else(|e| {
                panic!(
                    "Create timeline file from {} failed, err: {}",
                    f.to_string_lossy(),
                    e
                )
            })
        });
        EventQueue {
            netsim,
            event_counter: 0,
            event_to_flow: HashMap::new(),
            pending: HashMap::new(),
            in_queue_not_started: HashMap::new(),
            started: HashMap::new(),
            ended: HashMap::new(),
            rev_dependencies: HashMap::new(),
            event_queue: PriorityQueue::new(),
            queue_counter: 0,
            timeline,
        }
    }

    fn record_timeline_action(&mut self, id: EventId, action: Action) {
        // Only save the action in memory when timeline is enabled
        // Action will appear only once in the timeline
        // An action may not have associated span either due to the span has a start_time == end_time,
        // or due to implementation error.
        if let Some(timeline) = self.timeline.as_mut() {
            timeline
                .write(&timeline::Record::new_action(id, action))
                .unwrap();
        }
    }

    fn record_timeline_span(&mut self, id: EventId, start_time: i64, end_time: i64) {
        if let Some(timeline) = self.timeline.as_mut() {
            // A span of the save EventId can appear many times in the timeline
            // Only the last appearance is valid.
            // if !timeline.is_action(id) {
            //     return;
            // }
            timeline
                .write(&timeline::Record::new_span(id, start_time, end_time))
                .unwrap();
        }
    }

    fn push_event(&mut self, event: PushEvent, time: i64) {
        let prio = Reverse((time, self.queue_counter));
        self.queue_counter += 1;
        match event {
            PushEvent::End(id) => {
                self.event_queue.push(Event::End(id), prio);
            }
            PushEvent::Point(id) => {
                log::debug!("{}: Adding Point {}", time, id);
                self.event_queue.push(Event::Point(id), prio);
            }

            PushEvent::Start(id, action @ Action::Computation(..)) => {
                log::debug!("{}: Adding {:?} {}", time, action, id);
                self.event_queue.push(Event::Start(id), prio);
                self.in_queue_not_started.insert(id, action.clone());
                self.record_timeline_action(id, action);
            }
            PushEvent::Start(id, Action::Communication(mut flow, meta)) => {
                // Add a new flow to the network simulator dynamically
                assert!(flow.token.is_none());
                // update the actual start time and associate it with the event_id
                let mut start_ts: netsim::Timestamp = time.try_into().unwrap();
                start_ts *= 1000; // us to ns
                flow.token = Some(netsim::Token(id.0 as _));
                let rec = TraceRecord::new(start_ts, flow.clone(), None);
                log::debug!("{}: Adding {:?} {}", time, rec, id);

                self.record_timeline_action(id, Action::Communication(flow.clone(), meta.clone()));

                let on_event_result = self.netsim.on_event(netsim::Event::FlowArrive(vec![rec]));

                use netsim::simulator::OnEventResult;
                match on_event_result {
                    OnEventResult::Ok | OnEventResult::SimulationFinished => {
                        // we don't really care if it is the last flow in the network simluator
                        // we may add new flows to it later
                        self.event_queue.push(Event::Start(id), prio);
                        self.in_queue_not_started
                            .insert(id, Action::Communication(flow, meta));
                    }
                    OnEventResult::Rollback(rollbacked_flows) => {
                        log::debug!(
                            "Rollbacked flows: {:?}",
                            rollbacked_flows
                                .iter()
                                .map(|f| f.flow.token)
                                .collect::<Vec<_>>()
                        );
                        let mut rollbacked_flowset: HashSet<_> =
                            rollbacked_flows.into_iter().map(|rec| rec.flow).collect();
                        let mut new_flow_ended = false;
                        while !rollbacked_flowset.is_empty() {
                            let (app_event_ts, finished_flows) = self.netsim_run_one_step(None);
                            let end_time = (app_event_ts / 1000) as i64;
                            for r in finished_flows {
                                let event = EventId(r.flow.token.unwrap().0 as _);
                                rollbacked_flowset.remove(&r.flow);
                                if r.flow == flow {
                                    log::trace!("r.flow == rec.flow, flow: {:?}", r.flow);
                                    // the flow just added is finished
                                    new_flow_ended = true;
                                    self.end_event(id, time, end_time);
                                } else if rollbacked_flowset.contains(&r.flow) {
                                    self.add_end_time_to(event, end_time);
                                } else {
                                    assert_eq!(app_event_ts, r.ts + r.dura.unwrap());
                                    log::debug!("{}: Adding End {} at {}", time, event, end_time);
                                    if let Some(overwritten) = self.started.insert(
                                        event,
                                        StartedRecord {
                                            start_time: r.ts as i64 / 1000,
                                        },
                                    ) {
                                        assert!(
                                            overwritten.start_time == r.ts as i64 / 1000,
                                            "{} vs {}",
                                            overwritten.start_time,
                                            r.ts
                                        );
                                    }
                                    self.push_event(PushEvent::End(event), end_time);
                                }
                            }
                        }
                        if !new_flow_ended {
                            self.event_queue.push(Event::Start(id), prio);
                            self.in_queue_not_started
                                .insert(id, Action::Communication(flow, meta));
                        }
                    }
                }
            }
        }
    }

    fn pop_event(&mut self) -> Option<(Event, i64)> {
        match self.event_queue.pop() {
            None => None,
            Some((event, Reverse((time, _)))) => Some((event, time)),
        }
    }

    pub fn peek_next_time(&mut self) -> Option<i64> {
        loop {
            match self.event_queue.peek() {
                Some((_, Reverse((time, _)))) => break Some(*time),
                None => {
                    if !self.try_proceed_netsim("peek_next_time") {
                        break None;
                    }
                }
            }
        }
    }

    fn netsim_run_one_step(&mut self, until: Option<i64>) -> (u64, Vec<TraceRecord>) {
        let app_event = self.netsim.run_one_step(until.map(|x| (x * 1000) as _));
        match app_event.event {
            netsim::app::AppEventKind::FlowComplete(finished_flows) => {
                (app_event.ts, finished_flows)
            }
            app_event => panic!("unexpected app_event {:?}", app_event),
        }
    }

    fn end_event(&mut self, id: EventId, start_time: i64, end_time: i64) {
        self.ended.insert(id, (start_time, end_time));
        self.record_timeline_span(id, start_time, end_time);

        let mut no_dep_events = Vec::new();
        if let Some(rev_deps) = self.rev_dependencies.get(&id) {
            for e in rev_deps {
                if let Some(pending) = self.pending.get_mut(e) {
                    if pending.dependencies.remove(&Some(id)) {
                        pending.dep_end_time = max(pending.dep_end_time, end_time);
                        if pending.dependencies.is_empty() {
                            no_dep_events.push(*e);
                        }
                    }
                }
            }
        }
        for e in no_dep_events {
            let pending = self.pending.remove(&e).unwrap();
            let start_time = max(pending.dep_end_time, pending.arrive_time);
            match pending.action {
                None => self.push_event(PushEvent::Point(e), start_time),
                Some(action) => {
                    self.push_event(PushEvent::Start(e, action), start_time);
                }
            }
        }
    }

    fn try_proceed_netsim(&mut self, info: &str) -> bool {
        let (app_event_ts, finished_flows) = self.netsim_run_one_step(None);
        log::trace!("{}: netsim.ts: {}", info, app_event_ts);

        if finished_flows.is_empty() {
            return false;
        }

        // handle finished flows
        for r in finished_flows {
            let flow_event_id = EventId(r.flow.token.unwrap().0 as _);
            assert_eq!(app_event_ts, r.ts + r.dura.unwrap());
            let comm_end_time = (app_event_ts / 1000) as _; // ns to us
            log::debug!(
                "{}: Adding End {} at {}",
                info,
                flow_event_id,
                comm_end_time
            );
            self.push_event(PushEvent::End(flow_event_id), comm_end_time);
        }

        return true;
    }

    pub fn execute(&mut self) -> QueueStep {
        loop {
            match self.pop_event() {
                None => {
                    if !self.try_proceed_netsim("QueueStep::EmptyQueue") {
                        break QueueStep::EmptyQueue;
                    }
                }
                Some((event, event_time)) => match event {
                    Event::Point(id) => {
                        self.end_event(id, event_time, event_time);
                        log::debug!("{}: Reached point {}", event_time, id);
                        break QueueStep::ReachedPoint(id, event_time);
                    }
                    Event::End(id) => {
                        let started_record = self.started.remove(&id).unwrap();
                        self.end_event(id, started_record.start_time, event_time);
                        log::debug!(
                            "{}: Event {} ended (started at {})",
                            event_time,
                            id,
                            started_record.start_time
                        );
                        break QueueStep::EventEnded(id, event_time);
                    }
                    Event::Start(id) => {
                        let action = self.in_queue_not_started.remove(&id).unwrap();
                        match &action {
                            Action::Computation(_, compute_duration, _meta) => {
                                // For computation, we immediately can estimate its finish time.
                                let compute_end_time = event_time + *compute_duration;
                                log::debug!(
                                    "{}: Adding End {} at {}",
                                    event_time,
                                    id,
                                    compute_end_time
                                );
                                self.push_event(PushEvent::End(id), compute_end_time);
                            }
                            Action::Communication(..) => {
                                // simulator run to next flow completion, could be multiple flows complete
                                // at the same time
                                // let (app_event_ts, finished_flows) = self.netsim_run_one_step(self.peek_next_time());
                            }
                        };
                        log::debug!("{}: Event {} started, {:?}", event_time, id, action);
                        self.started.insert(
                            id,
                            StartedRecord {
                                start_time: event_time,
                            },
                        );
                        break QueueStep::EventStarted(id, event_time);
                    }
                },
            }
        }
    }

    pub fn next_event_id(&self) -> EventId {
        EventId(self.event_counter)
    }

    pub fn add_action_or_point(
        &mut self,
        depends_on: Vec<Option<EventId>>,
        action: Option<Action>,
        mut current_time: i64,
    ) -> EventId {
        let id = EventId(self.event_counter);
        self.event_counter += 1;

        if let Some(Action::Communication(flow, _meta)) = &action {
            self.event_to_flow.insert(id, flow.clone());
        };

        let mut depends = HashSet::new();
        for dep in &depends_on {
            if let Some(dep) = dep {
                match self.ended.get(dep) {
                    None => {
                        depends.insert(Some(*dep));
                    }
                    Some((_, dep_end_time)) => current_time = max(current_time, *dep_end_time),
                }
                self.rev_dependencies
                    .entry(*dep)
                    .or_insert_with(HashSet::new)
                    .insert(id);
            } else {
                depends.insert(*dep);
            }
        }

        if depends.is_empty() {
            match action {
                Some(action) => {
                    self.push_event(PushEvent::Start(id, action), current_time);
                }
                None => {
                    self.push_event(PushEvent::Point(id), current_time);
                }
            }
        } else {
            log::debug!("{}: Pending {:?} {}", current_time, action, id);
            log::debug!("{}: depends: {:?}", current_time, depends);
            self.pending.insert(
                id,
                Pending {
                    dependencies: depends,
                    arrive_time: current_time,
                    dep_end_time: 0,
                    action,
                },
            );
        }
        id
    }

    pub fn add_dependency(&mut self, id: EventId, dep: Option<EventId>) {
        log::trace!("add_dependency, id: {id}, dep: {dep:?}");
        if let Some(dep) = dep {
            self.rev_dependencies
                .entry(dep)
                .or_insert_with(HashSet::new)
                .insert(id);
            if self.ended.contains_key(&dep) {
                return;
            }
        }
        assert!(self.pending.contains_key(&id));
        self.pending.entry(id).and_modify(|pending| {
            pending.dependencies.insert(dep);
        });
    }

    pub fn remove_none_dependency(&mut self, id: EventId) {
        log::trace!("remove_none_dependency, id: {id}");
        let mut no_dep = false;
        self.pending.entry(id).and_modify(|pending| {
            pending.dependencies.remove(&None);
            no_dep = pending.dependencies.is_empty();
        });

        if no_dep {
            log::trace!("remove_none_dependency, no_dep, id: {id}");
            let pending = self.pending.remove(&id).unwrap();
            let start_time = max(pending.dep_end_time, pending.arrive_time);
            match pending.action {
                None => self.push_event(PushEvent::Point(id), start_time),
                Some(action) => {
                    self.push_event(PushEvent::Start(id, action), start_time);
                }
            }
        }
    }

    pub fn add_end_time_to(&mut self, id: EventId, new_end_time: i64) {
        let is_executing_action = self
            .event_queue
            .change_priority_by(&Event::End(id), |Reverse((time, _))| *time = new_end_time);
        let is_unreached_point = self
            .event_queue
            .change_priority_by(&Event::Point(id), |Reverse((time, _))| *time = new_end_time);

        if !(is_executing_action || is_unreached_point) {
            self.ended
                .get_mut(&id)
                .map(|(start_time, end_time)| {
                    *end_time = new_end_time;
                    (*start_time, *end_time)
                })
                .map(|(s, e)| self.record_timeline_span(id, s, e));

            if let Some(rev_deps) = self.rev_dependencies.get(&id) {
                let rev_deps: Vec<_> = rev_deps.iter().cloned().collect();
                for e in rev_deps {
                    self.change_start_time_notless(e, new_end_time);
                }
            }
        }
    }

    fn change_start_time_notless(&mut self, id: EventId, new_start_time: i64) {
        match self.pending.get_mut(&id) {
            Some(pending) => pending.dep_end_time = max(pending.dep_end_time, new_start_time),
            None => {
                if let Some(flow) = self.event_to_flow.get(&id) {
                    self.netsim
                        .update_flow_start_ts(flow, (new_start_time * 1000) as _)
                }

                let is_in_queue_not_started = self
                    .event_queue
                    .change_priority_by(&Event::Start(id), |Reverse((time, _))| {
                        *time = max(*time, new_start_time)
                    });
                let is_unreached_point = self
                    .event_queue
                    .change_priority_by(&Event::Point(id), |Reverse((time, _))| {
                        *time = max(*time, new_start_time)
                    });

                if !(is_in_queue_not_started || is_unreached_point) {
                    match self.started.get_mut(&id) {
                        Some(started_record) => {
                            if started_record.start_time < new_start_time {
                                started_record.start_time = new_start_time;
                                let found = self.event_queue.change_priority_by(
                                    &Event::End(id),
                                    |Reverse((time, _))| {
                                        *time += new_start_time - started_record.start_time
                                    },
                                );
                                assert!(found, "id: {}", id);
                            }
                        }
                        None => match self.ended.get_mut(&id) {
                            Some((start_time, end_time)) => {
                                if *start_time < new_start_time {
                                    let delta = new_start_time - *start_time;
                                    *start_time = new_start_time;
                                    *end_time += delta;
                                    let new_end_time = *end_time;
                                    self.record_timeline_span(id, new_start_time, new_end_time);

                                    if let Some(rev_deps) = self.rev_dependencies.get(&id) {
                                        let rev_deps: Vec<_> = rev_deps.iter().cloned().collect();
                                        for e in rev_deps {
                                            self.change_start_time_notless(e, new_end_time);
                                        }
                                    }
                                }
                            }
                            None => (),
                        },
                    }
                }
            }
        }
    }

    pub fn query(&self, id: &EventId) -> Option<i64> {
        match self.ended.get(id) {
            Some((_, end_time)) => Some(*end_time),
            None => None,
        }
    }
}
