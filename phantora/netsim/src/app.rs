use fnv::FnvHashMap as HashMap;

use crate::simulator::{Event, Events, TimerId};
use crate::{Timestamp, Token, Trace, TraceRecord};

#[derive(Debug, Clone)]
pub struct AppEvent {
    pub ts: Timestamp,
    pub event: AppEventKind,
}

impl AppEvent {
    #[inline]
    pub fn new(ts: Timestamp, event: AppEventKind) -> Self {
        AppEvent { ts, event }
    }
}

#[derive(Debug, Clone)]
pub enum AppEventKind {
    /// On start, an application returns all flows it can start, whatever the timestamps are.
    AppStart,
    FlowComplete(Vec<TraceRecord>),
    /// Time up event triggered
    // Notification(Option<Token>, TimerId),
    AdapterNotification(Option<Token>, TimerId),
    UserNotification(Option<Token>),
}

/// An Applicatoin can interact with the simulator based on the flow completion event it received.
/// It dynamically start new flows according to the finish and arrive event of some flows.
pub trait Application: std::fmt::Debug {
    type Output;
    fn on_event(&mut self, event: AppEvent) -> Events;
    fn answer(&mut self) -> Self::Output;
}

/// A Replayer is an application that takes a trace and replay the trace.
#[derive(Debug)]
pub struct Replayer {
    trace: Trace,
    num_flows: usize,
    completed: Trace,
}

impl Application for Replayer {
    type Output = Trace;
    fn on_event(&mut self, event: AppEvent) -> Events {
        match event.event {
            AppEventKind::AppStart => Event::FlowArrive(self.trace.recs.clone()).into(),
            AppEventKind::FlowComplete(mut flows) => {
                self.completed.recs.append(&mut flows);
                if self.completed.recs.len() == self.num_flows {
                    (Event::AppFinish).into()
                } else {
                    Events::new()
                }
            }
            AppEventKind::UserNotification(_) | AppEventKind::AdapterNotification(..) => {
                Events::new()
            } // Replayer does not handle these events
        }
    }
    fn answer(&mut self) -> Self::Output {
        self.completed.clone()
    }
}

impl Replayer {
    pub fn new(trace: Trace) -> Self {
        let num_flows = trace.recs.len();
        Replayer {
            trace,
            num_flows,
            completed: Default::default(),
        }
    }
}

/// Sequence is a application combinator, it takes a sequence of applications.
/// Each application depends on previous one and only starts after the previous one is finished.
/// The result is a list of outputs of the applications with the order kepted.
#[derive(Default, Debug)]
pub struct Sequence<'a, T> {
    apps: Vec<Box<dyn Application<Output = T> + 'a>>,
    output: Vec<T>,
    cur_app: usize,
    start_time: Vec<Timestamp>,
}

impl<'a, T> Sequence<'a, T> {
    pub fn new() -> Self {
        Sequence {
            apps: Vec::new(),
            output: Vec::new(),
            cur_app: 0,
            start_time: vec![0], // the first app starts at time 0
        }
    }

    pub fn add(&mut self, app: Box<dyn Application<Output = T> + 'a>) {
        self.apps.push(app);
    }
}

impl<'a, T: Clone + std::fmt::Debug> Application for Sequence<'a, T> {
    type Output = Vec<T>;

    fn on_event(&mut self, event: AppEvent) -> Events {
        let now = event.ts;

        let app_start_time = self.start_time[self.cur_app];

        // hijack all timestamps
        let new_event = match event.event {
            AppEventKind::FlowComplete(mut flows) => {
                for f in &mut flows {
                    f.ts -= app_start_time;
                }
                AppEvent::new(now - app_start_time, AppEventKind::FlowComplete(flows))
            }
            AppEventKind::AppStart
            | AppEventKind::AdapterNotification(..)
            | AppEventKind::UserNotification(_) => AppEvent::new(now - app_start_time, event.event),
        };

        let events = self.apps[self.cur_app].on_event(new_event);
        log::trace!("Sequence sim_events: {:?}", events);

        // hijack timestamps of all flows
        events
            .into_iter()
            .flat_map(|sim_event| {
                match sim_event {
                    Event::AppFinish => {
                        self.output.push(self.apps[self.cur_app].answer());
                        // start the next app or finish
                        self.cur_app += 1;
                        if self.cur_app < self.apps.len() {
                            self.start_time.push(now);
                            self.apps[self.cur_app]
                                .on_event(AppEvent::new(0, AppEventKind::AppStart))
                        } else {
                            // we are ending here
                            (Event::AppFinish).into()
                        }
                    }
                    Event::FlowArrive(mut flows) => {
                        // becase all events in this flatmap are from the same app
                        // it is good to use the same app_start_time
                        for f in &mut flows {
                            f.ts += app_start_time;
                        }
                        Event::FlowArrive(flows).into()
                    }
                    Event::UserRegisterTimer(..) => sim_event.into(),
                    Event::AdapterRegisterTimer(..) => panic!("now allowed"),
                }
            })
            .collect()
    }

    fn answer(&mut self) -> Self::Output {
        self.output.clone()
    }
}

#[derive(Debug)]
enum TimerRequestor {
    Adapter,
    UserApp(Option<Token>), // user token
}

/// AppGroup is an application created by combining multiple applications started at different timestamps.
/// It marks flows from each individual application.
/// It works like a proxy, intercepting, modifying, and forwarding simulator events to and from corresponding apps.
#[derive(Default, Debug)]
pub struct AppGroup<'a, T> {
    // Vec<(start time offset, application)>
    apps: Vec<(Timestamp, Box<dyn Application<Output = T> + 'a>)>,
    output: Vec<(usize, T)>,
    stored_flow_token: HashMap<usize, Option<Token>>,
    stored_timer_token: HashMap<TimerId, TimerRequestor>,
}

pub trait AppGroupTokenCoding {
    fn encode(app_id: usize) -> Token;
    fn decode(&self) -> usize;
}

impl AppGroupTokenCoding for Token {
    fn encode(app_id: usize) -> Token {
        Token(app_id)
    }
    fn decode(&self) -> usize {
        self.0
    }
}

impl<'a, T> Application for AppGroup<'a, T>
where
    T: Clone + std::fmt::Debug,
{
    // a list of (app_id, App::Output)
    type Output = Vec<(usize, T)>;

    fn on_event(&mut self, event: AppEvent) -> Events {
        if self.output.len() == self.apps.len() {
            return (Event::AppFinish).into();
        }

        log::trace!("AppGroup receive an app_event {:?}", event);
        let events = match event.event {
            AppEventKind::AppStart => {
                // start all apps, modify the start time of flow, gather all flows
                // register notification to start apps
                (0..self.apps.len())
                    .map(|app_id| {
                        let start_off = self.apps[app_id].0;
                        // println!("start off {:?}", start_off);
                        let token = Token::encode(app_id);
                        let new_timer_id = TimerId::new();
                        self.stored_timer_token
                            .insert(new_timer_id, TimerRequestor::Adapter)
                            .ok_or(())
                            .unwrap_err();
                        Event::AdapterRegisterTimer(start_off, Some(token), new_timer_id)
                    })
                    .collect()
            }
            AppEventKind::FlowComplete(recs) => {
                // dispatch flows to different apps by looking at the token
                let mut flows = vec![vec![]; self.apps.len()];
                for r in recs {
                    let app_id = r.flow.token.unwrap().decode();
                    assert!(app_id < self.apps.len());
                    let mut f = r.clone();
                    f.ts -= self.apps[app_id].0; // f.ts -= start_off;
                    f.flow.token = self.stored_flow_token.remove(&r.flow.id).unwrap(); // restore the token
                    flows[app_id].push(f);
                }

                let ts = event.ts;
                flows
                    .into_iter()
                    .enumerate()
                    .map(|(app_id, recs)| {
                        if !recs.is_empty() {
                            self.forward(app_id, ts, AppEventKind::FlowComplete(recs))
                        } else {
                            Events::new()
                        }
                    })
                    .flatten()
                    .collect()
            }
            AppEventKind::AdapterNotification(token, timer_id) => {
                // now the timestamp is at self.apps[app_id].start_off
                log::trace!(
                    "AdapterNotification, token: {:?}, timer_id: {:?}",
                    token,
                    timer_id
                );
                match self.stored_timer_token[&timer_id] {
                    TimerRequestor::Adapter => {
                        let app_id = token.unwrap().decode();
                        self.forward(app_id, event.ts, AppEventKind::AppStart)
                    }
                    TimerRequestor::UserApp(user_token) => {
                        let app_id = token.unwrap().decode();
                        self.stored_timer_token.remove(&timer_id).unwrap();
                        log::trace!(
                            "app_id: {}, ts: {}, user_token: {:?}",
                            app_id,
                            event.ts,
                            user_token
                        );
                        self.forward(app_id, event.ts, AppEventKind::UserNotification(user_token))
                    }
                }
            }
            AppEventKind::UserNotification(..) => {
                panic!("impossible for an adapter to receive user notification");
            }
        };

        if self.output.len() == self.apps.len() {
            (Event::AppFinish).into()
        } else {
            events
        }
    }

    fn answer(&mut self) -> Self::Output {
        // it may not be in its original order
        self.output.clone()
    }
}

impl<'a, T> AppGroup<'a, T>
where
    T: Clone,
{
    pub fn new() -> Self {
        AppGroup {
            apps: Default::default(),
            output: Default::default(),
            stored_flow_token: HashMap::default(),
            stored_timer_token: HashMap::default(),
        }
    }

    pub fn add(
        &mut self,
        start_ts: Timestamp,
        app: Box<dyn Application<Output = T> + 'a>,
    ) -> usize {
        let app_id = self.apps.len();
        self.apps.push((start_ts, app));
        app_id
    }

    fn forward(&mut self, app_id: usize, ts: Timestamp, app_event_kind: AppEventKind) -> Events {
        let start_off = self.apps[app_id].0;
        let app_event = AppEvent::new(ts - start_off, app_event_kind);

        self.apps[app_id]
            .1
            .on_event(app_event)
            .into_iter()
            .flat_map(|event| {
                match event {
                    Event::FlowArrive(mut recs) => {
                        recs.iter_mut().for_each(|r| {
                            r.ts += start_off;
                            self.stored_flow_token
                                .insert(r.flow.id, r.flow.token)
                                .ok_or(())
                                .unwrap_err();
                            r.flow.token = Some(Token::encode(app_id));
                        });
                        Event::FlowArrive(recs).into()
                    }
                    Event::AppFinish => {
                        log::debug!("AppGroup, user app notifies AppFinish, app_id: {}", app_id);
                        let app = &mut self.apps[app_id].1;
                        self.output.push((app_id, app.answer()));
                        Events::new()
                    }
                    Event::AdapterRegisterTimer(..) => {
                        panic!("user not allowed to register timer as adapters");
                    }
                    Event::UserRegisterTimer(dura, token) => {
                        // panic!(
                        //     "impossible to receive timer registration from user app, token: {:?}",
                        //     token
                        // );
                        log::debug!(
                            "Event::UserRegisterTime, dura: {}, token: {:?}",
                            dura,
                            token
                        );
                        let new_token = Token::encode(app_id);
                        let new_timer_id = TimerId::new();
                        self.stored_timer_token
                            .insert(new_timer_id, TimerRequestor::UserApp(token))
                            .ok_or(())
                            .unwrap_err();
                        Event::AdapterRegisterTimer(dura, Some(new_token), new_timer_id).into()
                    }
                }
            })
            .collect()
    }
}
