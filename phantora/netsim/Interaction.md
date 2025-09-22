# The Interaction Between Phantora and netsim 

## netsim abstraction
netsim provides two important abstraction and a bunch of implementations and helper utilities around them.

### Simulator
netsim is an event-based simulator (who else doesn't?). It takes a cluster network configuration, a flow trace or a composible `dyn Application` that dynamically generates new flows, maintains the network state (mainly the flow state), and computes when the next event should happen. It solves the problem of flow sharing within the constraint of max-min-fairness model. The simulator implements the `Executor` trait.

### Executor
An `trait Executor` is how we drive the simulator. Taking an `Box<dyn Application>` as the input argument, an `Executor` provides APIs to run the network simulation to the end or to run the simulation until the next event (we will mention how to proceed the simuation for a certain duration or to advance the timestamp for a delta). If it runs to completion, the API returns the output for this simulation (e.g., the completion time for all the flows). If it runs one step, the simulator will solve the next first event that the application should know and return it (e.g., a flow completes, an timer registered by the app expires). Steping forward one step at a time, the App developer will be responsible for keep calling the API in a loop to finish the simulation.
```rust
pub trait Executor<'a> {
  fn run_to_completion<T>(&mut self, mut app: Box<dyn Application<Output = T> + 'a>) -> T;
  fn on_event(&mut self, event: Event) -> bool;
  fn run_one_step(&mut self) -> AppEvent;
}
```

### App
An `trait Application { type Output; }` is an application that takes __notification events__ (i.e., an `AppEvent`) from the simulator as input (such as `AppEventKind::AppStart` or `AppEventKind::FlowComplete`), and generates a simulator event (in the code it is called an `Event` for convenience) that instructs the next step of the simulator. Once complete, the `Application` should also provide an API to provide an answer to this execution.

To give an example, an `Replayer` implements `Application` that initially takes a flow trace as input. When receiving an `AppEvent`, it organizes all the flows in the trace as an event of `EventKind::FlowArriave(Vec<TraceRecord>)` and returns it to the simulator. The simulator in turn calls into the `Application::on_event` and pass the first ready event
```rust
pub trait Application {
  type Output;
  fn on_event(&mut self, event: AppEvent) -> Events;
  fn answer(&mut self) -> Application::Output;
}
```

An `Application` is also composible. Our library provides some example combinator like `Sequence` and `AppGroup`. A `Sequence` app maintains a list of applications (added via `seqapp.add()`) and starts the next one only the previous one has finished. An `AppGroup` finishes when all the apps of it finish.

For netsim to interact with the outside world, there are mainly two approach. One way is to implement the `Application` trait where new flows can be generated based on the completion of previous flows. However, this approach is limited in some cases. netsim runs in an main loop, so it is required for netsim simulator to take control of the execution at some point. For cases where this is inconvenient, e.g., the app itself has a mainloop, the user can implement their own interaction logic with the simualtor.