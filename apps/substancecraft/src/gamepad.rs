//! Gamepad buttons for input bindings (A11Y-02). A background thread owns
//! the platform gamepad backend (gilrs) and forwards button presses to the
//! UI thread, waking egui so a press is handled even while nothing else
//! repaints. Without a backend (tests, or a platform gilrs cannot open) the
//! app simply sees no gamepad.

use std::collections::BTreeSet;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Weak};
use std::time::Duration;

use eframe::egui;
use serde::{Deserialize, Serialize};

/// A gamepad button, in the layout-neutral names of the standard gamepad
/// (South is A on Xbox pads and Cross on PlayStation pads).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GamepadButton {
    South,
    East,
    North,
    West,
    LeftBumper,
    RightBumper,
    LeftTrigger,
    RightTrigger,
    Select,
    Start,
    LeftStick,
    RightStick,
    DPadUp,
    DPadDown,
    DPadLeft,
    DPadRight,
}

impl GamepadButton {
    pub const ALL: [Self; 16] = [
        Self::South,
        Self::East,
        Self::North,
        Self::West,
        Self::LeftBumper,
        Self::RightBumper,
        Self::LeftTrigger,
        Self::RightTrigger,
        Self::Select,
        Self::Start,
        Self::LeftStick,
        Self::RightStick,
        Self::DPadUp,
        Self::DPadDown,
        Self::DPadLeft,
        Self::DPadRight,
    ];

    /// Stable name kept in the preferences file.
    pub fn name(self) -> &'static str {
        match self {
            Self::South => "South",
            Self::East => "East",
            Self::North => "North",
            Self::West => "West",
            Self::LeftBumper => "LeftBumper",
            Self::RightBumper => "RightBumper",
            Self::LeftTrigger => "LeftTrigger",
            Self::RightTrigger => "RightTrigger",
            Self::Select => "Select",
            Self::Start => "Start",
            Self::LeftStick => "LeftStick",
            Self::RightStick => "RightStick",
            Self::DPadUp => "DPadUp",
            Self::DPadDown => "DPadDown",
            Self::DPadLeft => "DPadLeft",
            Self::DPadRight => "DPadRight",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|button| button.name() == name)
    }

    /// Name shown in the bindings list.
    pub fn label(self) -> &'static str {
        match self {
            Self::South => "Pad South (A / Cross)",
            Self::East => "Pad East (B / Circle)",
            Self::North => "Pad North (Y / Triangle)",
            Self::West => "Pad West (X / Square)",
            Self::LeftBumper => "Pad left bumper",
            Self::RightBumper => "Pad right bumper",
            Self::LeftTrigger => "Pad left trigger",
            Self::RightTrigger => "Pad right trigger",
            Self::Select => "Pad Select",
            Self::Start => "Pad Start",
            Self::LeftStick => "Pad left stick press",
            Self::RightStick => "Pad right stick press",
            Self::DPadUp => "Pad d-pad up",
            Self::DPadDown => "Pad d-pad down",
            Self::DPadLeft => "Pad d-pad left",
            Self::DPadRight => "Pad d-pad right",
        }
    }

    fn from_gilrs(button: gilrs::Button) -> Option<Self> {
        use gilrs::Button;
        Some(match button {
            Button::South => Self::South,
            Button::East => Self::East,
            Button::North => Self::North,
            Button::West => Self::West,
            Button::LeftTrigger => Self::LeftBumper,
            Button::RightTrigger => Self::RightBumper,
            Button::LeftTrigger2 => Self::LeftTrigger,
            Button::RightTrigger2 => Self::RightTrigger,
            Button::Select => Self::Select,
            Button::Start => Self::Start,
            Button::LeftThumb => Self::LeftStick,
            Button::RightThumb => Self::RightStick,
            Button::DPadUp => Self::DPadUp,
            Button::DPadDown => Self::DPadDown,
            Button::DPadLeft => Self::DPadLeft,
            Button::DPadRight => Self::DPadRight,
            _ => return None,
        })
    }
}

/// What the backend thread reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadEvent {
    Pressed(GamepadButton),
    Released(GamepadButton),
    /// A pad went away: nothing on it is held any more.
    Disconnected,
}

/// Gamepad state as the UI thread sees it.
#[derive(Default)]
pub struct Gamepads {
    events: Option<Receiver<PadEvent>>,
    /// Dropped with the app, which tells the backend thread to end.
    _alive: Option<Arc<()>>,
    down: BTreeSet<GamepadButton>,
    /// Buttons pressed since the last [`Self::poll`], in press order.
    pressed: Vec<GamepadButton>,
}

impl Gamepads {
    /// Gamepads fed by `events` (the backend thread, or a test).
    pub fn from_events(events: Receiver<PadEvent>) -> Self {
        Self {
            events: Some(events),
            ..Self::default()
        }
    }

    /// Starts the platform backend on its own thread; it repaints `context`
    /// whenever a button changes.
    pub fn start(context: &egui::Context) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        let alive = Arc::new(());
        let watch = Arc::downgrade(&alive);
        let context = context.clone();
        let spawned = std::thread::Builder::new()
            .name("gamepads".to_owned())
            .spawn(move || run_backend(&sender, &watch, &context));
        match spawned {
            Ok(_) => Self {
                _alive: Some(alive),
                ..Self::from_events(receiver)
            },
            Err(_) => Self::default(),
        }
    }

    /// Takes the events that arrived since the last frame.
    pub fn poll(&mut self) {
        self.pressed.clear();
        let Some(events) = &self.events else {
            return;
        };
        loop {
            match events.try_recv() {
                Ok(PadEvent::Pressed(button)) => {
                    // Repeats while held are not new presses.
                    if self.down.insert(button) {
                        self.pressed.push(button);
                    }
                }
                Ok(PadEvent::Released(button)) => {
                    self.down.remove(&button);
                }
                Ok(PadEvent::Disconnected) => self.down.clear(),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.events = None;
                    self.down.clear();
                    break;
                }
            }
        }
    }

    /// Buttons first pressed this frame.
    pub fn pressed(&self) -> &[GamepadButton] {
        &self.pressed
    }

    pub fn is_down(&self, button: GamepadButton) -> bool {
        self.down.contains(&button)
    }
}

/// Forwards gilrs button events until the app goes away. A platform
/// without a usable backend ends the thread quietly.
fn run_backend(sender: &Sender<PadEvent>, alive: &Weak<()>, context: &egui::Context) {
    let Ok(mut gilrs) = gilrs::Gilrs::new() else {
        return;
    };
    while alive.strong_count() > 0 {
        // Wakes now and then even when idle, so a closed app ends the thread.
        let Some(event) = gilrs.next_event_blocking(Some(Duration::from_millis(500))) else {
            continue;
        };
        let forwarded = match event.event {
            gilrs::EventType::ButtonPressed(button, _) => {
                GamepadButton::from_gilrs(button).map(PadEvent::Pressed)
            }
            gilrs::EventType::ButtonReleased(button, _) => {
                GamepadButton::from_gilrs(button).map(PadEvent::Released)
            }
            gilrs::EventType::Disconnected => Some(PadEvent::Disconnected),
            _ => None,
        };
        if let Some(forwarded) = forwarded {
            if sender.send(forwarded).is_err() {
                return;
            }
            context.request_repaint();
        }
    }
}
