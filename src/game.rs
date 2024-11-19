use bevy::{ecs::schedule::ScheduleLabel, prelude::*};
use bevy_rapier2d::prelude::*;

pub const FPS: f32 = 50.0;
pub const SCALE: f32 = 30.0; // Affects how fast-paced the game is, forces should be adjusted as well

pub const MAIN_ENGINE_POWER: f32 = 13.0;
pub const SIDE_ENGINE_POWER: f32 = 0.6;

pub const INITIAL_RANDOM: f32 = 1000.0; // Set 1500 to make game harder

pub const LANDER_POLY: [Vec2; 6] = [
    Vec2::new(-14.0, 17.0),  // Left Up
    Vec2::new(-17.0, 0.0),   // Left Central
    Vec2::new(-17.0, -10.0), // Left Lower
    Vec2::new(17.0, -10.0),  // Right Lower
    Vec2::new(17.0, 0.0),    // Right Central
    Vec2::new(14.0, 17.0),   // Right Upper
];
pub const LEG_AWAY: f32 = 20.0; // Horizontal distance off center
pub const LEG_DOWN: f32 = 18.0; // Vertical distance off center
pub const LEG_W: f32 = 2.0; // Leg width
pub const LEG_H: f32 = 8.0; // Leg height
pub const LEG_SPRING_TORQUE: f32 = 40.0;

pub const SIDE_ENGINE_HEIGHT: f32 = 14.0;
pub const SIDE_ENGINE_AWAY: f32 = 12.0; // Horizontal distance off center

pub const VIEWPORT_W: f32 = 600.0; // Width of the window
pub const VIEWPORT_H: f32 = 400.0; // Height of the window

pub struct State([f32; Self::SIZE]);
impl State {
    pub const SIZE: usize = 8;

    pub fn position_x(&self) -> f32 {
        self.0[0]
    }
    pub fn position_y(&self) -> f32 {
        self.0[1]
    }
    pub fn velocity_x(&self) -> f32 {
        self.0[2]
    }
    pub fn velocity_y(&self) -> f32 {
        self.0[3]
    }
    pub fn orientation_angle(&self) -> f32 {
        self.0[4]
    }
    pub fn angular_velocity(&self) -> f32 {
        self.0[5]
    }
    pub fn left_arm_contact(&self) -> f32 {
        self.0[6]
    }
    pub fn right_arm_contact(&self) -> f32 {
        self.0[7]
    }

    pub fn is_left_arm_contact(&self) -> bool {
        self.left_arm_contact() == 1.0
    }
    pub fn is_right_arm_contact(&self) -> bool {
        self.right_arm_contact() == 1.0
    }
}

pub enum PlayStepAction {
    Nothing,
    ThrusterLeft,
    ThrusterRight,
    ThrusterMain,
}
impl PlayStepAction {
    pub fn from_index(action_index: u8) -> Self {
        match action_index {
            0 => Self::Nothing,
            1 => Self::ThrusterLeft,
            2 => Self::ThrusterRight,
            3 => Self::ThrusterMain,
            _ => panic!("Index out of bounds: {action_index}"),
        }
    }

    pub fn to_index(&self) -> u8 {
        match self {
            Self::Nothing => 0,
            Self::ThrusterLeft => 1,
            Self::ThrusterRight => 2,
            Self::ThrusterMain => 3,
        }
    }
}

/// Next state, reward, done.
///
/// Next state: next state of the environment.
/// Reward: reward for performing the action.
/// Done: true if the episode ended.
pub type PlayStepResult = (State, f32, bool);

/// Wind effects applied to lander.
pub struct Wind {
    power: f32,
    turbulence_power: f32,
}
impl Default for Wind {
    fn default() -> Self {
        Self {
            power: 15.0,
            turbulence_power: 1.5,
        }
    }
}

#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
pub struct GameStepSchedule;

#[derive(Resource)]
pub struct GameUpdater {
    timer: Timer,
}

pub struct GamePlugin {
    gravity: f32,
    enable_wind: Option<Wind>,
}
impl Default for GamePlugin {
    fn default() -> Self {
        Self {
            gravity: -10.0,
            enable_wind: None,
        }
    }
}
impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(RapierConfiguration {
            gravity: Vec2::new(0.0, self.gravity),
            timestep_mode: TimestepMode::Fixed {
                dt: 1.0 / 60.0,
                substeps: 1,
            },
            ..RapierConfiguration::new(1.0)
        });
        app.add_plugins(
            RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0)
                .in_schedule(GameStepSchedule),
        );
        app.add_plugins(RapierDebugRenderPlugin::default());

        app.insert_resource(GameUpdater {
            timer: Timer::from_seconds(1.0 / FPS, TimerMode::Repeating),
        });
        app.add_systems(Update, game_updater);
    }
}

fn game_updater(mut commands: Commands, time: Res<Time>, mut updater: ResMut<GameUpdater>) {
    updater.timer.tick(time.delta());

    while updater.timer.finished() {
        commands.add(|world: &mut World| {
            world.run_schedule(GameStepSchedule);
        });
        updater.timer.reset();
    }
}
