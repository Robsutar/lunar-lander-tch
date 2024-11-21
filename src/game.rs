use bevy::{
    app::MainScheduleOrder,
    ecs::schedule::{ScheduleBuildSettings, ScheduleLabel},
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
    },
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};
use bevy_rapier2d::prelude::*;
use rand::Rng;

use crate::util::extract_2d_angle;

pub const FPS: f32 = 50.0;
pub const SCALE: f32 = 30.0; // Affects how fast-paced the game is, forces should be adjusted as well

pub const MAIN_ENGINE_POWER: f32 = 13.0;
pub const SIDE_ENGINE_POWER: f32 = 0.6;

pub const INITIAL_RANDOM: f32 = 1000.0; // Set 1500 to make game harder

pub const LANDER_POLY: [Vec2; 6] = [
    Vec2::new(-14.0, 17.0),  // Left Upper
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

#[derive(Clone)]
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
    pub fn left_leg_contact(&self) -> f32 {
        self.0[6]
    }
    pub fn right_leg_contact(&self) -> f32 {
        self.0[7]
    }

    pub fn is_left_leg_contact(&self) -> bool {
        self.left_leg_contact() == 1.0
    }
    pub fn is_right_leg_contact(&self) -> bool {
        self.right_leg_contact() == 1.0
    }
}

#[derive(Event)]
pub struct GameInitEvent {
    pub initial_state: State,
}

#[derive(Event)]
pub struct GameResetEvent {
    pub initial_state: State,
}

#[derive(Event)]
pub enum StepActionEvent {
    Nothing,
    ThrusterLeft,
    ThrusterRight,
    ThrusterMain,
}
impl StepActionEvent {
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

    pub fn to_force(
        &self,
        center_rotation: Quat,
        main_engine_power: f32,
        side_engine_power: f32,
    ) -> ExternalImpulse {
        match self {
            StepActionEvent::Nothing => ExternalImpulse {
                impulse: Vec2::ZERO,
                torque_impulse: 0.0,
            },
            StepActionEvent::ThrusterLeft => ExternalImpulse {
                impulse: (center_rotation * Vec3::new(side_engine_power, 0.0, 0.0)).truncate(),
                torque_impulse: 0.0,
            },
            StepActionEvent::ThrusterRight => ExternalImpulse {
                impulse: (center_rotation * Vec3::new(-side_engine_power, 0.0, 0.0)).truncate(),
                torque_impulse: 0.0,
            },
            StepActionEvent::ThrusterMain => ExternalImpulse {
                impulse: (center_rotation * Vec3::new(0.0, main_engine_power, 0.0)).truncate(),
                torque_impulse: 0.0,
            },
        }
    }
}

#[derive(Event, Clone)]
pub struct StepResultEvent {
    /// Next state of the environment, after the action.
    next_state: State,
    /// Reward for performing the action.
    reward: f32,
    /// True if the episode ended.
    done: bool,
}
impl StepResultEvent {
    pub fn unpack(self) -> (State, f32, bool) {
        (self.next_state, self.reward, self.done)
    }
}

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

#[derive(Component)]
pub struct LanderCenter;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LegState {
    InAir,
    InGround,
}

#[derive(Component)]
pub struct Game {
    pub center_id: Entity,
    pub leg_ids: [Entity; 2],
    pub ground_id: Entity,

    pub helipad_y: f32,

    pub wind: Option<Wind>,

    pub frame: usize,

    pub game_over: Option<StepResultEvent>,
}
impl Game {
    pub fn reset(commands: &mut Commands) {
        commands.add(|world: &mut World| {
            world.run_schedule(GameResetSchedule);
        })
    }

    pub fn play_step(
        commands: &mut Commands,
        ev_step_action: &mut EventWriter<StepActionEvent>,
        action: StepActionEvent,
    ) {
        ev_step_action.send(action);
        commands.add(|world: &mut World| {
            world.run_schedule(PreGameStepSchedule);
        })
    }
}

#[derive(Resource)]
pub struct GameUpdater {
    timer: Timer,
}

#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
pub struct PostGameInitSchedule;

#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
struct GameResetSchedule;
#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
pub struct PostGameResetSchedule;

#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
pub struct AvailableUpdateSchedule;

#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
struct PreGameStepSchedule;
#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
struct PhysicsStepSchedule;
#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
struct PostPhysicsStepSchedule;
#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
pub struct PostGameStepSchedule;

pub struct GamePlugin {
    gravity: f32,
}
impl Default for GamePlugin {
    fn default() -> Self {
        Self { gravity: -10.0 }
    }
}
impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.init_schedule(PostGameInitSchedule);

        app.init_schedule(GameResetSchedule);
        app.init_schedule(PostGameResetSchedule);

        app.init_schedule(AvailableUpdateSchedule);

        app.init_schedule(PreGameStepSchedule);
        app.init_schedule(PhysicsStepSchedule);
        app.init_schedule(PostPhysicsStepSchedule);
        app.init_schedule(PostGameStepSchedule);

        app.insert_resource(GameUpdater {
            timer: Timer::from_seconds(1.0 / FPS, TimerMode::Repeating),
        });
        app.insert_resource(RapierConfiguration {
            gravity: Vec2::new(0.0, self.gravity),
            timestep_mode: TimestepMode::Fixed {
                dt: 1.0 / 60.0,
                substeps: SCALE as usize,
            },
            ..RapierConfiguration::new(1.0)
        });
        app.add_plugins(
            RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(1.0 / SCALE)
                .in_schedule(PhysicsStepSchedule),
        );
        app.add_plugins(RapierDebugRenderPlugin::default());

        app.add_event::<GameInitEvent>();
        app.add_event::<GameResetEvent>();
        app.add_event::<StepActionEvent>();
        app.add_event::<StepResultEvent>();

        app.add_systems(PostStartup, game_init);

        app.add_systems(Update, update_available_schedule);
        app.add_systems(PreGameStepSchedule, game_pre_update);
        app.add_systems(PostPhysicsStepSchedule, game_post_physics_update);
    }
}

fn game_init(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut ev_init: EventWriter<GameInitEvent>,
) {
    // Create assets
    let center_pbr = {
        let positions: Vec<[f32; 3]> = LANDER_POLY
            .iter()
            .map(|p| [p.x / SCALE, p.y / SCALE, 0.0])
            .collect();
        let indices = vec![
            0, 1, 2, // First triangle
            0, 2, 3, // Second triangle
            0, 3, 4, // Third triangle
            0, 4, 5, // Fourth triangle
        ];

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_indices(Indices::U32(indices));

        (
            Mesh2dHandle(meshes.add(mesh)),
            materials.add(Color::srgb_u8(128, 102, 230)),
        )
    };

    let leg_pbr = {
        let mut mesh: Mesh = Cuboid::new(LEG_W * 2.0 / SCALE, LEG_H * 2.0 / SCALE, 0.0).into();
        mesh.transform_by(Transform::from_xyz(0.0, -LEG_H / SCALE, 0.0));

        (
            Mesh2dHandle(meshes.add(mesh)),
            materials.add(Color::srgb_u8(77, 77, 128)),
        )
    };

    let flag_pbr = {
        let mut mesh: Mesh = Triangle2d::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, -10.0 / SCALE),
            Vec2::new(25.0 / SCALE, -5.0 / SCALE),
        )
        .into();
        mesh.transform_by(Transform::from_xyz(0.0, 50.0 / SCALE, 0.0));

        (
            Mesh2dHandle(meshes.add(mesh)),
            materials.add(Color::srgb_u8(204, 204, 0)),
        )
    };

    let flag_handle_pbr = {
        let mut mesh: Mesh = Cuboid::new(1.0 / SCALE, 50.0 / SCALE, 0.0).into();
        mesh.transform_by(Transform::from_xyz(0.0, 25.0 / SCALE, 0.0));

        (
            Mesh2dHandle(meshes.add(mesh)),
            materials.add(Color::srgb_u8(77, 77, 128)),
        )
    };

    let ground_material = materials.add(Color::WHITE);

    let mut rng = rand::thread_rng();

    let w: f32 = VIEWPORT_W / SCALE;
    let h = VIEWPORT_H / SCALE;
    let chunks = 11;
    let helipad_y = h / 4.0;

    let terrain_poly = {
        // Create the terrain.
        let chunk_x: Vec<f32> = (0..chunks)
            .map(|i| w / (chunks as f32 - 1.0) * i as f32)
            .collect();

        let mut height: Vec<f32> = (0..chunks).map(|_| rng.gen_range(0.0..h / 2.0)).collect();

        // Helipad flag place.
        height[chunks / 2 - 2] = helipad_y;
        height[chunks / 2 - 1] = helipad_y;
        height[chunks / 2 + 0] = helipad_y;
        height[chunks / 2 + 1] = helipad_y;
        height[chunks / 2 + 2] = helipad_y;

        height.insert(0, helipad_y);
        height.insert(height.len() - 1, helipad_y);

        let smooth_y: Vec<f32> = (1..=chunks)
            .map(|i| (height[i - 1] + height[i] + height[i + 1]) / 3.0)
            .collect();

        let mut terrain_poly: Vec<Vec2> = Vec::new();
        terrain_poly.push(Vec2::new(w / 2.0, -0.0));
        for (x, y) in chunk_x.into_iter().rev().zip(smooth_y.into_iter().rev()) {
            terrain_poly.push(Vec2::new(x - w / 2.0, y - 0.0));
        }
        terrain_poly.push(Vec2::new(-w / 2.0, -0.0));

        terrain_poly
    };

    let terrain_mesh = {
        let mut earcut = earcut::Earcut::new();
        let mut terrain_indices: Vec<u32> = Vec::new();

        let data: Vec<[f32; 2]> = terrain_poly.iter().map(|&v| [v.x, v.y]).collect();

        earcut.earcut(data.into_iter(), &[], &mut terrain_indices);

        let mut terrain_mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        let terrain_meshed_positions: Vec<[f32; 3]> =
            terrain_poly.iter().map(|p| [p.x, p.y, 0.0]).collect();
        terrain_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, terrain_meshed_positions);
        terrain_mesh.insert_indices(Indices::U32(terrain_indices));

        Mesh2dHandle(meshes.add(terrain_mesh))
    };

    let ground_id = commands
        .spawn(Collider::polyline(terrain_poly, None))
        .insert(MaterialMesh2dBundle {
            transform: Transform::from_xyz(0.0, -h / 2.0, 0.0),
            mesh: terrain_mesh,
            material: ground_material.clone(),
            ..Default::default()
        })
        .with_children(|parent| {
            for i in [-1.0, 1.0] {
                let x_distance = w / (chunks as f32 - 1.0);

                parent.spawn(MaterialMesh2dBundle {
                    transform: Transform::from_xyz(i * x_distance, helipad_y, 0.0),
                    mesh: flag_handle_pbr.0.clone(),
                    material: flag_handle_pbr.1.clone(),
                    ..Default::default()
                });
                parent.spawn(MaterialMesh2dBundle {
                    transform: Transform::from_xyz(i * x_distance, helipad_y, 0.0),
                    mesh: flag_pbr.0.clone(),
                    material: flag_pbr.1.clone(),
                    ..Default::default()
                });
            }
        })
        .id();

    let module_position = Vec2::new(0.0, 5.0);

    // Create the module center.
    let center_id = commands
        .spawn(RigidBody::Dynamic)
        .insert(
            Collider::convex_polyline(LANDER_POLY.iter().map(|a| *a / SCALE).collect()).unwrap(),
        )
        .insert(Restitution::coefficient(0.0))
        .insert(ColliderMassProperties::Density(5.0))
        .insert(MaterialMesh2dBundle {
            transform: Transform::from_xyz(module_position.x, module_position.y, 0.0),
            mesh: center_pbr.0.clone(),
            material: center_pbr.1.clone(),
            ..Default::default()
        })
        .insert(Velocity::zero())
        .insert(LanderCenter)
        .id();

    let leg_collider = Collider::convex_polyline(vec![
        Vec2::new(-LEG_W / SCALE, 0.0),                  // Left Upper
        Vec2::new(-LEG_W / SCALE, -LEG_H * 2.0 / SCALE), // Left Lower
        Vec2::new(LEG_W / SCALE, -LEG_H * 2.0 / SCALE),  // Right Lower
        Vec2::new(LEG_W / SCALE, 0.0),                   // Right Upper
    ])
    .unwrap();
    let leg_angle = 15f32.to_radians();

    // Create left and right legs.
    let mut leg_ids = Vec::new();
    for i in [-1.0, 1.0] {
        let leg_translation = Vec2::new(i * LEG_AWAY / SCALE, 0.0);

        let leg_id = commands
            .spawn(RigidBody::Dynamic)
            .insert(Collider::compound(vec![(
                Vec2::ZERO,
                0.0,
                leg_collider.clone(),
            )]))
            .insert(Restitution::coefficient(0.0))
            .insert(ColliderMassProperties::Density(1.0))
            .insert(MaterialMesh2dBundle {
                transform: Transform::from_xyz(
                    module_position.x + leg_translation.x,
                    module_position.y + leg_translation.y,
                    0.0,
                )
                .with_rotation(Quat::from_rotation_z(i * leg_angle)),
                mesh: leg_pbr.0.clone(),
                material: leg_pbr.1.clone(),
                ..Default::default()
            })
            .insert(ImpulseJoint::new(
                center_id,
                RevoluteJointBuilder::new()
                    .local_anchor2(Vec2::new(0.0, 0.0)) // Leg anchor
                    .local_anchor1(leg_translation) // Module anchor
                    .limits([-leg_angle, leg_angle]) // Rotation limits
                    .motor(0.0, i * 0.3, 0.0, LEG_SPRING_TORQUE),
            ))
            .id();

        leg_ids.push(leg_id);
    }

    commands.spawn(Game {
        center_id,
        leg_ids: leg_ids.try_into().unwrap(),
        ground_id,
        helipad_y,
        wind: None,
        frame: 0,
        game_over: None,
    });

    ev_init.send(GameInitEvent {
        initial_state: State([
            module_position.x / (VIEWPORT_W / SCALE / 2.0),
            (module_position.y - (helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2.0),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]),
    });

    commands.add(|world: &mut World| {
        world.run_schedule(PostGameInitSchedule);
    })
}

fn update_available_schedule(
    mut commands: Commands,
    time: Res<Time>,
    mut updater: ResMut<GameUpdater>,
) {
    updater.timer.tick(time.delta());
    while updater.timer.finished() {
        commands.add(|world: &mut World| {
            world.run_schedule(AvailableUpdateSchedule);
        });
        updater.timer.reset();
    }
}

fn game_pre_update(
    mut commands: Commands,
    mut ev_step_action: ResMut<Events<StepActionEvent>>,
    q_game: Query<&Game>,
    q_center: Query<&Transform, With<LanderCenter>>,
    mut ev_step_result: EventWriter<StepResultEvent>,
) {
    let action = ev_step_action.drain().next().unwrap();

    let game = q_game.single();
    let center_transform = q_center.get(game.center_id).unwrap();

    // Check if the simulation is already finished
    if let Some(result) = &game.game_over {
        ev_step_result.send(result.clone());
        commands.add(|world: &mut World| {
            world.run_schedule(PostGameStepSchedule);
        });
        return;
    }

    // TODO: find better value to this
    let arbitrary_force_m = 0.1;

    // Apply action in simulation

    commands.entity(game.center_id).insert(action.to_force(
        center_transform.rotation,
        MAIN_ENGINE_POWER * arbitrary_force_m,
        SIDE_ENGINE_POWER,
    ));

    commands.add(|world: &mut World| {
        world.run_schedule(PhysicsStepSchedule);
        world.run_schedule(PostPhysicsStepSchedule);
    });
}

fn game_post_physics_update(
    mut commands: Commands,
    rapier_context: Res<RapierContext>,
    q_game: Query<&Game>,
    q_center: Query<(&Transform, &Velocity), With<LanderCenter>>,
    mut ev_step_result: EventWriter<StepResultEvent>,
) {
    let game = q_game.single();
    let (center_transform, center_velocity) = q_center.get(game.center_id).unwrap();

    // Read state after actions applied
    let mut arm_states = [LegState::InAir; 2];

    for (i, arm) in game.leg_ids.iter().enumerate() {
        let arm = *arm;

        for contact_pair in rapier_context.contact_pairs_with(arm) {
            if contact_pair.has_any_active_contact() {
                let other_collider = if contact_pair.collider1() == arm {
                    contact_pair.collider2()
                } else {
                    contact_pair.collider1()
                };

                if other_collider == game.ground_id {
                    arm_states[i] = LegState::InGround;
                }
            }
        }
    }

    let result = StepResultEvent {
        next_state: State([
            center_transform.translation.x / (VIEWPORT_W / SCALE / 2.0),
            (center_transform.translation.y - (game.helipad_y + LEG_DOWN / SCALE))
                / (VIEWPORT_H / SCALE / 2.0),
            center_velocity.linvel.x * (VIEWPORT_W / SCALE / 2.0) / FPS,
            center_velocity.linvel.y * (VIEWPORT_H / SCALE / 2.0) / FPS,
            extract_2d_angle(center_transform.rotation),
            20.0 * center_velocity.angvel / FPS,
            if arm_states[0] == LegState::InGround {
                1.0
            } else {
                0.0
            },
            if arm_states[1] == LegState::InGround {
                1.0
            } else {
                0.0
            },
        ]),
        reward: 1.0, // TODO:
        done: false, // TODO:
    };

    ev_step_result.send(result);
    commands.add(|world: &mut World| {
        world.run_schedule(PostGameStepSchedule);
    });
}
