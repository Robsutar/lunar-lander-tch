use std::time::Duration;

use bevy::{
    ecs::schedule::ScheduleLabel,
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
    },
    sprite::{ColorMesh2dBundle, Mesh2dHandle},
};
use bevy_rapier2d::prelude::*;
use rand::Rng;

use crate::{particle::*, util::*};

pub const FPS: f32 = 50.0;
pub const SCALE: f32 = 30.0; // Affects how fast-paced the game is, forces should be adjusted as well

pub const MAIN_ENGINE_POWER: f32 = 13.0 * 0.1; // 0.1 is an arbitrary value to make rapier2d accord to box2d
pub const SIDE_ENGINE_POWER: f32 = 0.6;

pub const INITIAL_RANDOM: f32 = 1000.0 * 0.001; // Set 1500 to make game harder, 0.001 is an arbitrary value to make rapier2d accord to box2d

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
pub const LEG_SPRING_TORQUE: f32 = 40.0 * 2000.0; // 2000.0 is an arbitrary value to make rapier2d accord to box2d

pub const SIDE_ENGINE_HEIGHT: f32 = 14.0;
pub const SIDE_ENGINE_AWAY: f32 = 12.0; // Horizontal distance off center

pub const VIEWPORT_W: f32 = 600.0; // Width of the window
pub const VIEWPORT_H: f32 = 400.0; // Height of the window

pub const GROUND_COLLISION_GROUP: Group = Group::GROUP_10;

#[derive(Debug, Clone)]
pub struct State(pub [f32; Self::SIZE]);
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
pub struct GameResetEvent {
    pub initial_state: State,
}

#[derive(Debug, Event, PartialEq, Eq, Clone)]
pub enum Action {
    Nothing,
    ThrusterLeft,
    ThrusterRight,
    ThrusterMain,
}
impl Action {
    pub const SIZE: usize = 4;

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
    ) -> Option<ExternalImpulse> {
        match self {
            Action::Nothing => None,
            Action::ThrusterLeft => Some(ExternalImpulse {
                impulse: (center_rotation * Vec3::new(side_engine_power, 0.0, 0.0)).truncate(),
                torque_impulse: 0.0,
            }),
            Action::ThrusterRight => Some(ExternalImpulse {
                impulse: (center_rotation * Vec3::new(-side_engine_power, 0.0, 0.0)).truncate(),
                torque_impulse: 0.0,
            }),
            Action::ThrusterMain => Some(ExternalImpulse {
                impulse: (center_rotation * Vec3::new(0.0, main_engine_power, 0.0)).truncate(),
                torque_impulse: 0.0,
            }),
        }
    }

    fn to_spawn_particle(
        &self,
        thruster_particle: Particle,
        mut center_transform: Transform,
    ) -> Option<SpawnParticleEvent> {
        if *self == Action::Nothing {
            return None;
        }

        let (translation, speed) = match self {
            Action::Nothing => panic!(),
            Action::ThrusterLeft => (
                Vec2::new(-SIDE_ENGINE_AWAY / SCALE, 0.0),
                Vec2::new(SIDE_ENGINE_POWER * 300.0 / SCALE, 0.0),
            ),
            Action::ThrusterRight => (
                Vec2::new(SIDE_ENGINE_AWAY / SCALE, 0.0),
                Vec2::new(-SIDE_ENGINE_POWER * 300.0 / SCALE, 0.0),
            ),
            Action::ThrusterMain => (
                Vec2::new(0.0, -SIDE_ENGINE_HEIGHT / SCALE),
                Vec2::new(0.0, -MAIN_ENGINE_POWER * 300.0 / SCALE),
            ),
        };

        center_transform.translation +=
            center_transform.rotation * Vec3::new(translation.x, translation.y, 0.0);

        Some(SpawnParticleEvent {
            particle: thruster_particle,
            initial_transform: center_transform,
            initial_velocity: (center_transform.rotation * Vec3::new(-speed.x, speed.y, 0.0))
                .truncate(),
        })
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

#[derive(Component)]
pub struct LanderCenter;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LegState {
    InAir,
    InGround,
}

enum PostUpdateCall {
    GameReset,
    GameStep(Action),
}

#[derive(Component)]
pub struct Game {
    thruster_particle: Particle,

    center_id: Entity,
    leg_ids: [Entity; 2],
    ground_id: Entity,

    helipad_y: f32,

    frame: usize,
    next_post_update_call: PostUpdateCall,
    prev_shaping: Option<f32>,
    is_finished: Option<StepResultEvent>,
}
impl Game {
    pub fn reset(commands: &mut Commands) {
        commands.add(|world: &mut World| {
            world.run_schedule(GameResetSchedule);
        })
    }

    pub fn play_step(
        commands: &mut Commands,
        ev_step_action: &mut EventWriter<Action>,
        action: Action,
    ) {
        ev_step_action.send(action);
        commands.add(|world: &mut World| {
            world.run_schedule(PreGameStepSchedule);
        })
    }

    pub fn frame(&self) -> usize {
        self.frame
    }
}

#[derive(Resource)]
pub struct GameUpdater {
    timer: Timer,
}

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
struct ResetPostPhysicsStepSchedule;
#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
struct StepPostPhysicsStepSchedule;
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
        app.init_schedule(GameResetSchedule);
        app.init_schedule(PostGameResetSchedule);

        app.init_schedule(AvailableUpdateSchedule);

        app.init_schedule(PreGameStepSchedule);
        app.init_schedule(PhysicsStepSchedule);
        app.init_schedule(ResetPostPhysicsStepSchedule);
        app.init_schedule(StepPostPhysicsStepSchedule);
        app.init_schedule(PostGameStepSchedule);

        app.insert_resource(GameUpdater {
            timer: Timer::from_seconds(1.0 / FPS, TimerMode::Repeating),
        });
        app.insert_resource(RapierConfiguration {
            gravity: Vec2::new(0.0, self.gravity),
            timestep_mode: TimestepMode::Fixed {
                dt: 1.0 / FPS,
                substeps: SCALE as usize,
            },
            ..RapierConfiguration::new(1.0)
        });
        app.add_plugins(
            RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(1.0 / SCALE)
                .in_schedule(PhysicsStepSchedule),
        );

        app.add_plugins(ParticlePlugin {
            schedule: PostGameStepSchedule,
        });

        app.add_event::<GameResetEvent>();
        app.add_event::<Action>();
        app.add_event::<StepResultEvent>();

        app.add_systems(PostStartup, game_init);
        app.add_systems(GameResetSchedule, game_reset);

        app.add_systems(Update, update_available_schedule);
        app.add_systems(PreGameStepSchedule, game_pre_update);
        app.add_systems(ResetPostPhysicsStepSchedule, game_post_physics_update);
        app.add_systems(StepPostPhysicsStepSchedule, game_post_physics_update);
    }
}

fn spawn_terrain_poly_mesh(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    chunks: usize,
    w: f32,
    h: f32,
    helipad_y: f32,
) -> Entity {
    let mut rng = rand::thread_rng();

    let ground_material = materials.add(Color::WHITE);

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

        terrain_mesh
    };

    commands
        .spawn(Collider::polyline(terrain_poly, None))
        .insert(ColorMesh2dBundle {
            transform: Transform::from_xyz(0.0, -h / 2.0, 0.0),
            mesh: Mesh2dHandle(meshes.add(terrain_mesh)),
            material: ground_material.clone(),
            ..Default::default()
        })
        .insert(CollisionGroups::new(GROUND_COLLISION_GROUP, Group::ALL))
        .insert(Friction {
            coefficient: 0.1,
            combine_rule: CoefficientCombineRule::Min,
        })
        .id()
}

fn game_init(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let mut rng = rand::thread_rng();

    // Create assets
    let thruster_particle = Particle {
        lifetime: Duration::from_millis(1500),
        color: materials.add(Color::srgb_u8(232, 204, 42)),
        mesh: Mesh2dHandle(meshes.add(Cuboid::new(3.0 / SCALE, 3.0 / SCALE, 0.0))),
        friction: 0.1,
        collision_radius: 1.5 / SCALE,
        collision_groups: CollisionGroups::new(Group::GROUP_13, GROUND_COLLISION_GROUP),
    };

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

    let w: f32 = VIEWPORT_W / SCALE;
    let h = VIEWPORT_H / SCALE;
    let chunks = 11;
    let helipad_y = h / 4.0;

    let ground_id = spawn_terrain_poly_mesh(
        &mut commands,
        &mut meshes,
        &mut materials,
        chunks,
        w,
        h,
        helipad_y,
    );

    for i in [-1.0, 1.0] {
        let x_distance = w / (chunks as f32 - 1.0);

        commands.spawn(ColorMesh2dBundle {
            transform: Transform::from_xyz(i * x_distance, helipad_y - h / 2.0, 0.0),
            mesh: flag_handle_pbr.0.clone(),
            material: flag_handle_pbr.1.clone(),
            ..Default::default()
        });
        commands.spawn(ColorMesh2dBundle {
            transform: Transform::from_xyz(i * x_distance, helipad_y - h / 2.0, 0.0),
            mesh: flag_pbr.0.clone(),
            material: flag_pbr.1.clone(),
            ..Default::default()
        });
    }

    let center_position = Vec2::new(0.0, VIEWPORT_H / SCALE / 2.0);

    // Create the module center.
    let center_id = commands
        .spawn(RigidBody::Dynamic)
        .insert(
            Collider::convex_polyline(LANDER_POLY.iter().map(|a| *a / SCALE).collect()).unwrap(),
        )
        .insert(Restitution::coefficient(0.0))
        .insert(ColliderMassProperties::Density(5.0))
        .insert(ColorMesh2dBundle {
            transform: Transform::from_xyz(center_position.x, center_position.y, 0.0),
            mesh: center_pbr.0.clone(),
            material: center_pbr.1.clone(),
            ..Default::default()
        })
        .insert(Velocity::zero())
        .insert(LanderCenter)
        .insert(CollisionGroups::new(
            Group::GROUP_11,
            GROUND_COLLISION_GROUP,
        ))
        .insert(ExternalImpulse {
            impulse: Vec2::new(
                rng.gen_range(-INITIAL_RANDOM..INITIAL_RANDOM),
                rng.gen_range(-INITIAL_RANDOM..INITIAL_RANDOM),
            ),
            torque_impulse: 0.0,
        })
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

        let joint_limits = if i == -1.0 {
            [-leg_angle * 2.0, 0.0]
        } else {
            [0.0, leg_angle * 2.0]
        };

        let leg_id = commands
            .spawn(RigidBody::Dynamic)
            .insert(Collider::compound(vec![(
                Vec2::ZERO,
                0.0,
                leg_collider.clone(),
            )]))
            .insert(Restitution::coefficient(0.0))
            .insert(ColliderMassProperties::Density(1.0))
            .insert(ColorMesh2dBundle {
                transform: Transform::from_xyz(
                    center_position.x + leg_translation.x,
                    center_position.y + leg_translation.y,
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
                    .limits(joint_limits) // Rotation limits
                    .motor_position(i * leg_angle, LEG_SPRING_TORQUE, 150.0),
            ))
            .insert(CollisionGroups::new(
                Group::GROUP_12,
                GROUND_COLLISION_GROUP,
            ))
            .id();

        leg_ids.push(leg_id);
    }

    commands.spawn(Game {
        thruster_particle,

        center_id,
        leg_ids: leg_ids.try_into().unwrap(),
        ground_id,

        helipad_y,

        frame: 0,
        next_post_update_call: PostUpdateCall::GameReset,
        prev_shaping: None,
        is_finished: None,
    });

    commands.add(|world: &mut World| {
        world.run_schedule(PhysicsStepSchedule);
        world.run_schedule(ResetPostPhysicsStepSchedule);
    });
}

fn game_reset(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut q_game: Query<&mut Game>,
) {
    let mut rng = rand::thread_rng();

    let mut game = q_game.single_mut();

    commands.entity(game.ground_id).despawn_recursive();

    let w: f32 = VIEWPORT_W / SCALE;
    let h = VIEWPORT_H / SCALE;
    let chunks = 11;
    let helipad_y = h / 4.0;

    let ground_id = spawn_terrain_poly_mesh(
        &mut commands,
        &mut meshes,
        &mut materials,
        chunks,
        w,
        h,
        helipad_y,
    );
    game.ground_id = ground_id;

    let center_position = Vec2::new(0.0, VIEWPORT_H / SCALE / 2.0);

    commands
        .entity(game.center_id)
        .insert(Transform::from_translation(Vec3::new(
            center_position.x,
            center_position.y,
            0.0,
        )))
        .insert(Velocity::zero())
        .insert(ExternalImpulse {
            impulse: Vec2::new(
                rng.gen_range(-INITIAL_RANDOM..INITIAL_RANDOM),
                rng.gen_range(-INITIAL_RANDOM..INITIAL_RANDOM),
            ),
            torque_impulse: 0.0,
        });

    for (index, i) in [-1.0, 1.0].into_iter().enumerate() {
        let leg = game.leg_ids[index];
        let leg_translation = Vec2::new(i * LEG_AWAY / SCALE, 0.0);

        commands
            .entity(leg)
            .insert(Transform::from_xyz(
                center_position.x + leg_translation.x,
                center_position.y + leg_translation.y,
                0.0,
            ))
            .insert(Velocity::zero());
    }

    game.frame = 0;
    game.next_post_update_call = PostUpdateCall::GameReset;
    game.prev_shaping = None;
    game.is_finished = None;

    commands.add(|world: &mut World| {
        world.run_schedule(PhysicsStepSchedule);
        world.run_schedule(ResetPostPhysicsStepSchedule);
    });
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
    mut ev_spawn_particle: EventWriter<SpawnParticleEvent>,
    mut ev_step_action: ResMut<Events<Action>>,
    mut q_game: Query<&mut Game>,
    q_center: Query<&Transform, With<LanderCenter>>,
    mut ev_step_result: EventWriter<StepResultEvent>,
) {
    let action = ev_step_action.drain().next().unwrap();

    let mut game = q_game.single_mut();
    let center_transform = q_center.get(game.center_id).unwrap();

    // Check if the simulation is already finished
    if let Some(result) = &game.is_finished {
        ev_step_result.send(result.clone());
        commands.add(|world: &mut World| {
            world.run_schedule(PostGameStepSchedule);
        });
        return;
    }

    // Apply action in simulation
    if let Some(force) = action.to_force(
        center_transform.rotation,
        MAIN_ENGINE_POWER,
        SIDE_ENGINE_POWER,
    ) {
        commands.entity(game.center_id).insert(force);
    }
    // % 4 to avoid lag in debug mode
    if game.frame % 4 == 0 {
        if let Some(spawn_particle) =
            action.to_spawn_particle(game.thruster_particle.clone(), *center_transform)
        {
            ev_spawn_particle.send(spawn_particle);
        }
    }

    game.next_post_update_call = PostUpdateCall::GameStep(action);

    commands.add(|world: &mut World| {
        world.run_schedule(PhysicsStepSchedule);
        world.run_schedule(StepPostPhysicsStepSchedule);
    });
}

fn game_post_physics_update(
    mut commands: Commands,
    rapier_context: Res<RapierContext>,
    mut q_game: Query<&mut Game>,
    q_center: Query<(&Transform, &Velocity, &RapierRigidBodyHandle), With<LanderCenter>>,
    mut ev_reset: EventWriter<GameResetEvent>,
    mut ev_step: EventWriter<StepResultEvent>,
) {
    let mut game = q_game.single_mut();
    let (center_transform, center_velocity, center_body_handle) =
        q_center.get(game.center_id).unwrap();

    // Read state after actions applied
    let mut leg_states = [LegState::InAir; 2];

    for (i, leg) in game.leg_ids.iter().enumerate() {
        let leg = *leg;

        for contact_pair in rapier_context.contact_pairs_with(leg) {
            if contact_pair.has_any_active_contact() {
                let other_collider = if contact_pair.collider1() == leg {
                    contact_pair.collider2()
                } else {
                    contact_pair.collider1()
                };

                if other_collider == game.ground_id {
                    leg_states[i] = LegState::InGround;
                }
            }
        }
    }

    let next_state = State([
        center_transform.translation.x / (VIEWPORT_W / SCALE / 2.0),
        (center_transform.translation.y - (game.helipad_y + LEG_DOWN / SCALE))
            / (VIEWPORT_H / SCALE / 2.0),
        center_velocity.linvel.x * (VIEWPORT_W / SCALE / 2.0) / FPS,
        center_velocity.linvel.y * (VIEWPORT_H / SCALE / 2.0) / FPS,
        extract_2d_angle(center_transform.rotation),
        20.0 * center_velocity.angvel / FPS,
        if leg_states[0] == LegState::InGround {
            1.0
        } else {
            0.0
        },
        if leg_states[1] == LegState::InGround {
            1.0
        } else {
            0.0
        },
    ]);

    match &game.next_post_update_call {
        PostUpdateCall::GameReset => {
            ev_reset.send(GameResetEvent {
                initial_state: next_state,
            });
            commands.add(|world: &mut World| {
                world.run_schedule(PostGameResetSchedule);
            })
        }
        PostUpdateCall::GameStep(action) => {
            let (reward, done) = {
                if let Some(_) =
                    rapier_context
                        .contact_pairs_with(game.center_id)
                        .find(|contact_pair| {
                            if contact_pair.has_any_active_contact() {
                                let other_collider = if contact_pair.collider1() == game.center_id {
                                    contact_pair.collider2()
                                } else {
                                    contact_pair.collider1()
                                };

                                if other_collider == game.ground_id {
                                    return true;
                                }
                            }

                            false
                        })
                {
                    (-100.0, true)
                } else if rapier_context
                    .bodies
                    .get(center_body_handle.0)
                    .unwrap()
                    .is_sleeping()
                {
                    (100.0, true)
                } else if next_state.position_x() > 1.0
                    || next_state.position_x() < -1.0
                    || next_state.position_y() > 1.0
                {
                    (-100.0, true)
                } else {
                    let shaping = {
                        let position_distance = -100.0
                            * (next_state.position_x().powi(2) + next_state.position_y().powi(2))
                                .sqrt();
                        let velocity_penalty = -100.0
                            * (next_state.velocity_x().powi(2) + next_state.velocity_y().powi(2))
                                .sqrt();
                        let angle_penalty = -100.0 * next_state.orientation_angle().abs();
                        let leg_contact_reward = 10.0 * next_state.left_leg_contact()
                            + 10.0 * next_state.right_leg_contact();

                        position_distance + velocity_penalty + angle_penalty + leg_contact_reward
                    };

                    let reward = {
                        let (m_power, s_power) = match action {
                            Action::Nothing => (0.0, 0.0),
                            Action::ThrusterLeft => (0.0, 1.0),
                            Action::ThrusterRight => (0.0, 1.0),
                            Action::ThrusterMain => (1.0, 0.0),
                        };

                        let fuel_penalty = m_power * 0.30 + s_power * 0.03;
                        let delta_shaping = if let Some(prev_shaping) = game.prev_shaping {
                            shaping - prev_shaping
                        } else {
                            0.0
                        };

                        game.prev_shaping = Some(shaping);
                        delta_shaping - fuel_penalty
                    };

                    (reward, false)
                }
            };

            let result = StepResultEvent {
                next_state,
                reward,
                done,
            };

            if done {
                game.is_finished = Some(result.clone())
            } else {
                game.frame += 1;
            }

            ev_step.send(result);
            commands.add(|world: &mut World| {
                world.run_schedule(PostGameStepSchedule);
            });
        }
    }
}
