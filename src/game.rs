use bevy::{
    ecs::schedule::ScheduleLabel,
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
        render_resource::ShaderType,
    },
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};
use bevy_rapier2d::prelude::*;
use rand::Rng;

pub const WINDOW_ZOOM: f32 = 2.0; // Affects only visually the scale of the window, adding zoom to camera.

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

pub type MeshMaterial2d = (Mesh2dHandle, Handle<ColorMaterial>);

#[derive(Resource)]
pub struct GameAssets {
    center_pbr: MeshMaterial2d,
    leg_pbr: MeshMaterial2d,
    flag_pbr: MeshMaterial2d,
    flag_handle_pbr: MeshMaterial2d,
    ground_material: Handle<ColorMaterial>,
}

#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
pub struct PreGameStepSchedule;

#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
pub struct GameStepSchedule;

#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
pub struct PostGameStepSchedule;

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
                substeps: SCALE as usize,
            },
            ..RapierConfiguration::new(1.0)
        });
        app.add_plugins(
            RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(1.0 / SCALE)
                .in_schedule(GameStepSchedule),
        );
        app.add_plugins(RapierDebugRenderPlugin::default());

        app.add_systems(Startup, init_assets);
        app.add_systems(PostStartup, init_game);

        app.init_schedule(PreGameStepSchedule);
        app.init_schedule(GameStepSchedule);
        app.init_schedule(PostGameStepSchedule);
        app.insert_resource(GameUpdater {
            timer: Timer::from_seconds(1.0 / FPS, TimerMode::Repeating),
        });
        app.add_systems(Update, game_updater);
    }
}

fn init_assets(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
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

    let moon_material = materials.add(Color::WHITE);

    commands.insert_resource(GameAssets {
        center_pbr,
        leg_pbr,
        flag_pbr,
        flag_handle_pbr,
        ground_material: moon_material,
    });
}

fn init_game(mut commands: Commands, assets: Res<GameAssets>, mut meshes: ResMut<Assets<Mesh>>) {
    let mut rng = rand::thread_rng();

    // Create camera
    commands.spawn(Camera2dBundle {
        transform: Transform::from_scale(Vec3::new(
            1.0 / SCALE / WINDOW_ZOOM,
            1.0 / SCALE / WINDOW_ZOOM,
            1.0 / SCALE / WINDOW_ZOOM,
        )),
        ..Default::default()
    });

    let w = VIEWPORT_W / SCALE;
    let h = VIEWPORT_H / SCALE;

    // Create the terrain.
    let chunks = 11;
    let chunk_x: Vec<f32> = (0..chunks)
        .map(|i| w / (chunks as f32 - 1.0) * i as f32)
        .collect();

    let mut height: Vec<f32> = (0..chunks).map(|_| rng.gen_range(0.0..h / 2.0)).collect();

    let helipad_x1 = height[chunks / 2 - 1];
    let helipad_x2 = height[chunks / 2 + 1];
    let helipad_y = h / 4.0;

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

    commands
        .spawn(Collider::polyline(terrain_poly, None))
        .insert(MaterialMesh2dBundle {
            transform: Transform::from_xyz(0.0, -h / 2.0, 0.0),
            mesh: terrain_mesh,
            material: assets.ground_material.clone(),
            ..Default::default()
        })
        .with_children(|parent| {
            for i in [-1.0, 1.0] {
                let x_distance = w / (chunks as f32 - 1.0);

                parent.spawn(MaterialMesh2dBundle {
                    transform: Transform::from_xyz(i * x_distance, helipad_y, 0.0),
                    mesh: assets.flag_handle_pbr.0.clone(),
                    material: assets.flag_handle_pbr.1.clone(),
                    ..Default::default()
                });
                parent.spawn(MaterialMesh2dBundle {
                    transform: Transform::from_xyz(i * x_distance, helipad_y, 0.0),
                    mesh: assets.flag_pbr.0.clone(),
                    material: assets.flag_pbr.1.clone(),
                    ..Default::default()
                });
            }
        });

    let module_position = Vec2::new(0.0, 5.0);

    // Create the module center.
    let module_center = commands
        .spawn(RigidBody::Dynamic)
        .insert(
            Collider::convex_polyline(LANDER_POLY.iter().map(|a| *a / SCALE).collect()).unwrap(),
        )
        .insert(Restitution::coefficient(0.0))
        .insert(ColliderMassProperties::Density(5.0))
        .insert(MaterialMesh2dBundle {
            transform: Transform::from_xyz(module_position.x, module_position.y, 0.0),
            mesh: assets.center_pbr.0.clone(),
            material: assets.center_pbr.1.clone(),
            ..Default::default()
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
    for i in [-1.0, 1.0] {
        let leg_translation = Vec2::new(i * LEG_AWAY / SCALE, 0.0);

        commands
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
                ),
                mesh: assets.leg_pbr.0.clone(),
                material: assets.leg_pbr.1.clone(),
                ..Default::default()
            })
            .insert(ImpulseJoint::new(
                module_center,
                RevoluteJointBuilder::new()
                    .local_anchor2(Vec2::new(0.0, 0.0)) // Leg anchor
                    .local_anchor1(leg_translation) // Module anchor
                    .limits([-leg_angle, leg_angle]) // Rotation limits
                    .motor(0.0, i * 0.3, 0.0, LEG_SPRING_TORQUE),
            ));
    }
}

fn game_updater(mut commands: Commands, time: Res<Time>, mut updater: ResMut<GameUpdater>) {
    updater.timer.tick(time.delta());

    while updater.timer.finished() {
        commands.add(|world: &mut World| {
            world.run_schedule(PreGameStepSchedule);
            world.run_schedule(GameStepSchedule);
            world.run_schedule(PostGameStepSchedule);
        });
        updater.timer.reset();
    }
}
