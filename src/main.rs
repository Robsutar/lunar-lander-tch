mod game;
use bevy_rapier2d::prelude::*;

fn main() {
    let mut app = App::default();
    app.add_plugins(DefaultPlugins);
    app.add_plugins(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0));
    app.add_plugins(RapierDebugRenderPlugin::default());
    app.add_systems(Startup, setup);
    app.run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());

    // Create the ground.
    commands
        .spawn(Collider::cuboid(500.0, 50.0))
        .insert(TransformBundle::from(Transform::from_xyz(0.0, -100.0, 0.0)));

    let module_position = Vec2::new(0.0, 250.0);

    // Create the module center.
    let module_center = commands
        .spawn(RigidBody::Dynamic)
        .insert(
            Collider::convex_polyline(vec![
                Vec2::new(-125.0, -100.0), // Lower left
                Vec2::new(125.0, -100.0),  // Lower right
                Vec2::new(125.0, -25.0),   // Center right
                Vec2::new(100.0, 100.0),   // Upper right
                Vec2::new(-100.0, 100.0),  // Upper left
                Vec2::new(-125.0, -25.0),  // Center left
            ])
            .unwrap(),
        )
        .insert(Restitution::coefficient(0.0))
        .insert(ColliderMassProperties::Density(5.0))
        .insert(TransformBundle::from(Transform::from_xyz(
            module_position.x,
            module_position.y,
            0.0,
        )))
        .id();

    let arm_collider = Collider::convex_polyline(vec![
        Vec2::new(-12.5, -100.0), // Lower left
        Vec2::new(12.5, -100.0),  // Lower right
        Vec2::new(12.5, 0.0),     // Upper right
        Vec2::new(-12.5, 0.0),    // Upper left
    ])
    .unwrap();
    let arm_angle = 15f32.to_radians();

    let leg_translation = Vec2::new(-150.0, -50.0);

    // Create left leg.
    commands
        .spawn(RigidBody::Dynamic)
        .insert(Collider::compound(vec![(
            Vec2::ZERO,
            -arm_angle,
            arm_collider.clone(),
        )]))
        .insert(Restitution::coefficient(0.0))
        .insert(ColliderMassProperties::Density(1.0))
        .insert(TransformBundle::from(Transform::from_xyz(
            module_position.x + leg_translation.x,
            module_position.y + leg_translation.y,
            0.0,
        )))
        .insert(ImpulseJoint::new(
            module_center,
            RevoluteJointBuilder::new()
                .local_anchor2(Vec2::new(0.0, 0.0)) // Leg anchor
                .local_anchor1(leg_translation) // Module anchor
                .limits([-arm_angle, arm_angle]) // Rotation limits
                .motor(0.0, -0.3, 0.0, 40.0),
        ));

    let leg_translation = Vec2::new(150.0, -50.0);

    // Create right leg.
    commands
        .spawn(RigidBody::Dynamic)
        .insert(Collider::compound(vec![(
            Vec2::ZERO,
            arm_angle,
            arm_collider.clone(),
        )]))
        .insert(Restitution::coefficient(0.0))
        .insert(ColliderMassProperties::Density(1.0))
        .insert(TransformBundle::from(Transform::from_xyz(
            module_position.x + leg_translation.x,
            module_position.y + leg_translation.y,
            0.0,
        )))
        .insert(ImpulseJoint::new(
            module_center,
            RevoluteJointBuilder::new()
                .local_anchor2(Vec2::new(0.0, 0.0)) // Leg anchor
                .local_anchor1(leg_translation) // Module anchor
                .limits([-arm_angle, arm_angle]) // Rotation limits
                .motor(0.0, 0.3, 0.0, 40.0),
        ));
}
