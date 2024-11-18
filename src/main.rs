use bevy::prelude::*;
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
        .insert(Restitution::coefficient(0.7))
        .insert(TransformBundle::from(Transform::from_xyz(0.0, 400.0, 0.0)))
        .id();

    // Create left leg.
    commands
        .spawn(RigidBody::Dynamic)
        .insert(Collider::cuboid(12.5, 50.0))
        .insert(Restitution::coefficient(0.7))
        .insert(TransformBundle::from(Transform::from_xyz(
            -150.0, 300.0, 0.0,
        )))
        .insert(ImpulseJoint::new(
            module_center,
            RevoluteJointBuilder::new()
                .local_anchor2(Vec2::new(0.0, 50.0)) // Leg anchor
                .local_anchor1(Vec2::new(-150.0, -50.0)), // Module anchor
        ));

    // Create right leg.
    commands
        .spawn(RigidBody::Dynamic)
        .insert(Collider::cuboid(12.5, 50.0))
        .insert(Restitution::coefficient(0.7))
        .insert(TransformBundle::from(Transform::from_xyz(
            150.0, 300.0, 0.0,
        )))
        .insert(ImpulseJoint::new(
            module_center,
            RevoluteJointBuilder::new()
                .local_anchor2(Vec2::new(0.0, 50.0)) // Leg anchor
                .local_anchor1(Vec2::new(150.0, -50.0)), // Module anchor
        ));
}
