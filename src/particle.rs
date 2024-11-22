use std::time::{Duration, Instant};

use bevy::{ecs::schedule::ScheduleLabel, prelude::*, sprite::Mesh2dHandle};

#[derive(Component, Clone)]
pub struct Particle {
    pub lifetime: Duration,
    pub color: Handle<ColorMaterial>,
    pub mesh: Mesh2dHandle,
    pub friction: Vec2,
}

#[derive(Event, Clone)]
pub struct SpawnParticleEvent {
    pub particle: Particle,
    pub initial_transform: Transform,
    pub initial_velocity: Vec2,
}

#[derive(Component)]
struct SpawnedParticle {
    lifetime: Duration,
    friction: Vec2,
    spawn_instant: Instant,
    velocity: Vec2,
}

pub struct ParticlePlugin<T: ScheduleLabel + Clone> {
    pub schedule: T,
}

impl<T: ScheduleLabel + Clone> Plugin for ParticlePlugin<T> {
    fn build(&self, app: &mut App) {
        app.add_event::<SpawnParticleEvent>();

        app.add_systems(
            self.schedule.clone(),
            (spawn_pending_particles, update_spawned_particles),
        );
    }
}

fn spawn_pending_particles(
    mut commands: Commands,
    mut ev_spawn_particle: ResMut<Events<SpawnParticleEvent>>,
) {
    let now = Instant::now();
    for spawning in ev_spawn_particle.drain() {
        commands
            .spawn(ColorMesh2dBundle {
                mesh: spawning.particle.mesh,
                material: spawning.particle.color,
                transform: spawning.initial_transform,
                ..Default::default()
            })
            .insert(SpawnedParticle {
                lifetime: spawning.particle.lifetime,
                friction: spawning.particle.friction,
                spawn_instant: now,
                velocity: spawning.initial_velocity,
            });
    }
}

fn update_spawned_particles(
    time: Res<Time>,
    mut commands: Commands,
    mut q_spawned_particles: Query<(Entity, &mut Transform, &mut SpawnedParticle)>,
) {
    let dt = time.delta_seconds();
    let now = Instant::now();

    for (entity, mut transform, mut spawned) in q_spawned_particles.iter_mut() {
        if now - spawned.spawn_instant >= spawned.lifetime {
            commands.entity(entity).despawn();
            continue;
        }

        let velocity_reducer = 1.0 - spawned.friction * dt;
        spawned.velocity *= velocity_reducer;

        let transform_add = transform.rotation * spawned.velocity.extend(0.0) * dt;
        transform.translation += transform_add;
    }
}
