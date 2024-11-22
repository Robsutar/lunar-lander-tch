use std::time::{Duration, Instant};

use bevy::{ecs::schedule::ScheduleLabel, prelude::*, sprite::Mesh2dHandle};
use bevy_rapier2d::prelude::*;

#[derive(Component, Clone)]
pub struct Particle {
    pub lifetime: Duration,
    pub color: Handle<ColorMaterial>,
    pub mesh: Mesh2dHandle,
    pub friction: f32,
    pub collision_radius: f32,
    pub collision_groups: CollisionGroups,
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
    spawn_instant: Instant,
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
    if false {
        return;
    }

    let now = Instant::now();
    for spawning in ev_spawn_particle.drain() {
        commands
            .spawn(RigidBody::Dynamic)
            .insert(Collider::ball(spawning.particle.collision_radius))
            .insert(Restitution::coefficient(0.3))
            .insert(Damping {
                linear_damping: spawning.particle.friction,
                angular_damping: spawning.particle.friction,
            })
            .insert(Velocity {
                linvel: spawning.initial_velocity,
                angvel: 0.0,
            })
            .insert(ColorMesh2dBundle {
                mesh: spawning.particle.mesh,
                material: spawning.particle.color,
                transform: spawning.initial_transform,
                ..Default::default()
            })
            .insert(spawning.particle.collision_groups)
            .insert(SpawnedParticle {
                lifetime: spawning.particle.lifetime,
                spawn_instant: now,
            });
    }
}

fn update_spawned_particles(
    mut commands: Commands,
    mut q_spawned_particles: Query<(Entity, &SpawnedParticle)>,
) {
    let now = Instant::now();

    for (entity, spawned) in q_spawned_particles.iter_mut() {
        if now - spawned.spawn_instant >= spawned.lifetime {
            commands.entity(entity).despawn_recursive();
        }
    }
}
