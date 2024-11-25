use std::time::{Duration, Instant};

use bevy::{ecs::schedule::ScheduleLabel, prelude::*, sprite::Mesh2dHandle};
use bevy_rapier2d::prelude::*;

#[derive(Component, Clone)]
pub struct Particle {
    pub lifetime: Duration,
    pub color: Color,
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
    mut materials: ResMut<Assets<ColorMaterial>>,
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
                material: materials.add(spawning.particle.color),
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
    mut q_spawned_particles: Query<(Entity, &SpawnedParticle, &Handle<ColorMaterial>)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let now = Instant::now();

    for (entity, spawned, color_handle) in q_spawned_particles.iter_mut() {
        let elapsed = now - spawned.spawn_instant;
        if elapsed >= spawned.lifetime {
            commands.entity(entity).despawn_recursive();
        } else {
            let progress = elapsed.as_secs_f32() / spawned.lifetime.as_secs_f32();

            let color_material = materials.get_mut(color_handle).unwrap();
            color_material.color.set_alpha(1.0 - progress);
        }
    }
}
