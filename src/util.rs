use std::collections::VecDeque;

use bevy::math::Quat;

pub struct FixedVecDeque<T> {
    deque: VecDeque<T>,
    max_len: usize,
}

impl<T> FixedVecDeque<T> {
    pub fn new(max_len: usize) -> Self {
        Self {
            deque: VecDeque::new(),
            max_len,
        }
    }

    pub fn push(&mut self, value: T) {
        if self.deque.len() == self.max_len {
            self.deque.pop_front();
        }
        self.deque.push_back(value);
    }

    pub fn as_deque(&self) -> &VecDeque<T> {
        &self.deque
    }

    pub fn len(&self) -> usize {
        self.deque.len()
    }
}

pub fn extract_2d_angle(quat: Quat) -> f32 {
    let extracted_angle = 2.0 * quat.w.acos();

    let axis = quat.xyz().normalize();
    let corrected_angle = if axis.z < 0.0 {
        -extracted_angle
    } else {
        extracted_angle
    };

    corrected_angle
}
