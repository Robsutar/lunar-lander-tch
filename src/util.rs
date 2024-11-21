use bevy::math::Quat;

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
