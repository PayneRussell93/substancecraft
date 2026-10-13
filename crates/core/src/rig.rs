//! Skeletons, skin weights, and animation clips of imported meshes, and the
//! linear-blend skinning that poses them.
//!
//! A [`Rig`] belongs to one imported mesh and is never saved in a project:
//! it is read again with the mesh file. Painting and baking always use the
//! mesh as imported (its bind pose); a pose only changes what the viewport
//! shows and where the pointer lands on it.
//!
//! Each joint's local transform is `T · pre · R · mid · S · post`, where
//! `T`, `R`, and `S` are its (animatable) translation, rotation, and scale.
//! glTF joints leave `pre`, `mid`, and `post` at identity; FBX joints use
//! them for their offsets, pivots, and pre- and post-rotations. A skin
//! binding attaches vertices to a joint through
//! `before · global(joint) · inverse_bind`, which is the identity in the
//! pose the mesh was bound in.

use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Influences kept per vertex; importers keep the strongest.
pub const MAX_INFLUENCES: usize = 4;
/// Joints a rig may have.
pub const MAX_JOINTS: usize = 4096;
/// Keys a track may have.
pub const MAX_TRACK_KEYS: usize = 1_000_000;
/// Clips a rig may have.
pub const MAX_CLIPS: usize = 256;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Joint {
    pub name: String,
    pub parent: Option<u32>,
    /// Rest translation, rotation, and scale, used where no clip animates
    /// them.
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub pre: Mat4,
    pub mid: Mat4,
    pub post: Mat4,
}

impl Joint {
    /// A joint with a plain translation-rotation-scale local transform.
    pub fn new(
        name: impl Into<String>,
        parent: Option<u32>,
        translation: Vec3,
        rotation: Quat,
        scale: Vec3,
    ) -> Self {
        Self {
            name: name.into(),
            parent,
            translation,
            rotation,
            scale,
            pre: Mat4::IDENTITY,
            mid: Mat4::IDENTITY,
            post: Mat4::IDENTITY,
        }
    }

    fn local(&self, translation: Vec3, rotation: Quat, scale: Vec3) -> Mat4 {
        Mat4::from_translation(translation)
            * self.pre
            * Mat4::from_quat(rotation.normalize())
            * self.mid
            * Mat4::from_scale(scale)
            * self.post
    }

    fn is_finite(&self) -> bool {
        self.translation.is_finite()
            && self.rotation.is_finite()
            && self.rotation.length_squared() > f32::EPSILON
            && self.scale.is_finite()
            && self.pre.is_finite()
            && self.mid.is_finite()
            && self.post.is_finite()
    }
}

/// Attaches vertices to a joint (see the module notes).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkinBinding {
    pub joint: u32,
    pub before: Mat4,
    pub inverse_bind: Mat4,
}

/// One binding's share of a vertex.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Influence {
    pub binding: u32,
    pub weight: f32,
}

/// Keyframes of one property, interpolated linearly (or held, for steps)
/// and clamped outside their range.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Track<T> {
    /// Seconds, non-decreasing.
    pub times: Vec<f32>,
    pub values: Vec<T>,
    #[serde(default)]
    pub step: bool,
}

impl<T: Copy> Track<T> {
    fn is_valid(&self, finite: impl Fn(&T) -> bool) -> bool {
        !self.times.is_empty()
            && self.times.len() == self.values.len()
            && self.times.len() <= MAX_TRACK_KEYS
            && self.times.iter().all(|time| time.is_finite())
            && self.times.windows(2).all(|pair| pair[0] <= pair[1])
            && self.values.iter().all(finite)
    }

    fn sample(&self, time: f32, blend: impl Fn(T, T, f32) -> T) -> Option<T> {
        let last = self.times.len().checked_sub(1)?;
        let after = self.times.partition_point(|key| *key <= time);
        if after == 0 {
            return self.values.first().copied();
        }
        if after > last {
            return self.values.get(last).copied();
        }
        let (start, end) = (self.times[after - 1], self.times[after]);
        let (from, to) = (self.values[after - 1], self.values[after]);
        if self.step || end <= start {
            return Some(from);
        }
        Some(blend(from, to, (time - start) / (end - start)))
    }
}

/// The animated properties of one joint in a clip.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JointAnimation {
    pub joint: u32,
    pub translation: Option<Track<Vec3>>,
    pub rotation: Option<Track<Quat>>,
    pub scale: Option<Track<Vec3>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationClip {
    pub name: String,
    /// Seconds; times run from 0.
    pub duration: f32,
    pub joints: Vec<JointAnimation>,
}

impl AnimationClip {
    /// The latest key time of any track, at least 0.
    pub fn key_span(&self) -> f32 {
        self.joints
            .iter()
            .flat_map(|joint| {
                [
                    joint.translation.as_ref().map(|track| &track.times),
                    joint.rotation.as_ref().map(|track| &track.times),
                    joint.scale.as_ref().map(|track| &track.times),
                ]
            })
            .flatten()
            .filter_map(|times| times.last().copied())
            .fold(0.0, f32::max)
    }
}

/// A clip and a time in it.
#[derive(Clone, Copy, Debug)]
pub struct ClipSample<'a> {
    pub clip: &'a AnimationClip,
    pub time: f32,
    /// Holds root joints (those without a parent) at their rest
    /// translation, so root motion does not carry the mesh away.
    pub in_place: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Rig {
    pub joints: Vec<Joint>,
    pub bindings: Vec<SkinBinding>,
    /// Per vertex of the mesh as imported; a vertex with no weight stays
    /// where it is.
    pub influences: Vec<[Influence; MAX_INFLUENCES]>,
    pub clips: Vec<AnimationClip>,
}

#[derive(Debug, Error, PartialEq)]
pub enum RigError {
    #[error("the rig has {0} joints; at most {MAX_JOINTS} are supported")]
    TooManyJoints(usize),
    #[error("joint {0} has a missing or cyclic parent")]
    InvalidParent(usize),
    #[error("joint {0} has a non-finite or degenerate transform")]
    InvalidJoint(usize),
    #[error("skin binding {0} is invalid")]
    InvalidBinding(usize),
    #[error("the rig has influences for {found} vertices but the mesh has {expected}")]
    InfluenceCount { expected: usize, found: usize },
    #[error("vertex {0} has an invalid influence")]
    InvalidInfluence(usize),
    #[error("animation clip `{0}` is invalid")]
    InvalidClip(String),
    #[error("the rig has {0} clips; at most {MAX_CLIPS} are supported")]
    TooManyClips(usize),
}

impl Rig {
    /// Checks the rig against a mesh of `vertex_count` vertices.
    pub fn validate(&self, vertex_count: usize) -> Result<(), RigError> {
        if self.joints.len() > MAX_JOINTS {
            return Err(RigError::TooManyJoints(self.joints.len()));
        }
        if self.clips.len() > MAX_CLIPS {
            return Err(RigError::TooManyClips(self.clips.len()));
        }
        for (index, joint) in self.joints.iter().enumerate() {
            if !joint.is_finite() {
                return Err(RigError::InvalidJoint(index));
            }
        }
        self.order()?;
        for (index, binding) in self.bindings.iter().enumerate() {
            if binding.joint as usize >= self.joints.len()
                || !binding.before.is_finite()
                || !binding.inverse_bind.is_finite()
            {
                return Err(RigError::InvalidBinding(index));
            }
        }
        if self.influences.len() != vertex_count {
            return Err(RigError::InfluenceCount {
                expected: vertex_count,
                found: self.influences.len(),
            });
        }
        for (vertex, influences) in self.influences.iter().enumerate() {
            let valid = influences.iter().all(|influence| {
                influence.weight.is_finite()
                    && influence.weight >= 0.0
                    && (influence.weight == 0.0
                        || (influence.binding as usize) < self.bindings.len())
            });
            if !valid {
                return Err(RigError::InvalidInfluence(vertex));
            }
        }
        for clip in &self.clips {
            let joints_valid = clip.joints.iter().all(|animation| {
                (animation.joint as usize) < self.joints.len()
                    && animation
                        .translation
                        .as_ref()
                        .is_none_or(|track| track.is_valid(|value| value.is_finite()))
                    && animation.rotation.as_ref().is_none_or(|track| {
                        track.is_valid(|value| value.is_finite() && value.length_squared() > 0.0)
                    })
                    && animation
                        .scale
                        .as_ref()
                        .is_none_or(|track| track.is_valid(|value| value.is_finite()))
            });
            if !joints_valid || !clip.duration.is_finite() || clip.duration < 0.0 {
                return Err(RigError::InvalidClip(clip.name.clone()));
            }
        }
        Ok(())
    }

    /// Joint indices with every parent before its children.
    fn order(&self) -> Result<Vec<usize>, RigError> {
        // 0 unvisited, 1 in progress, 2 done.
        let mut state = vec![0_u8; self.joints.len()];
        let mut order = Vec::with_capacity(self.joints.len());
        for start in 0..self.joints.len() {
            let mut chain = Vec::new();
            let mut current = Some(start);
            while let Some(joint) = current {
                match state[joint] {
                    2 => break,
                    1 => return Err(RigError::InvalidParent(joint)),
                    _ => {}
                }
                state[joint] = 1;
                chain.push(joint);
                current = match self.joints[joint].parent {
                    Some(parent) if (parent as usize) < self.joints.len() => Some(parent as usize),
                    Some(_) => return Err(RigError::InvalidParent(joint)),
                    None => None,
                };
            }
            for joint in chain.into_iter().rev() {
                state[joint] = 2;
                order.push(joint);
            }
        }
        Ok(order)
    }

    pub fn clip(&self, name: &str) -> Option<&AnimationClip> {
        self.clips.iter().find(|clip| clip.name == name)
    }

    /// Global transform of every joint: at rest, or at a time in a clip.
    pub fn joint_globals(&self, sample: Option<ClipSample<'_>>) -> Result<Vec<Mat4>, RigError> {
        let mut translation: Vec<Vec3> =
            self.joints.iter().map(|joint| joint.translation).collect();
        let mut rotation: Vec<Quat> = self.joints.iter().map(|joint| joint.rotation).collect();
        let mut scale: Vec<Vec3> = self.joints.iter().map(|joint| joint.scale).collect();
        if let Some(ClipSample {
            clip,
            time,
            in_place,
        }) = sample
        {
            for animation in &clip.joints {
                let joint = animation.joint as usize;
                if joint >= self.joints.len() {
                    return Err(RigError::InvalidClip(clip.name.clone()));
                }
                let held = in_place && self.joints[joint].parent.is_none();
                if let Some(value) = animation
                    .translation
                    .as_ref()
                    .filter(|_| !held)
                    .and_then(|track| track.sample(time, Vec3::lerp))
                {
                    translation[joint] = value;
                }
                if let Some(value) = animation
                    .rotation
                    .as_ref()
                    .and_then(|track| track.sample(time, slerp_shortest))
                {
                    rotation[joint] = value;
                }
                if let Some(value) = animation
                    .scale
                    .as_ref()
                    .and_then(|track| track.sample(time, Vec3::lerp))
                {
                    scale[joint] = value;
                }
            }
        }
        let mut globals = vec![Mat4::IDENTITY; self.joints.len()];
        for index in self.order()? {
            let joint = &self.joints[index];
            let local = joint.local(translation[index], rotation[index], scale[index]);
            globals[index] = match joint.parent {
                Some(parent) => globals[parent as usize] * local,
                None => local,
            };
        }
        Ok(globals)
    }

    /// One matrix per binding for the given joint globals.
    pub fn skin_matrices(&self, globals: &[Mat4]) -> Vec<Mat4> {
        self.bindings
            .iter()
            .map(|binding| {
                let global = globals
                    .get(binding.joint as usize)
                    .copied()
                    .unwrap_or(Mat4::IDENTITY);
                binding.before * global * binding.inverse_bind
            })
            .collect()
    }

    /// `positions` (of the mesh as imported, multiplied by `scale`) posed at
    /// rest or at a time in a clip. Weights are normalised per vertex; a
    /// vertex without weight keeps its position.
    pub fn pose_positions(
        &self,
        positions: &[Vec3],
        scale: f32,
        sample: Option<ClipSample<'_>>,
    ) -> Result<Vec<Vec3>, RigError> {
        if positions.len() != self.influences.len() {
            return Err(RigError::InfluenceCount {
                expected: positions.len(),
                found: self.influences.len(),
            });
        }
        let matrices = self.skin_matrices(&self.joint_globals(sample)?);
        let scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        Ok(positions
            .iter()
            .zip(&self.influences)
            .map(|(position, influences)| {
                let total: f32 = influences.iter().map(|influence| influence.weight).sum();
                if total <= f32::EPSILON {
                    return *position;
                }
                let local = *position / scale;
                let posed = influences
                    .iter()
                    .filter(|influence| influence.weight > 0.0)
                    .filter_map(|influence| {
                        matrices
                            .get(influence.binding as usize)
                            .map(|matrix| matrix.transform_point3(local) * influence.weight)
                    })
                    .sum::<Vec3>()
                    / total;
                if posed.is_finite() {
                    posed * scale
                } else {
                    *position
                }
            })
            .collect())
    }
}

/// Keeps the `MAX_INFLUENCES` strongest of `weights` (binding, weight),
/// dropping non-positive and non-finite ones.
pub fn strongest_influences(
    weights: impl IntoIterator<Item = (u32, f32)>,
) -> [Influence; MAX_INFLUENCES] {
    let mut kept: Vec<Influence> = weights
        .into_iter()
        .filter(|(_, weight)| weight.is_finite() && *weight > 0.0)
        .map(|(binding, weight)| Influence { binding, weight })
        .collect();
    kept.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then(a.binding.cmp(&b.binding))
    });
    let mut influences = [Influence::default(); MAX_INFLUENCES];
    for (slot, influence) in influences.iter_mut().zip(kept) {
        *slot = influence;
    }
    influences
}

fn slerp_shortest(from: Quat, to: Quat, amount: f32) -> Quat {
    let to = if from.dot(to) < 0.0 { -to } else { to };
    from.normalize().slerp(to.normalize(), amount).normalize()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A two-joint arm along +X: the shoulder at the origin, the elbow one
    /// unit out. Vertex 0 follows the shoulder, vertex 1 the elbow, vertex
    /// 2 both equally, and vertex 3 nothing.
    fn arm() -> (Rig, Vec<Vec3>) {
        let shoulder = Joint::new("Shoulder", None, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
        let elbow = Joint::new("Elbow", Some(0), Vec3::X, Quat::IDENTITY, Vec3::ONE);
        let binding = |joint: u32, at: Vec3| SkinBinding {
            joint,
            before: Mat4::IDENTITY,
            inverse_bind: Mat4::from_translation(-at),
        };
        let one = |binding: u32| strongest_influences([(binding, 1.0)]);
        let bend = AnimationClip {
            name: "Bend".to_owned(),
            duration: 2.0,
            joints: vec![JointAnimation {
                joint: 1,
                rotation: Some(Track {
                    times: vec![0.0, 2.0],
                    values: vec![
                        Quat::IDENTITY,
                        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                    ],
                    step: false,
                }),
                ..JointAnimation::default()
            }],
        };
        let rig = Rig {
            joints: vec![shoulder, elbow],
            bindings: vec![binding(0, Vec3::ZERO), binding(1, Vec3::X)],
            influences: vec![
                one(0),
                one(1),
                strongest_influences([(0, 0.5), (1, 0.5)]),
                [Influence::default(); MAX_INFLUENCES],
            ],
            clips: vec![bend],
        };
        let positions = vec![
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(5.0, 5.0, 5.0),
        ];
        (rig, positions)
    }

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-5
    }

    #[test]
    fn tc_0316_skinning_poses_vertices_by_their_joints_and_keeps_the_bind_pose() {
        let (rig, positions) = arm();
        rig.validate(positions.len()).unwrap();

        // At rest, the rig is in its bind pose: nothing moves.
        let rest = rig.pose_positions(&positions, 1.0, None).unwrap();
        assert!(rest.iter().zip(&positions).all(|(a, b)| close(*a, *b)));

        // Halfway through, the elbow has turned 45°; at the end, 90°.
        let clip = rig.clip("Bend").unwrap();
        let end = rig
            .pose_positions(
                &positions,
                1.0,
                Some(ClipSample {
                    clip,
                    time: 2.0,
                    in_place: false,
                }),
            )
            .unwrap();
        assert!(close(end[0], positions[0]), "the shoulder vertex stays");
        assert!(close(end[1], Vec3::new(1.0, 1.0, 0.0)), "{}", end[1]);
        assert!(close(end[2], Vec3::new(1.5, 0.5, 0.0)), "a blend of both");
        assert!(close(end[3], positions[3]), "unweighted vertices stay");
        let half = rig
            .pose_positions(
                &positions,
                1.0,
                Some(ClipSample {
                    clip,
                    time: 1.0,
                    in_place: false,
                }),
            )
            .unwrap();
        let diagonal = std::f32::consts::FRAC_1_SQRT_2;
        assert!(close(half[1], Vec3::new(1.0 + diagonal, diagonal, 0.0)));
        // Times outside the clip clamp to its ends.
        let after = rig
            .pose_positions(
                &positions,
                1.0,
                Some(ClipSample {
                    clip,
                    time: 9.0,
                    in_place: false,
                }),
            )
            .unwrap();
        assert_eq!(after, end);

        // Scaled meshes pose the same shape, scaled.
        let scaled: Vec<Vec3> = positions.iter().map(|p| *p * 2.0).collect();
        let posed = rig
            .pose_positions(
                &scaled,
                2.0,
                Some(ClipSample {
                    clip,
                    time: 2.0,
                    in_place: false,
                }),
            )
            .unwrap();
        assert!(close(posed[1], end[1] * 2.0));

        // Root motion moves everything unless the clip plays in place.
        let mut travelling = rig.clone();
        travelling.clips[0].joints.push(JointAnimation {
            joint: 0,
            translation: Some(Track {
                times: vec![0.0],
                values: vec![Vec3::new(5.0, 0.0, 0.0)],
                step: false,
            }),
            ..JointAnimation::default()
        });
        let clip = &travelling.clips[0];
        let sample = |in_place| {
            Some(ClipSample {
                clip,
                time: 0.0,
                in_place,
            })
        };
        let moved = travelling
            .pose_positions(&positions, 1.0, sample(false))
            .unwrap();
        assert!(close(moved[0], positions[0] + Vec3::new(5.0, 0.0, 0.0)));
        let held = travelling
            .pose_positions(&positions, 1.0, sample(true))
            .unwrap();
        assert!(close(held[0], positions[0]));
        let clip = rig.clip("Bend").unwrap();
        assert_eq!(clip.key_span(), 2.0);
    }

    #[test]
    fn tc_0317_rigs_are_validated_and_influences_keep_the_strongest() {
        let (rig, positions) = arm();
        assert_eq!(
            rig.validate(3),
            Err(RigError::InfluenceCount {
                expected: 3,
                found: 4
            })
        );
        let mut cyclic = rig.clone();
        cyclic.joints[0].parent = Some(1);
        assert!(matches!(
            cyclic.validate(positions.len()),
            Err(RigError::InvalidParent(_))
        ));
        let mut missing = rig.clone();
        missing.joints[1].parent = Some(7);
        assert_eq!(
            missing.validate(positions.len()),
            Err(RigError::InvalidParent(1))
        );
        let mut binding = rig.clone();
        binding.bindings[0].joint = 9;
        assert_eq!(
            binding.validate(positions.len()),
            Err(RigError::InvalidBinding(0))
        );
        let mut influence = rig.clone();
        influence.influences[0][0].binding = 5;
        assert_eq!(
            influence.validate(positions.len()),
            Err(RigError::InvalidInfluence(0))
        );
        let mut unsorted = rig.clone();
        unsorted.clips[0].joints[0].rotation.as_mut().unwrap().times = vec![2.0, 0.0];
        assert_eq!(
            unsorted.validate(positions.len()),
            Err(RigError::InvalidClip("Bend".to_owned()))
        );
        let mut joint = rig;
        joint.joints[1].scale = Vec3::NAN;
        assert_eq!(
            joint.validate(positions.len()),
            Err(RigError::InvalidJoint(1))
        );

        let kept = strongest_influences([
            (0, 0.1),
            (1, 0.4),
            (2, -1.0),
            (3, 0.3),
            (4, f32::NAN),
            (5, 0.2),
            (6, 0.05),
        ]);
        let bindings: Vec<u32> = kept.iter().map(|influence| influence.binding).collect();
        assert_eq!(bindings, [1, 3, 5, 0]);

        // Steps hold their value until the next key.
        let track = Track {
            times: vec![0.0, 1.0],
            values: vec![Vec3::ZERO, Vec3::ONE],
            step: true,
        };
        assert_eq!(track.sample(0.9, Vec3::lerp), Some(Vec3::ZERO));
        assert_eq!(track.sample(1.0, Vec3::lerp), Some(Vec3::ONE));
    }

    #[test]
    fn joints_with_pivots_compose_their_local_transform_in_order() {
        // An FBX-style joint rotating 90° about a pivot at +X.
        let mut joint = Joint::new("Pivot", None, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
        joint.pre = Mat4::from_translation(Vec3::X);
        joint.mid = Mat4::from_translation(-Vec3::X);
        let local = joint.local(
            Vec3::ZERO,
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            Vec3::ONE,
        );
        // The pivot itself stays; the origin swings round it.
        assert!(close(local.transform_point3(Vec3::X), Vec3::X));
        assert!(close(
            local.transform_point3(Vec3::ZERO),
            Vec3::new(1.0, -1.0, 0.0)
        ));
    }
}
