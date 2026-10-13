//! Skeletons, skin weights, and animation stacks of FBX meshes.
//!
//! Every `Model` becomes a joint, with FBX's full local transform
//! `T · Roff · Rp · Rpre · R · Rpost⁻¹ · Rp⁻¹ · Soff · Sp · S · Sp⁻¹` and its
//! rotation order. A skinned mesh binds through its clusters: the vertices
//! (in geometry space, as imported) follow
//! `M⁻¹ · global(bone) · TransformLink⁻¹ · M`, the identity in the bind pose,
//! where `M` is the mesh's bind transform: the cluster `Transform` in the
//! FBX SDK's convention, or `TransformLink · Transform` where an exporter
//! stores the inverse bind matrix there (both are read; see
//! [`prefer_second_reading`]). A mesh without a skin follows its own model rigidly.
//! Each animation stack becomes a clip from the first layer's curves; the
//! X, Y, and Z curves of a property are sampled at the union of their keys
//! and interpolated linearly (curve tangents are not read).

use std::collections::{BTreeMap, BTreeSet};

use fbxcel_dom::{
    fbxcel::{low::v7400::AttributeValue, tree::v7400::NodeHandle},
    v7400::object::{ObjectHandle, ObjectId, geometry::MeshHandle},
};
use glam::{Mat4, Quat, Vec3};
use substancecraft_core::{
    AnimationClip, Joint, JointAnimation, Rig, SkinBinding, Track, strongest_influences,
};

use crate::fbx::FbxImportError;

/// FBX time units per second.
const KTIME_PER_SECOND: f64 = 46_186_158_000.0;
/// Keys a sampled property may have.
const MAX_PROPERTY_KEYS: usize = 100_000;

/// The rig of `mesh`, whose imported vertex `i` came from control point
/// `control_points[i]`; `None` when the file neither skins nor animates.
pub(crate) fn read_rig(
    mesh: &MeshHandle<'_>,
    control_points: &[u32],
) -> Result<Option<Rig>, FbxImportError> {
    let document = mesh.document();
    let mut skins: Vec<ObjectHandle<'_>> = sources(mesh)
        .filter(|object| object.class() == "Deformer" && object.subclass() == "Skin")
        .collect();
    skins.sort_by_key(ObjectHandle::object_id);
    let mut stacks: Vec<ObjectHandle<'_>> = document
        .objects()
        .filter(|object| matches!(object.class(), "AnimationStack" | "AnimStack"))
        .collect();
    // The DOM does not keep file order; ids give a stable one.
    stacks.sort_by_key(ObjectHandle::object_id);
    if skins.is_empty() && stacks.is_empty() {
        return Ok(None);
    }

    let mut models: Vec<ObjectHandle<'_>> = document
        .objects()
        .filter(|object| object.class() == "Model")
        .collect();
    models.sort_by_key(ObjectHandle::object_id);
    let index: BTreeMap<ObjectId, u32> = models
        .iter()
        .enumerate()
        .map(|(position, model)| (model.object_id(), position as u32))
        .collect();
    let transforms: Vec<ModelTransform> = models.iter().map(ModelTransform::read).collect();
    let joints: Vec<Joint> = models
        .iter()
        .zip(&transforms)
        .map(|(model, transform)| {
            let parent = destinations(model)
                .find(|object| object.class() == "Model")
                .and_then(|parent| index.get(&parent.object_id()).copied());
            transform.joint(model.name().unwrap_or("Joint"), parent)
        })
        .collect();
    let mut rig = Rig {
        joints,
        ..Rig::default()
    };

    if skins.is_empty() {
        // No skin: the mesh moves with its own model.
        let owner = destinations(mesh)
            .find(|object| object.class() == "Model")
            .and_then(|model| index.get(&model.object_id()).copied());
        if let Some(joint) = owner {
            let rest = rig
                .joint_globals(None)
                .map_err(|error| FbxImportError::InvalidMesh(error.to_string()))?;
            rig.bindings.push(SkinBinding {
                joint,
                before: rest[joint as usize].inverse(),
                inverse_bind: Mat4::IDENTITY,
            });
            rig.influences = vec![strongest_influences([(0, 1.0)]); control_points.len()];
        } else {
            rig.influences = vec![strongest_influences([]); control_points.len()];
        }
    } else {
        // Clusters first, then the mesh's bind transform they share.
        struct Cluster<'a> {
            joint: u32,
            transform: Mat4,
            link: Mat4,
            node: NodeHandle<'a>,
        }
        let mut clusters = Vec::new();
        for skin in &skins {
            let mut objects: Vec<ObjectHandle<'_>> = sources(skin)
                .filter(|object| {
                    matches!(object.class(), "SubDeformer" | "Deformer")
                        && object.subclass() == "Cluster"
                })
                .collect();
            objects.sort_by_key(ObjectHandle::object_id);
            for cluster in objects {
                let Some(joint) = sources(&cluster)
                    .find(|object| object.class() == "Model")
                    .and_then(|bone| index.get(&bone.object_id()).copied())
                else {
                    continue;
                };
                let node = cluster.node();
                clusters.push(Cluster {
                    joint,
                    transform: matrix(&node, "Transform").unwrap_or(Mat4::IDENTITY),
                    link: matrix(&node, "TransformLink").unwrap_or(Mat4::IDENTITY),
                    node,
                });
            }
        }
        // The FBX SDK stores the mesh's bind transform as `Transform`; some
        // exporters store the inverse bind matrix `TransformLink⁻¹ · mesh`
        // there instead. The mesh's transform is one matrix for every
        // cluster, so the reading that agrees across clusters wins; a tie
        // (one cluster) goes to the reading nearer the identity.
        let as_stored: Vec<Mat4> = clusters.iter().map(|cluster| cluster.transform).collect();
        let as_inverse_bind: Vec<Mat4> = clusters
            .iter()
            .map(|cluster| cluster.link * cluster.transform)
            .collect();
        let mesh_bind = if prefer_second_reading(&as_stored, &as_inverse_bind) {
            as_inverse_bind
        } else {
            as_stored
        };
        let mut weights: BTreeMap<u32, Vec<(u32, f32)>> = BTreeMap::new();
        for (cluster, mesh) in clusters.iter().zip(mesh_bind) {
            let binding = rig.bindings.len() as u32;
            rig.bindings.push(SkinBinding {
                joint: cluster.joint,
                before: mesh.inverse(),
                inverse_bind: cluster.link.inverse() * mesh,
            });
            let indexes = child_array(&cluster.node, "Indexes", |value| value.get_arr_i32())
                .unwrap_or_default();
            let cluster_weights =
                child_array(&cluster.node, "Weights", |value| value.get_arr_f64())
                    .unwrap_or_default();
            for (control_point, weight) in indexes.iter().zip(cluster_weights) {
                if let Ok(control_point) = u32::try_from(*control_point) {
                    weights
                        .entry(control_point)
                        .or_default()
                        .push((binding, *weight as f32));
                }
            }
        }
        rig.influences = control_points
            .iter()
            .map(|control_point| {
                strongest_influences(weights.get(control_point).cloned().unwrap_or_default())
            })
            .collect();
    }

    for stack in &stacks {
        rig.clips.push(read_clip(stack, &index, &transforms)?);
    }
    rig.validate(control_points.len())
        .map_err(|error| FbxImportError::InvalidMesh(error.to_string()))?;
    Ok(Some(rig))
}

/// Largest element-wise difference of two matrices.
fn matrix_distance(a: Mat4, b: Mat4) -> f32 {
    (a - b)
        .to_cols_array()
        .iter()
        .fold(0.0, |largest, value| largest.max(value.abs()))
}

/// How far apart the most different two of `matrices` are.
fn spread(matrices: &[Mat4]) -> f32 {
    matrices.first().map_or(0.0, |first| {
        matrices.iter().fold(0.0, |largest, matrix| {
            largest.max(matrix_distance(*first, *matrix))
        })
    })
}

/// True when `second` is the better reading of a mesh bind transform: it
/// agrees across clusters where `first` does not, or, as closely as
/// `first` does, it is nearer the identity.
fn prefer_second_reading(first: &[Mat4], second: &[Mat4]) -> bool {
    const AGREEMENT: f32 = 1e-3;
    let (first_spread, second_spread) = (spread(first), spread(second));
    if (first_spread - second_spread).abs() > AGREEMENT * (1.0 + first_spread.max(second_spread)) {
        return second_spread < first_spread;
    }
    let off_identity = |matrices: &[Mat4]| {
        matrices
            .first()
            .map_or(0.0, |matrix| matrix_distance(*matrix, Mat4::IDENTITY))
    };
    off_identity(second) < off_identity(first)
}

fn sources<'a>(object: &ObjectHandle<'a>) -> impl Iterator<Item = ObjectHandle<'a>> + 'a {
    object
        .source_objects()
        .filter_map(|connected| connected.object_handle())
}

fn destinations<'a>(object: &ObjectHandle<'a>) -> impl Iterator<Item = ObjectHandle<'a>> + 'a {
    object
        .destination_objects()
        .filter_map(|connected| connected.object_handle())
}

fn number(value: &AttributeValue) -> Option<f64> {
    value
        .get_f64()
        .or_else(|| value.get_f32().map(f64::from))
        .or_else(|| value.get_i32().map(f64::from))
        .or_else(|| value.get_i64().map(|value| value as f64))
        .or_else(|| value.get_i16().map(f64::from))
}

fn child_array<'a, T>(
    node: &NodeHandle<'a>,
    name: &str,
    read: impl Fn(&'a AttributeValue) -> Option<&'a [T]>,
) -> Option<&'a [T]> {
    node.first_child_by_name(name)
        .and_then(|child| child.attributes().first())
        .and_then(read)
}

/// A 4×4 matrix stored as 16 column-major doubles.
fn matrix(node: &NodeHandle<'_>, name: &str) -> Option<Mat4> {
    let values = child_array(node, name, |value| value.get_arr_f64())?;
    let values: [f64; 16] = values.try_into().ok()?;
    let matrix = Mat4::from_cols_array(&values.map(|value| value as f32));
    matrix.is_finite().then_some(matrix)
}

/// `Properties70` entries of an object: name → numeric values.
fn properties(object: &ObjectHandle<'_>) -> BTreeMap<String, Vec<f64>> {
    let mut properties = BTreeMap::new();
    let Some(block) = object.node().first_child_by_name("Properties70") else {
        return properties;
    };
    for property in block.children_by_name("P") {
        let attributes = property.attributes();
        let Some(name) = attributes.first().and_then(AttributeValue::get_string) else {
            continue;
        };
        let values = attributes.iter().skip(4).filter_map(number).collect();
        properties.insert(name.to_owned(), values);
    }
    properties
}

/// Rotation from Euler angles in degrees, applied in FBX's `order`
/// (0 XYZ, 1 XZY, 2 YZX, 3 YXZ, 4 ZXY, 5 ZYX; others as XYZ).
fn euler(degrees: Vec3, order: i32) -> Quat {
    let x = Quat::from_rotation_x(degrees.x.to_radians());
    let y = Quat::from_rotation_y(degrees.y.to_radians());
    let z = Quat::from_rotation_z(degrees.z.to_radians());
    match order {
        1 => y * z * x,
        2 => x * z * y,
        3 => z * x * y,
        4 => y * x * z,
        5 => x * y * z,
        _ => z * y * x,
    }
}

/// The local-transform properties of one FBX model.
struct ModelTransform {
    translation: Vec3,
    /// Euler degrees, before `order`.
    rotation: Vec3,
    scale: Vec3,
    order: i32,
    pre: Mat4,
    mid: Mat4,
    post: Mat4,
}

impl ModelTransform {
    fn read(model: &ObjectHandle<'_>) -> Self {
        let properties = properties(model);
        let vector = |name: &str, default: Vec3| {
            properties
                .get(name)
                .filter(|values| values.len() >= 3)
                .map(|values| Vec3::new(values[0] as f32, values[1] as f32, values[2] as f32))
                .filter(|value| value.is_finite())
                .unwrap_or(default)
        };
        let order = properties
            .get("RotationOrder")
            .and_then(|values| values.first())
            .map_or(0, |value| *value as i32);
        let rotation_offset = vector("RotationOffset", Vec3::ZERO);
        let rotation_pivot = vector("RotationPivot", Vec3::ZERO);
        let scaling_offset = vector("ScalingOffset", Vec3::ZERO);
        let scaling_pivot = vector("ScalingPivot", Vec3::ZERO);
        // Pre- and post-rotations are always XYZ.
        let pre_rotation = euler(vector("PreRotation", Vec3::ZERO), 0);
        let post_rotation = euler(vector("PostRotation", Vec3::ZERO), 0);
        Self {
            translation: vector("Lcl Translation", Vec3::ZERO),
            rotation: vector("Lcl Rotation", Vec3::ZERO),
            scale: vector("Lcl Scaling", Vec3::ONE),
            order,
            pre: Mat4::from_translation(rotation_offset)
                * Mat4::from_translation(rotation_pivot)
                * Mat4::from_quat(pre_rotation),
            mid: Mat4::from_quat(post_rotation.inverse())
                * Mat4::from_translation(-rotation_pivot)
                * Mat4::from_translation(scaling_offset)
                * Mat4::from_translation(scaling_pivot),
            post: Mat4::from_translation(-scaling_pivot),
        }
    }

    fn joint(&self, name: &str, parent: Option<u32>) -> Joint {
        Joint {
            pre: self.pre,
            mid: self.mid,
            post: self.post,
            ..Joint::new(
                name,
                parent,
                self.translation,
                euler(self.rotation, self.order),
                self.scale,
            )
        }
    }
}

/// One FBX animation curve: key times (FBX units) and values.
struct Curve {
    times: Vec<i64>,
    values: Vec<f32>,
}

impl Curve {
    fn read(curve: &ObjectHandle<'_>) -> Option<Self> {
        let node = curve.node();
        let times = child_array(&node, "KeyTime", |value| value.get_arr_i64())?.to_vec();
        let values = child_array(&node, "KeyValueFloat", |value| value.get_arr_f32())
            .map(<[f32]>::to_vec)
            .or_else(|| {
                child_array(&node, "KeyValueDouble", |value| value.get_arr_f64())
                    .map(|values| values.iter().map(|value| *value as f32).collect())
            })?;
        (times.len() == values.len() && !times.is_empty()).then_some(Self { times, values })
    }

    fn sample(&self, time: i64) -> f32 {
        let after = self.times.partition_point(|key| *key <= time);
        if after == 0 {
            return self.values[0];
        }
        if after >= self.times.len() {
            return self.values[self.values.len() - 1];
        }
        let (start, end) = (self.times[after - 1], self.times[after]);
        let (from, to) = (self.values[after - 1], self.values[after]);
        if end <= start {
            return from;
        }
        from + (to - from) * ((time - start) as f64 / (end - start) as f64) as f32
    }
}

/// A property's X, Y, and Z curves and its curve node's defaults.
type AnimatedProperty = ([Option<Curve>; 3], [Option<f32>; 3]);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Property {
    Translation,
    Rotation,
    Scale,
}

/// One animation stack as a clip, read from its first layer.
fn read_clip(
    stack: &ObjectHandle<'_>,
    index: &BTreeMap<ObjectId, u32>,
    transforms: &[ModelTransform],
) -> Result<AnimationClip, FbxImportError> {
    let name = stack
        .name()
        .filter(|name| !name.trim().is_empty())
        .unwrap_or("Take")
        .to_owned();
    // (joint, property) → component curves and the curve node's defaults.
    let mut animated: BTreeMap<(u32, Property), AnimatedProperty> = BTreeMap::new();
    if let Some(layer) =
        sources(stack).find(|object| matches!(object.class(), "AnimationLayer" | "AnimLayer"))
    {
        for curve_node in sources(&layer)
            .filter(|object| matches!(object.class(), "AnimationCurveNode" | "AnimCurveNode"))
        {
            let target = curve_node.destination_objects().find_map(|connected| {
                let property = match connected.label()? {
                    "Lcl Translation" => Property::Translation,
                    "Lcl Rotation" => Property::Rotation,
                    "Lcl Scaling" => Property::Scale,
                    _ => return None,
                };
                let joint = index.get(&connected.object_id()).copied()?;
                Some((joint, property))
            });
            let Some(target) = target else {
                continue;
            };
            let defaults = properties(&curve_node);
            let mut curves: [Option<Curve>; 3] = [None, None, None];
            let mut fallback = [None; 3];
            for (component, label) in ["d|X", "d|Y", "d|Z"].into_iter().enumerate() {
                fallback[component] = defaults
                    .get(label)
                    .and_then(|values| values.first())
                    .map(|value| *value as f32);
                curves[component] = curve_node
                    .source_objects()
                    .filter(|connected| connected.label() == Some(label))
                    .filter_map(|connected| connected.object_handle())
                    .find(|object| matches!(object.class(), "AnimationCurve" | "AnimCurve"))
                    .and_then(|curve| Curve::read(&curve));
            }
            animated.insert(target, (curves, fallback));
        }
    }

    let stack_properties = properties(stack);
    let key_times: Vec<i64> = animated
        .values()
        .flat_map(|(curves, _)| curves.iter().flatten())
        .flat_map(|curve| curve.times.iter().copied())
        .collect();
    let start = stack_properties
        .get("LocalStart")
        .and_then(|values| values.first())
        .map(|value| *value as i64)
        .or_else(|| key_times.iter().copied().min())
        .unwrap_or(0);
    let stop = stack_properties
        .get("LocalStop")
        .and_then(|values| values.first())
        .map(|value| *value as i64)
        .or_else(|| key_times.iter().copied().max())
        .unwrap_or(start);
    let seconds = |time: i64| ((time - start) as f64 / KTIME_PER_SECOND) as f32;

    let mut joints: BTreeMap<u32, JointAnimation> = BTreeMap::new();
    for ((joint, property), (curves, fallback)) in animated {
        let times: BTreeSet<i64> = curves
            .iter()
            .flatten()
            .flat_map(|curve| curve.times.iter().copied())
            .collect();
        if times.is_empty() || times.len() > MAX_PROPERTY_KEYS {
            continue;
        }
        let rest = &transforms[joint as usize];
        let resting = match property {
            Property::Translation => rest.translation,
            Property::Rotation => rest.rotation,
            Property::Scale => rest.scale,
        };
        let value_at = |time: i64| {
            let component = |axis: usize| match &curves[axis] {
                Some(curve) => curve.sample(time),
                None => fallback[axis].unwrap_or(resting[axis]),
            };
            Vec3::new(component(0), component(1), component(2))
        };
        let track_times: Vec<f32> = times.iter().map(|time| seconds(*time)).collect();
        let values: Vec<Vec3> = times.iter().map(|time| value_at(*time)).collect();
        let animation = joints.entry(joint).or_insert_with(|| JointAnimation {
            joint,
            ..JointAnimation::default()
        });
        match property {
            Property::Translation => {
                animation.translation = Some(Track {
                    times: track_times,
                    values,
                    step: false,
                });
            }
            Property::Rotation => {
                animation.rotation = Some(Track {
                    times: track_times,
                    values: values
                        .into_iter()
                        .map(|degrees| euler(degrees, rest.order))
                        .collect(),
                    step: false,
                });
            }
            Property::Scale => {
                animation.scale = Some(Track {
                    times: track_times,
                    values,
                    step: false,
                });
            }
        }
    }
    let mut clip = AnimationClip {
        name,
        duration: seconds(stop).max(0.0),
        joints: joints.into_values().collect(),
    };
    if !clip.duration.is_finite() || clip.duration <= 0.0 {
        clip.duration = clip.key_span();
    }
    Ok(clip)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn euler_orders_apply_their_first_axis_first() {
        let degrees = Vec3::new(90.0, 90.0, 0.0);
        // XYZ: X first, then Y: +Y turns to +Z under X, then to +X under Y.
        let xyz = euler(degrees, 0) * Vec3::Y;
        assert!((xyz - Vec3::X).length() < 1e-5, "{xyz}");
        // YXZ: Y first leaves +Y alone, then X turns it to +Z.
        let yxz = euler(degrees, 3) * Vec3::Y;
        assert!((yxz - Vec3::Z).length() < 1e-5, "{yxz}");
    }

    #[test]
    fn tc_0322_cluster_transforms_are_read_in_either_convention() {
        let links = [
            Mat4::from_translation(Vec3::X),
            Mat4::from_rotation_z(1.0) * Mat4::from_translation(Vec3::Y * 2.0),
        ];
        let mesh = Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        // FBX SDK: `Transform` is the mesh's bind transform for every cluster.
        let stored = [mesh, mesh];
        let products: Vec<Mat4> = links.iter().zip(stored).map(|(l, t)| *l * t).collect();
        assert!(!prefer_second_reading(&stored, &products));
        // Inverse bind matrices: `TransformLink · Transform` is the mesh.
        let inverse_binds: Vec<Mat4> = links.iter().map(|l| l.inverse() * mesh).collect();
        let products: Vec<Mat4> = links
            .iter()
            .zip(&inverse_binds)
            .map(|(l, t)| *l * *t)
            .collect();
        assert!(prefer_second_reading(&inverse_binds, &products));
        // One cluster agrees with itself either way; the identity wins.
        assert!(!prefer_second_reading(&[Mat4::IDENTITY], &[links[1]]));
        assert!(prefer_second_reading(
            &[links[1].inverse()],
            &[Mat4::IDENTITY]
        ));
    }

    #[test]
    fn curves_interpolate_linearly_and_clamp() {
        let curve = Curve {
            times: vec![0, 10, 20],
            values: vec![0.0, 1.0, 3.0],
        };
        assert_eq!(curve.sample(-5), 0.0);
        assert_eq!(curve.sample(5), 0.5);
        assert_eq!(curve.sample(15), 2.0);
        assert_eq!(curve.sample(99), 3.0);
    }
}
