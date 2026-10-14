//! User-defined channels (feature 04).

#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;

use glam::Vec2;

use crate::{
    BakeContext, BakeError, ChannelSlot, DEFAULT_HEIGHT_STRENGTH, ExportOutput, ExportPreset,
    GROUP_CHANNELS, LayerFill, LayerId, MaterialChannel, MaterialLayer, MaterialProject,
    OutputLayout, PaintStroke, ProjectError, SourceComponent, StrokeSource, TextureResolution,
    TextureSet, TextureSetId, UserChannel, UserValues, user_channels_valid,
};

const WET: MaterialChannel = MaterialChannel::User(2);

fn wetness() -> UserChannel {
    UserChannel {
        slot: 2,
        name: "Wetness".to_owned(),
        gray: true,
        default: [40, 40, 40],
    }
}

fn set(layers: Vec<MaterialLayer>) -> TextureSet {
    TextureSet {
        id: TextureSetId(1),
        name: "Set".to_owned(),
        material_slot: "Default".to_owned(),
        resolution: TextureResolution {
            width: 16,
            height: 16,
        },
        channels: BTreeSet::from([MaterialChannel::BaseColor]),
        layers,
        height_strength: DEFAULT_HEIGHT_STRENGTH,
        export: None,
        stencils: Vec::new(),
        user_channels: vec![wetness()],
    }
}

fn values(slot: usize, value: [u8; 3]) -> UserValues {
    let mut values = UserValues::default();
    values.0[slot] = value;
    values
}

#[test]
fn tc_0358_user_channels_are_named_user0_to_user7_in_files() {
    assert_eq!(serde_json::to_string(&WET).unwrap(), "\"User2\"");
    assert_eq!(
        serde_json::from_str::<MaterialChannel>("\"User7\"").unwrap(),
        MaterialChannel::User(7)
    );
    for bad in [
        "\"User8\"",
        "\"User02\"",
        "\"User\"",
        "\"user2\"",
        "\"Wetness\"",
    ] {
        assert!(
            serde_json::from_str::<MaterialChannel>(bad).is_err(),
            "{bad}"
        );
    }
    // Built-in names are unchanged, so older projects still load.
    assert_eq!(
        serde_json::to_string(&MaterialChannel::BaseColor).unwrap(),
        "\"BaseColor\""
    );
    // Channel maps keyed by channel stay valid JSON objects.
    let modes = std::collections::BTreeMap::from([(WET, crate::BlendMode::Multiply)]);
    let json = serde_json::to_string(&modes).unwrap();
    assert_eq!(json, r#"{"User2":"Multiply"}"#);
    assert_eq!(
        serde_json::from_str::<std::collections::BTreeMap<MaterialChannel, crate::BlendMode>>(
            &json
        )
        .unwrap(),
        modes
    );

    // Slots and names are unique; grey channels default to a grey.
    assert!(user_channels_valid(&[wetness()]));
    let renamed = UserChannel {
        slot: 3,
        ..wetness()
    };
    assert!(!user_channels_valid(&[wetness(), renamed]), "same name");
    let moved = UserChannel {
        name: "Region".to_owned(),
        ..wetness()
    };
    assert!(!user_channels_valid(&[wetness(), moved]), "same slot");
    for bad in [
        UserChannel {
            slot: 8,
            ..wetness()
        },
        UserChannel {
            name: "  ".to_owned(),
            ..wetness()
        },
        UserChannel {
            default: [1, 2, 3],
            ..wetness()
        },
    ] {
        assert!(!user_channels_valid(&[bad]));
    }
    let mut project = MaterialProject::empty();
    let mut invalid = set(Vec::new());
    invalid.user_channels.push(wetness());
    project.texture_sets.push(invalid);
    assert_eq!(
        project.validate(),
        Err(ProjectError::InvalidTextureSetSettings(TextureSetId(1)))
    );
}

#[test]
fn tc_0359_fills_and_strokes_composite_user_channels_over_their_default() {
    // A fill writes 200 into the left half's wetness through a stroke-free
    // layer; a stroke paints 255 on the right.
    let mut fill = MaterialLayer::new_fill(
        LayerId(1),
        "Wet fill",
        LayerFill {
            user: values(2, [200; 3]),
            ..LayerFill::default()
        },
    );
    fill.channels = BTreeSet::from([WET]);
    fill.polygon_mask = None;
    let mut paint = MaterialLayer::new_paint(LayerId(2), "Puddle");
    paint.channels = BTreeSet::from([WET]);
    let mut stroke = PaintStroke::new(vec![Vec2::new(0.8, 0.5)], 2.0, [0, 0, 0, 255]);
    stroke.tip.hardness = 1.0;
    stroke.user = values(2, [255; 3]);
    paint.paint_strokes.push(stroke);

    let baked = set(vec![paint.clone()])
        .bake_channel(WET, BakeContext::default())
        .unwrap();
    let at = |texture: &crate::BakedTexture, u: f32| texture.sample_uv(Vec2::new(u, 0.5)).unwrap();
    assert_eq!(at(&baked, 0.8), [255, 255, 255, 255], "painted");
    assert_eq!(at(&baked, 0.2), [40, 40, 40, 255], "the channel's default");

    // A group passes its children's user channels through by default.
    assert!(GROUP_CHANNELS.contains(&WET));
    let mut group = MaterialLayer::new_group(LayerId(3), "Group");
    group.children = Some(vec![fill, paint]);
    let baked = set(vec![group])
        .bake_channel(WET, BakeContext::default())
        .unwrap();
    assert_eq!(at(&baked, 0.2), [200, 200, 200, 255]);
    assert_eq!(at(&baked, 0.8), [255, 255, 255, 255]);

    // Layers never write channels they do not enable.
    let base = set(Vec::new())
        .bake_channel(MaterialChannel::BaseColor, BakeContext::default())
        .unwrap();
    assert!(base.rgba8.chunks(4).all(|texel| texel[3] == 0));
}

#[test]
fn tc_0360_presets_export_user_channels_and_preflight_flags_missing_ones() {
    let mut paint = MaterialLayer::new_paint(LayerId(1), "Puddle");
    paint.channels = BTreeSet::from([WET]);
    let mut stroke = PaintStroke::new(vec![Vec2::splat(0.5)], 3.0, [0, 0, 0, 255]);
    stroke.tip.hardness = 1.0;
    stroke.user = values(2, [255; 3]);
    paint.paint_strokes.push(stroke);
    let set = set(vec![paint]);
    let preset = ExportPreset {
        name: "Game".to_owned(),
        outputs: vec![ExportOutput::new(
            "$textureSet_wet",
            OutputLayout::Gray,
            [
                ChannelSlot::channel(WET, SourceComponent::Red),
                ChannelSlot::ZERO,
                ChannelSlot::ZERO,
                ChannelSlot::ONE,
            ],
        )],
        max_resolution: None,
    };
    let exported = set.bake_export(&preset, BakeContext::default()).unwrap();
    assert_eq!(exported.len(), 1);
    let texture = &exported[0].texture;
    assert_eq!(texture.sample_uv(Vec2::splat(0.5)).unwrap()[0], 255);
    assert_eq!(texture.sample_uv(Vec2::splat(0.05)).unwrap()[0], 40);
    assert!(!exported[0].srgb[0], "user channels are data");
    assert!(
        set.export_preflight(&preset, BakeContext::default())
            .is_empty()
    );

    // A preset reading a slot the set does not define warns.
    let mut other = preset.clone();
    other.outputs[0].slots[0] =
        ChannelSlot::channel(MaterialChannel::User(5), SourceComponent::Red);
    let issues = set.export_preflight(&other, BakeContext::default());
    assert!(
        issues
            .iter()
            .any(|issue| issue.message.contains("User channel 5 is not defined")),
        "{issues:?}"
    );
}

#[test]
fn tc_0361_flattening_keeps_user_channels_and_refuses_sampling_strokes() {
    let mut paint = MaterialLayer::new_paint(LayerId(1), "Puddle");
    paint.channels = BTreeSet::from([MaterialChannel::BaseColor, WET]);
    let mut stroke = PaintStroke::new(vec![Vec2::splat(0.5)], 3.0, [255, 0, 0, 255]);
    stroke.user = values(2, [255; 3]);
    paint.paint_strokes.push(stroke);
    let flattened = set(vec![paint.clone()])
        .flatten_layer(LayerId(1), BakeContext::default())
        .unwrap();
    let channels: Vec<MaterialChannel> = flattened
        .channels
        .iter()
        .map(|(channel, _)| *channel)
        .collect();
    assert_eq!(channels, [MaterialChannel::BaseColor, WET]);
    let fill = flattened.to_fill_layer(
        LayerId(9),
        "Flat",
        &[crate::ResourceId(1), crate::ResourceId(2)],
    );
    assert_eq!(
        fill.fill.unwrap().textures.for_channel(WET),
        Some(crate::ResourceId(2))
    );
    assert!(fill.channels.contains(&WET));

    // Clone strokes read the stack beneath, which a flatten leaves out.
    paint.paint_strokes[0].source = Some(StrokeSource::Smudge { strength: 0.5 });
    assert_eq!(
        set(vec![paint])
            .flatten_layer(LayerId(1), BakeContext::default())
            .err(),
        Some(BakeError::FlattenReadsStack)
    );
}
