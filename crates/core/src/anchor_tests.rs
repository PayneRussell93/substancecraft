//! Anchors to effects within a layer (feature 19).

#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;

use glam::Vec2;

use crate::{
    AnchorEffect, AnchorMask, BakeContext, DEFAULT_HEIGHT_STRENGTH, LayerFill, LayerFilter,
    LayerId, MaterialChannel, MaterialLayer, PaintStroke, TextureResolution, TextureSet,
    TextureSetId, UserValues,
};

const SIZE: u32 = 17;

/// A fill of roughness 0.2 with a 0.9 dab in the middle, then Invert; and
/// a layer above masked by an anchor to it.
fn set(effect: Option<AnchorEffect>, channel: MaterialChannel) -> TextureSet {
    let mut source = MaterialLayer::new_fill(
        LayerId(1),
        "Source",
        LayerFill {
            roughness: 0.2,
            user: {
                let mut user = UserValues::default();
                user.0[0] = [255; 3];
                user
            },
            ..LayerFill::default()
        },
    );
    source.channels = BTreeSet::from([MaterialChannel::Roughness, MaterialChannel::User(0)]);
    let mut dab = PaintStroke::new(vec![Vec2::splat(0.5)], 3.0, [0, 0, 0, 255]);
    dab.tip.hardness = 1.0;
    dab.roughness = 0.9;
    source.paint_strokes.push(dab);
    source.filters = vec![LayerFilter::Invert];
    let mut masked = MaterialLayer::new_fill(LayerId(2), "Masked", LayerFill::default());
    masked.mask_anchor = Some(AnchorMask {
        effect,
        ..AnchorMask::new(LayerId(1), channel)
    });
    TextureSet {
        id: TextureSetId(1),
        name: "Anchors".to_owned(),
        material_slot: "Default".to_owned(),
        resolution: TextureResolution {
            width: SIZE,
            height: SIZE,
        },
        channels: BTreeSet::from([MaterialChannel::Roughness]),
        layers: vec![source, masked],
        height_strength: DEFAULT_HEIGHT_STRENGTH,
        export: None,
        stencils: Vec::new(),
        user_channels: Vec::new(),
    }
}

/// The anchor mask at the centre and at a corner.
fn read(effect: Option<AnchorEffect>, channel: MaterialChannel) -> (f32, f32) {
    let set = set(effect, channel);
    let mask = set
        .layer_mask_in(
            &set.layers[1],
            BakeContext::default(),
            None,
            set.texel_rect(),
        )
        .unwrap()
        .unwrap();
    let centre = (SIZE / 2 * SIZE + SIZE / 2) as usize;
    (mask[centre], mask[0])
}

fn close(actual: (f32, f32), expected: (f32, f32)) -> bool {
    (actual.0 - expected.0).abs() < 1e-5 && (actual.1 - expected.1).abs() < 1e-5
}

#[test]
fn tc_0363_anchors_read_a_layer_at_one_of_its_effects() {
    let rough = MaterialChannel::Roughness;
    // The whole layer: inverted fill and dab.
    assert!(close(read(None, rough), (0.1, 0.8)));
    // The fill alone, the fill with its dab, and through the filter.
    assert!(close(read(Some(AnchorEffect::Fill), rough), (0.2, 0.2)));
    assert!(close(read(Some(AnchorEffect::Paint), rough), (0.9, 0.2)));
    assert!(close(
        read(Some(AnchorEffect::Filter(0)), rough),
        (0.1, 0.8)
    ));
    // A filter removed since the anchor was set reads every filter left.
    assert!(close(
        read(Some(AnchorEffect::Filter(5)), rough),
        (0.1, 0.8)
    ));
    // User channels can be anchored to as well: the fill writes 1 and the
    // dab 0, and Invert flips them.
    assert!(close(
        read(Some(AnchorEffect::Fill), MaterialChannel::User(0)),
        (1.0, 1.0)
    ));
    assert!(close(read(None, MaterialChannel::User(0)), (1.0, 0.0)));

    // The effects offered follow the layer: fill, paint, then each filter;
    // groups only have filters.
    let source = &set(None, rough).layers[0];
    assert_eq!(
        AnchorEffect::of(source),
        [
            AnchorEffect::Fill,
            AnchorEffect::Paint,
            AnchorEffect::Filter(0)
        ]
    );
    let mut group = MaterialLayer::new_group(LayerId(5), "Group");
    group.filters = vec![LayerFilter::Invert, LayerFilter::Invert];
    assert_eq!(
        AnchorEffect::of(&group),
        [AnchorEffect::Filter(0), AnchorEffect::Filter(1)]
    );

    // Saved as `effect`, absent for whole-layer anchors; old files load.
    let anchor = AnchorMask {
        effect: Some(AnchorEffect::Filter(2)),
        ..AnchorMask::new(LayerId(1), rough)
    };
    let json = serde_json::to_string(&anchor).unwrap();
    assert!(json.contains(r#""effect":{"Filter":2}"#), "{json}");
    assert_eq!(serde_json::from_str::<AnchorMask>(&json).unwrap(), anchor);
    let old = r#"{"layer":1,"channel":"Roughness","low":0.0,"high":1.0}"#;
    assert_eq!(
        serde_json::from_str::<AnchorMask>(old).unwrap().effect,
        None
    );
}
