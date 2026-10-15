//! Viewport shader definitions: named, data-only descriptions of how the
//! viewport material shades the baked channels. The renderer's one material
//! exposes these features as parameters, so a definition is a JSON file the
//! UI can list and swap without compiling anything: built-in definitions
//! ship with the app and users add their own as `.json` files.

use serde::{Deserialize, Serialize};

/// Largest shader definition file read.
pub const MAX_SHADER_FILE_BYTES: usize = 64 * 1024;

/// How the viewport shades a texture set's baked channels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShaderDefinition {
    pub name: String,
    pub description: String,
    /// Specular reflectance of dielectrics, `0..=1` (0.5 is 4%).
    pub specular_level: f32,
    /// Strength of a clear varnish layer over the material, `0..=1`.
    pub clear_coat: f32,
    /// Roughness of that layer, `0..=1`.
    pub clear_coat_roughness: f32,
    /// Linear colour of cloth-like sheen at grazing angles (black: none).
    pub sheen_color: [f32; 3],
    /// Roughness of the sheen, `0..=1`.
    pub sheen_roughness: f32,
    /// Multiplier of the Emissive channel's glow, `0..=16`.
    pub emissive_scale: f32,
    /// Shows Base Color flat, without lighting.
    pub unlit: bool,
}

impl Default for ShaderDefinition {
    fn default() -> Self {
        Self {
            name: "Metal/Rough PBR".to_owned(),
            description: "Metallic/roughness physically based shading.".to_owned(),
            specular_level: 0.5,
            clear_coat: 0.0,
            clear_coat_roughness: 0.1,
            sheen_color: [0.0; 3],
            sheen_roughness: 0.5,
            emissive_scale: 1.0,
            unlit: false,
        }
    }
}

impl ShaderDefinition {
    /// Definitions that ship with the app; the first is the default.
    pub fn built_in() -> Vec<Self> {
        vec![
            Self::default(),
            Self {
                name: "Clear coat".to_owned(),
                description: "A glossy varnish over the material, as on car paint or \
                              lacquered wood."
                    .to_owned(),
                clear_coat: 1.0,
                clear_coat_roughness: 0.05,
                ..Self::default()
            },
            Self {
                name: "Cloth".to_owned(),
                description: "Soft sheen at grazing angles, as on velvet and fabric.".to_owned(),
                specular_level: 0.3,
                sheen_color: [0.6, 0.6, 0.6],
                sheen_roughness: 0.6,
                ..Self::default()
            },
            Self {
                name: "Unlit".to_owned(),
                description: "Base Color as painted, without lighting: for hand-painted \
                              and stylised assets."
                    .to_owned(),
                unlit: true,
                ..Self::default()
            },
        ]
    }

    pub fn is_valid(&self) -> bool {
        let unit = |value: f32| (0.0..=1.0).contains(&value);
        !self.name.trim().is_empty()
            && self.name.chars().count() <= 64
            && self.description.chars().count() <= 512
            && unit(self.specular_level)
            && unit(self.clear_coat)
            && unit(self.clear_coat_roughness)
            && self.sheen_color.iter().all(|channel| unit(*channel))
            && unit(self.sheen_roughness)
            && (0.0..=16.0).contains(&self.emissive_scale)
    }

    /// The renderer's shader parameters in order: specular level, clear
    /// coat, clear-coat roughness, sheen colour (3), sheen roughness,
    /// emissive scale, unlit (0 or 1).
    pub fn parameters(&self) -> [f32; 9] {
        [
            self.specular_level,
            self.clear_coat,
            self.clear_coat_roughness,
            self.sheen_color[0],
            self.sheen_color[1],
            self.sheen_color[2],
            self.sheen_roughness,
            self.emissive_scale,
            if self.unlit { 1.0 } else { 0.0 },
        ]
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    impl ShaderDefinition {
        fn parse(text: &str) -> Result<Self, String> {
            if text.len() > MAX_SHADER_FILE_BYTES {
                return Err("too large".to_owned());
            }
            let shader: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
            shader
                .is_valid()
                .then_some(shader)
                .ok_or_else(|| "out of range".to_owned())
        }
    }

    #[test]
    fn tc_0383_shader_definitions_parse_validate_and_ship_built_ins() {
        let built_in = ShaderDefinition::built_in();
        assert_eq!(built_in.len(), 4);
        assert!(built_in.iter().all(ShaderDefinition::is_valid));
        assert_eq!(built_in[0], ShaderDefinition::default());
        assert_eq!(ShaderDefinition::default().parameters()[0], 0.5);
        assert_eq!(built_in[3].parameters()[8], 1.0, "unlit flag");

        // A user file names only what it changes.
        let shader = ShaderDefinition::parse(
            r#"{ "name": "Wet", "clear_coat": 0.8, "specular_level": 0.6 }"#,
        )
        .unwrap();
        assert_eq!(shader.name, "Wet");
        assert_eq!(shader.clear_coat, 0.8);
        assert_eq!(shader.emissive_scale, 1.0);
        // Round trip.
        let json = serde_json::to_string(&shader).unwrap();
        assert_eq!(ShaderDefinition::parse(&json).unwrap(), shader);

        for bad in [
            r#"{ "name": "" }"#,
            r#"{ "name": "Hot", "emissive_scale": 50 }"#,
            r#"{ "name": "Odd", "sheen_color": [2, 0, 0] }"#,
            r#"{ "name": "Coat", "clear_coat": -1 }"#,
            "not json",
        ] {
            assert!(ShaderDefinition::parse(bad).is_err(), "{bad}");
        }
        assert!(ShaderDefinition::parse(&" ".repeat(MAX_SHADER_FILE_BYTES + 1)).is_err());
    }
}
