//! ASCII FBX 7.x reader.
//!
//! ASCII files hold the same node tree as binary ones, but without value
//! types and with object names written `Class::Name`. The tree is parsed,
//! typed the way binary FBX stores each value (object ids as 64-bit
//! integers, index arrays as 32-bit ones, `P` values by their declared
//! kind), and re-encoded as binary FBX 7.4, so both forms share one mesh
//! importer.

use std::io::Cursor;

use fbxcel::{low::FbxVersion, writer::v7400::binary::Writer};

use crate::fbx::{FbxImportError, MAX_DECODED_ATTRIBUTES, MAX_FBX_DEPTH, MAX_FBX_NODES};

/// A number as written; ASCII FBX does not say which type it is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Number {
    Int(i64),
    Float(f64),
}

impl Number {
    fn as_f64(self) -> f64 {
        match self {
            Self::Int(value) => value as f64,
            Self::Float(value) => value,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Number(Number),
    String(String),
    /// An unquoted word, such as the `T` or `Y` of a boolean flag.
    Word(String),
    Array(Vec<Number>),
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Node {
    name: String,
    values: Vec<Value>,
    children: Vec<Node>,
}

/// Arrays binary FBX stores as 32-bit integers.
const I32_ARRAYS: &[&str] = &[
    "PolygonVertexIndex",
    "Edges",
    "UVIndex",
    "NormalsIndex",
    "BinormalsIndex",
    "TangentsIndex",
    "ColorIndex",
    "Materials",
    "Smoothing",
    "Indexes",
    "TextureId",
    "PolygonGroup",
    "KeyAttrFlags",
    "KeyAttrRefCount",
];

/// Arrays binary FBX stores as 32-bit floats.
const F32_ARRAYS: &[&str] = &["KeyValueFloat", "KeyAttrDataFloat"];

/// `P` kinds whose numbers are 32-bit integers.
const I32_PROPERTY_KINDS: &[&str] = &[
    "int",
    "Integer",
    "enum",
    "Enum",
    "bool",
    "Bool",
    "Visibility Inheritance",
];

/// `P` kinds whose numbers are 64-bit integers.
const I64_PROPERTY_KINDS: &[&str] = &["KTime", "ULongLong"];

/// True when `bytes` look like ASCII rather than binary FBX.
pub(crate) fn is_ascii_fbx(bytes: &[u8]) -> bool {
    !bytes.starts_with(b"Kaydara FBX Binary")
}

/// Parses ASCII FBX and re-encodes it as binary FBX 7.4.
pub(crate) fn ascii_to_binary(bytes: &[u8]) -> Result<Vec<u8>, FbxImportError> {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    let text = std::str::from_utf8(bytes)
        .map_err(|_| FbxImportError::Parse("ASCII FBX is not valid UTF-8".to_owned()))?;
    let nodes = Parser::new(text).parse()?;
    check_version(text, &nodes)?;
    encode(&nodes)
}

/// Refuses files older than 7.4, by the header's `FBXVersion` or, when it
/// has none, the `; FBX x.y.z project file` comment.
fn check_version(text: &str, nodes: &[Node]) -> Result<(), FbxImportError> {
    let declared = nodes
        .iter()
        .find(|node| node.name == "FBXHeaderExtension")
        .and_then(|header| header.children.iter().find(|n| n.name == "FBXVersion"))
        .and_then(|version| match version.values.first() {
            Some(Value::Number(Number::Int(version))) => Some(*version),
            _ => None,
        })
        .or_else(|| {
            let comment = text.lines().next()?.trim().strip_prefix(';')?.trim();
            let version = comment.strip_prefix("FBX ")?.split_whitespace().next()?;
            let mut parts = version.split('.').map(str::parse::<i64>);
            let major = parts.next()?.ok()?;
            let minor = parts.next().and_then(Result::ok).unwrap_or(0);
            Some(major * 1000 + minor * 100)
        });
    match declared {
        Some(version) if version < 7_400 => Err(FbxImportError::UnsupportedVersion),
        _ => Ok(()),
    }
}

struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
    node_count: usize,
    decoded_bytes: u64,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            bytes: text.as_bytes(),
            pos: 0,
            node_count: 0,
            decoded_bytes: 0,
        }
    }

    fn parse(mut self) -> Result<Vec<Node>, FbxImportError> {
        let nodes = self.sequence(0, false)?;
        if nodes.is_empty() {
            return Err(self.error("no FBX nodes"));
        }
        Ok(nodes)
    }

    fn error(&self, message: &str) -> FbxImportError {
        let line = self.text[..self.pos.min(self.text.len())]
            .bytes()
            .filter(|byte| *byte == b'\n')
            .count()
            + 1;
        FbxImportError::Parse(format!("ASCII FBX line {line}: {message}"))
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// Skips spaces and `;` comments, and line breaks too when `lines`.
    fn skip_space(&mut self, lines: bool) {
        while let Some(byte) = self.peek() {
            match byte {
                b' ' | b'\t' | b'\r' => self.pos += 1,
                b'\n' if lines => self.pos += 1,
                b';' => {
                    while self.peek().is_some_and(|byte| byte != b'\n') {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    /// Nodes up to the end of input, or up to and including a `}`.
    fn sequence(&mut self, depth: usize, closed: bool) -> Result<Vec<Node>, FbxImportError> {
        if depth > MAX_FBX_DEPTH {
            return Err(FbxImportError::InvalidBinary(
                "node nesting limit exceeded".to_owned(),
            ));
        }
        let mut nodes = Vec::new();
        loop {
            self.skip_space(true);
            match self.peek() {
                None if closed => return Err(self.error("missing `}`")),
                None => return Ok(nodes),
                Some(b'}') if closed => {
                    self.pos += 1;
                    return Ok(nodes);
                }
                Some(b'}') => return Err(self.error("unexpected `}`")),
                Some(_) => nodes.push(self.node(depth)?),
            }
        }
    }

    fn node(&mut self, depth: usize) -> Result<Node, FbxImportError> {
        self.node_count += 1;
        if self.node_count > MAX_FBX_NODES {
            return Err(FbxImportError::InvalidBinary(
                "node count limit exceeded".to_owned(),
            ));
        }
        let name = self.token();
        if name.is_empty() || !name.bytes().all(is_name_byte) {
            return Err(self.error("expected a node name"));
        }
        if self.peek() != Some(b':') {
            return Err(self.error("expected `:` after a node name"));
        }
        self.pos += 1;
        let mut node = Node {
            name: name.to_owned(),
            ..Node::default()
        };
        loop {
            self.skip_space(false);
            match self.peek() {
                None | Some(b'\n' | b'}') => return Ok(node),
                Some(b'{') => {
                    self.pos += 1;
                    node.children = self.sequence(depth + 1, true)?;
                    return Ok(node);
                }
                // A comma continues the values, across line breaks too; a
                // leading one (as in `Content: ,`) stands for nothing.
                Some(b',') => {
                    self.pos += 1;
                    self.skip_space(true);
                }
                Some(b'*') => {
                    self.pos += 1;
                    let array = self.array()?;
                    node.values.push(Value::Array(array));
                }
                Some(b'"') => {
                    let string = self.string()?;
                    node.values.push(Value::String(string));
                }
                Some(_) => {
                    let token = self.token();
                    if token.is_empty() {
                        return Err(self.error("unexpected character"));
                    }
                    node.values.push(match number(token) {
                        Some(number) => Value::Number(number),
                        None => Value::Word(token.to_owned()),
                    });
                }
            }
        }
    }

    /// An unquoted run of characters: a name, number, or word.
    fn token(&mut self) -> &'a str {
        let start = self.pos;
        while self.peek().is_some_and(|byte| {
            !matches!(
                byte,
                b' ' | b'\t' | b'\r' | b'\n' | b',' | b':' | b'{' | b'}' | b';' | b'"' | b'*'
            )
        }) {
            self.pos += 1;
        }
        &self.text[start..self.pos]
    }

    fn string(&mut self) -> Result<String, FbxImportError> {
        self.pos += 1;
        let start = self.pos;
        while let Some(byte) = self.peek() {
            if byte == b'"' {
                let string = self.text[start..self.pos].replace("&quot;", "\"");
                self.pos += 1;
                return Ok(string);
            }
            self.pos += 1;
        }
        Err(self.error("unterminated string"))
    }

    /// `*count { a: n, n, … }`, after the `*`.
    fn array(&mut self) -> Result<Vec<Number>, FbxImportError> {
        let count = self
            .token()
            .parse::<u64>()
            .map_err(|_| self.error("expected an array length after `*`"))?;
        self.charge(count.saturating_mul(8))?;
        self.skip_space(true);
        if self.peek() != Some(b'{') {
            return Err(self.error("expected `{` to open an array"));
        }
        self.pos += 1;
        self.skip_space(true);
        let mut values = Vec::new();
        if self.peek() != Some(b'}') {
            if self.token() != "a" || self.peek() != Some(b':') {
                return Err(self.error("expected `a:` in an array"));
            }
            self.pos += 1;
            loop {
                self.skip_space(true);
                match self.peek() {
                    Some(b'}') => break,
                    Some(b',') => self.pos += 1,
                    None => return Err(self.error("missing `}` after an array")),
                    Some(_) => {
                        let token = self.token();
                        let value = number(token)
                            .ok_or_else(|| self.error("expected a number in an array"))?;
                        values.push(value);
                    }
                }
            }
        }
        self.pos += 1;
        if values.len() as u64 != count {
            return Err(self.error(&format!(
                "array declares {count} values but holds {}",
                values.len()
            )));
        }
        Ok(values)
    }

    fn charge(&mut self, bytes: u64) -> Result<(), FbxImportError> {
        self.decoded_bytes = self.decoded_bytes.saturating_add(bytes);
        if self.decoded_bytes > MAX_DECODED_ATTRIBUTES {
            return Err(FbxImportError::ResourceBudgetExceeded(self.decoded_bytes));
        }
        Ok(())
    }
}

fn is_name_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'|' | b'-' | b'.')
}

fn number(token: &str) -> Option<Number> {
    let first = *token.as_bytes().first()?;
    if !(first.is_ascii_digit() || matches!(first, b'-' | b'+' | b'.')) {
        return None;
    }
    if !token.contains(['.', 'e', 'E'])
        && let Ok(value) = token.parse::<i64>()
    {
        return Some(Number::Int(value));
    }
    token.parse::<f64>().ok().map(Number::Float)
}

/// How a node's numbers are stored in binary FBX.
#[derive(Clone, Copy, PartialEq)]
enum Scalars {
    /// Integers as 32-bit (64-bit when they do not fit), others as f64.
    Natural,
    /// Every integer as 64-bit.
    Wide,
    /// Every number as f64.
    Float,
}

fn encode(nodes: &[Node]) -> Result<Vec<u8>, FbxImportError> {
    let write_error = |error: fbxcel::writer::v7400::binary::Error| {
        FbxImportError::Parse(format!("ASCII FBX could not be converted: {error}"))
    };
    let mut writer = Writer::new(Cursor::new(Vec::new()), FbxVersion::V7_4).map_err(write_error)?;
    for node in nodes {
        write_node(&mut writer, node, None).map_err(write_error)?;
    }
    Ok(writer
        .finalize_and_flush(&Default::default())
        .map_err(write_error)?
        .into_inner())
}

fn write_node(
    writer: &mut Writer<Cursor<Vec<u8>>>,
    node: &Node,
    parent: Option<&str>,
) -> fbxcel::writer::v7400::binary::Result<()> {
    let object = parent == Some("Objects");
    let scalars = match node.name.as_str() {
        "C" | "Connect" | "Document" | "RootNode" => Scalars::Wide,
        "P" => match node.values.get(1) {
            Some(Value::String(kind)) if I64_PROPERTY_KINDS.contains(&kind.as_str()) => {
                Scalars::Wide
            }
            Some(Value::String(kind)) if I32_PROPERTY_KINDS.contains(&kind.as_str()) => {
                Scalars::Natural
            }
            _ => Scalars::Float,
        },
        _ => Scalars::Natural,
    };
    let mut attributes = writer.new_node(&node.name)?;
    for (index, value) in node.values.iter().enumerate() {
        match value {
            // Object ids are 64-bit.
            Value::Number(Number::Int(id)) if object && index == 0 => attributes.append_i64(*id)?,
            Value::Number(number) => match (scalars, number) {
                (Scalars::Float, number) => attributes.append_f64(number.as_f64())?,
                (Scalars::Wide, Number::Int(value)) => attributes.append_i64(*value)?,
                (_, Number::Int(value)) => match i32::try_from(*value) {
                    Ok(value) => attributes.append_i32(value)?,
                    Err(_) => attributes.append_i64(*value)?,
                },
                (_, Number::Float(value)) => attributes.append_f64(*value)?,
            },
            // `Class::Name` object names become binary `Name\0\x01Class`.
            Value::String(string) if object && index == 1 => match string.split_once("::") {
                Some((class, name)) => {
                    attributes.append_string_direct(&format!("{name}\0\u{1}{class}"))?;
                }
                None => attributes.append_string_direct(string)?,
            },
            Value::String(string) => attributes.append_string_direct(string)?,
            Value::Word(word) => match word.as_str() {
                "T" | "Y" => attributes.append_bool(true)?,
                "F" | "N" => attributes.append_bool(false)?,
                _ => attributes.append_string_direct(word)?,
            },
            Value::Array(values) => {
                let name = node.name.as_str();
                if name == "KeyTime" {
                    attributes.append_arr_i64_from_iter(
                        None,
                        values.iter().map(|value| match value {
                            Number::Int(value) => *value,
                            Number::Float(value) => *value as i64,
                        }),
                    )?;
                } else if I32_ARRAYS.contains(&name) {
                    attributes.append_arr_i32_from_iter(
                        None,
                        values.iter().map(|value| match value {
                            Number::Int(value) => *value as i32,
                            Number::Float(value) => *value as i32,
                        }),
                    )?;
                } else if F32_ARRAYS.contains(&name) {
                    attributes.append_arr_f32_from_iter(
                        None,
                        values.iter().map(|value| value.as_f64() as f32),
                    )?;
                } else {
                    attributes.append_arr_f64_from_iter(
                        None,
                        values.iter().map(|value| value.as_f64()),
                    )?;
                }
            }
        }
    }
    for child in &node.children {
        write_node(writer, child, Some(&node.name))?;
    }
    writer.close_node()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_arrays_strings_words_comments_and_continued_values() {
        let nodes = Parser::new(
            "; FBX 7.4.0 project file\n\
             Objects:  {\n\
             \tGeometry: 7, \"Geometry::Box\", \"Mesh\" {\n\
             \t\tVertices: *3 {\n\t\t\ta: 0,1.5,\n-2e1\n\t\t} \n\
             \t\tShading: T ; trailing comment\n\
             \t\tContent: ,\n\t\t\t\"QUJD\"\n\
             \t\tName: \"say &quot;hi&quot;\"\n\
             \t\tEmpty: *0 {\n\t\t}\n\
             \t}\n\
             }\n",
        )
        .parse()
        .unwrap();
        let geometry = &nodes[0].children[0];
        assert_eq!(geometry.name, "Geometry");
        assert_eq!(
            geometry.values,
            [
                Value::Number(Number::Int(7)),
                Value::String("Geometry::Box".to_owned()),
                Value::String("Mesh".to_owned()),
            ]
        );
        let child = |name: &str| {
            &geometry
                .children
                .iter()
                .find(|node| node.name == name)
                .unwrap()
                .values
        };
        assert_eq!(
            child("Vertices"),
            &[Value::Array(vec![
                Number::Int(0),
                Number::Float(1.5),
                Number::Float(-20.0)
            ])]
        );
        assert_eq!(child("Shading"), &[Value::Word("T".to_owned())]);
        assert_eq!(child("Content"), &[Value::String("QUJD".to_owned())]);
        assert_eq!(child("Name"), &[Value::String("say \"hi\"".to_owned())]);
        assert_eq!(child("Empty"), &[Value::Array(Vec::new())]);
    }

    #[test]
    fn malformed_ascii_reports_the_line() {
        let error = Parser::new("Objects:  {\n\tVertices: *4 {\n\t\ta: 1,2,3\n\t}\n}\n")
            .parse()
            .unwrap_err();
        assert!(
            matches!(&error, FbxImportError::Parse(message)
                if message.contains("line 4") && message.contains("declares 4 values")),
            "{error}"
        );
        assert!(matches!(
            Parser::new("Objects:  {\n").parse(),
            Err(FbxImportError::Parse(message)) if message.contains("missing `}`")
        ));
        assert!(matches!(
            Parser::new("").parse(),
            Err(FbxImportError::Parse(_))
        ));
    }

    #[test]
    fn versions_before_7_4_are_refused() {
        let old = "; FBX 6.1.0 project file\nObjects:  {\n}\n";
        assert_eq!(
            ascii_to_binary(old.as_bytes()),
            Err(FbxImportError::UnsupportedVersion)
        );
        let header = "FBXHeaderExtension:  {\n\tFBXVersion: 7300\n}\n";
        assert_eq!(
            ascii_to_binary(header.as_bytes()),
            Err(FbxImportError::UnsupportedVersion)
        );
        let current = "; FBX 7.5.0 project file\nObjects:  {\n}\n";
        assert!(ascii_to_binary(current.as_bytes()).is_ok());
    }
}
