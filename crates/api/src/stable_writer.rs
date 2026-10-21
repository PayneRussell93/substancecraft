//! A serde serializer that writes the stable project layout of
//! [`crate::project_format`] directly (PERF-05): sorted keys, two-space
//! indentation, one array element per line unless every element is a
//! scalar. It produces the same bytes as serializing to a
//! [`serde_json::Value`] and laying that out, without building the tree,
//! which for large projects costs several times the file's size in memory.
//!
//! Values are written straight into one buffer. An array starts inline and
//! is laid out again, once, if a non-scalar element turns up; an object's
//! fields are written in the order they come and reordered by key when it
//! ends.

use std::borrow::Cow;

use serde::Serialize;
use serde::ser::{self, Impossible};

use crate::{MAX_PROJECT_JSON_BYTES, ProjectFileError};

/// Writes `value` in the stable layout, with a trailing newline.
pub(crate) fn to_stable_json(value: &impl Serialize) -> Result<Vec<u8>, ProjectFileError> {
    let mut out = Vec::new();
    value
        .serialize(Writer {
            depth: 0,
            out: &mut out,
        })
        .map_err(|error| match error {
            Error::TooLarge => ProjectFileError::TooLarge(MAX_PROJECT_JSON_BYTES),
            Error::Message(message) => ProjectFileError::Encode(message),
        })?;
    out.push(b'\n');
    if out.len() > MAX_PROJECT_JSON_BYTES {
        return Err(ProjectFileError::TooLarge(MAX_PROJECT_JSON_BYTES));
    }
    Ok(out)
}

#[derive(Debug)]
enum Error {
    TooLarge,
    Message(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge => formatter.write_str("project JSON exceeds size limit"),
            Self::Message(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

impl ser::Error for Error {
    fn custom<T: std::fmt::Display>(message: T) -> Self {
        Self::Message(message.to_string())
    }
}

/// True when an f32's serde_json text is the same as that of the f64 the
/// text parses to (checked over sampled bit patterns by the format tests):
/// zero, and magnitudes where both write plain decimals.
pub(crate) fn writes_f32_directly(value: f32) -> bool {
    let magnitude = value.abs();
    magnitude == 0.0 || (1e-4..1e12).contains(&magnitude)
}

/// Writes `value` as serde_json does.
fn json<T: Serialize + ?Sized>(out: &mut Vec<u8>, value: &T) -> Result<(), Error> {
    serde_json::to_writer(out, value).map_err(|error| Error::Message(error.to_string()))
}

fn indent(depth: usize, out: &mut Vec<u8>) {
    out.extend(std::iter::repeat_n(b' ', depth * 2));
}

fn checked(out: &[u8]) -> Result<(), Error> {
    if out.len() > MAX_PROJECT_JSON_BYTES {
        return Err(Error::TooLarge);
    }
    Ok(())
}

/// Writes one value at `depth`; returns whether it was a scalar (which
/// includes empty arrays and objects).
struct Writer<'a> {
    depth: usize,
    out: &'a mut Vec<u8>,
}

impl<'a> Writer<'a> {
    fn scalar<T: Serialize + ?Sized>(self, value: &T) -> Result<bool, Error> {
        json(self.out, value)?;
        Ok(true)
    }

    /// Opens `{"variant": ` around a value one level deeper.
    fn open_variant(self, variant: &str) -> Result<(Writer<'a>, Option<usize>), Error> {
        self.out.extend_from_slice(b"{\n");
        indent(self.depth + 1, self.out);
        json(self.out, variant)?;
        self.out.extend_from_slice(b": ");
        Ok((
            Writer {
                depth: self.depth + 1,
                out: self.out,
            },
            Some(self.depth),
        ))
    }

    fn array(self, variant: Option<usize>) -> Elements<'a> {
        let start = self.out.len();
        self.out.push(b'[');
        Elements {
            depth: self.depth,
            out: self.out,
            start,
            scalars: Vec::new(),
            inline: true,
            count: 0,
            variant,
        }
    }

    fn object(self, variant: Option<usize>) -> Entries<'a> {
        let start = self.out.len();
        Entries {
            depth: self.depth,
            out: self.out,
            start,
            fields: Vec::new(),
            key: None,
            variant,
        }
    }
}

/// Closes a `{"variant": ...}` opened at `depth`.
fn close_variant(out: &mut Vec<u8>, depth: Option<usize>) {
    if let Some(depth) = depth {
        out.push(b'\n');
        indent(depth, out);
        out.push(b'}');
    }
}

macro_rules! scalars {
    ($($method:ident: $type:ty),* $(,)?) => {
        $(fn $method(self, value: $type) -> Result<bool, Error> {
            self.scalar(&value)
        })*
    };
}

impl<'a> ser::Serializer for Writer<'a> {
    type Ok = bool;
    type Error = Error;
    type SerializeSeq = Elements<'a>;
    type SerializeTuple = Elements<'a>;
    type SerializeTupleStruct = Elements<'a>;
    type SerializeTupleVariant = Elements<'a>;
    type SerializeMap = Entries<'a>;
    type SerializeStruct = Entries<'a>;
    type SerializeStructVariant = Entries<'a>;

    scalars! {
        serialize_bool: bool, serialize_i8: i8, serialize_i16: i16, serialize_i32: i32,
        serialize_i64: i64, serialize_i128: i128, serialize_u8: u8, serialize_u16: u16,
        serialize_u32: u32, serialize_u64: u64, serialize_u128: u128, serialize_char: char,
        serialize_f64: f64,
    }

    // A tree held every number as the f64 its text parsed to. Inside the
    // direct band an f32's own text is that f64's text; outside it the two
    // pick different notations, so the f64 is written.
    fn serialize_f32(self, value: f32) -> Result<bool, Error> {
        if writes_f32_directly(value) {
            return self.scalar(&value);
        }
        let mut text = [0_u8; 48];
        let length = {
            let mut cursor = &mut text[..];
            serde_json::to_writer(&mut cursor, &value)
                .map_err(|error| Error::Message(error.to_string()))?;
            48 - cursor.len()
        };
        let parsed = std::str::from_utf8(&text[..length])
            .ok()
            .and_then(|text| text.parse::<f64>().ok());
        match parsed {
            Some(parsed) => self.scalar(&parsed),
            // `null` for a non-finite value.
            None => {
                self.out.extend_from_slice(&text[..length]);
                Ok(true)
            }
        }
    }

    fn serialize_str(self, value: &str) -> Result<bool, Error> {
        self.scalar(value)
    }

    fn serialize_bytes(self, value: &[u8]) -> Result<bool, Error> {
        use ser::SerializeSeq;
        let mut elements = self.array(None);
        for byte in value {
            elements.serialize_element(byte)?;
        }
        elements.end()
    }

    fn serialize_none(self) -> Result<bool, Error> {
        self.out.extend_from_slice(b"null");
        Ok(true)
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<bool, Error> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<bool, Error> {
        self.serialize_none()
    }

    fn serialize_unit_struct(self, _: &'static str) -> Result<bool, Error> {
        self.serialize_none()
    }

    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<bool, Error> {
        self.scalar(variant)
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<bool, Error> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<bool, Error> {
        let depth = self.depth;
        let (inner, _) = self.open_variant(variant)?;
        let out = inner.out;
        value.serialize(Writer {
            depth: depth + 1,
            out: &mut *out,
        })?;
        close_variant(out, Some(depth));
        Ok(false)
    }

    fn serialize_seq(self, _: Option<usize>) -> Result<Elements<'a>, Error> {
        Ok(self.array(None))
    }

    fn serialize_tuple(self, _: usize) -> Result<Elements<'a>, Error> {
        Ok(self.array(None))
    }

    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Elements<'a>, Error> {
        Ok(self.array(None))
    }

    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Elements<'a>, Error> {
        let (inner, depth) = self.open_variant(variant)?;
        Ok(inner.array(depth))
    }

    fn serialize_map(self, _: Option<usize>) -> Result<Entries<'a>, Error> {
        Ok(self.object(None))
    }

    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Entries<'a>, Error> {
        Ok(self.object(None))
    }

    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Entries<'a>, Error> {
        let (inner, depth) = self.open_variant(variant)?;
        Ok(inner.object(depth))
    }
}

/// An array being written: inline (`[a, b]`) while every element so far
/// is a scalar, one element per line once one is not.
struct Elements<'a> {
    depth: usize,
    out: &'a mut Vec<u8>,
    /// Where the `[` is.
    start: usize,
    /// While inline: where each element's bytes lie in `out`.
    scalars: Vec<(usize, usize)>,
    inline: bool,
    count: usize,
    /// For a tuple variant: the depth of the wrapping object.
    variant: Option<usize>,
}

impl Elements<'_> {
    fn push<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        if self.inline {
            if self.count > 0 {
                self.out.extend_from_slice(b", ");
            }
            let begin = self.out.len();
            let scalar = value.serialize(Writer {
                depth: self.depth + 1,
                out: &mut *self.out,
            })?;
            let end = self.out.len();
            if scalar {
                self.scalars.push((begin, end));
            } else {
                self.go_multiline(begin, end);
            }
        } else {
            self.out.extend_from_slice(b",\n");
            indent(self.depth + 1, self.out);
            value.serialize(Writer {
                depth: self.depth + 1,
                out: &mut *self.out,
            })?;
        }
        self.count += 1;
        checked(self.out)
    }

    /// Lays the elements written so far out one per line; the last one,
    /// at `begin..end`, is the first non-scalar.
    fn go_multiline(&mut self, begin: usize, end: usize) {
        let region = self.out.split_off(self.start);
        let offset = |at: usize| at - self.start;
        self.out.extend_from_slice(b"[\n");
        for (scalar_begin, scalar_end) in &self.scalars {
            indent(self.depth + 1, self.out);
            self.out
                .extend_from_slice(&region[offset(*scalar_begin)..offset(*scalar_end)]);
            self.out.extend_from_slice(b",\n");
        }
        indent(self.depth + 1, self.out);
        self.out
            .extend_from_slice(&region[offset(begin)..offset(end)]);
        self.scalars = Vec::new();
        self.inline = false;
    }

    fn finish(self) -> Result<bool, Error> {
        if self.inline {
            self.out.push(b']');
        } else {
            self.out.push(b'\n');
            indent(self.depth, self.out);
            self.out.push(b']');
        }
        close_variant(self.out, self.variant);
        checked(self.out)?;
        Ok(self.count == 0 && self.variant.is_none())
    }
}

impl ser::SerializeSeq for Elements<'_> {
    type Ok = bool;
    type Error = Error;

    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }

    fn end(self) -> Result<bool, Error> {
        self.finish()
    }
}

impl ser::SerializeTuple for Elements<'_> {
    type Ok = bool;
    type Error = Error;

    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }

    fn end(self) -> Result<bool, Error> {
        self.finish()
    }
}

impl ser::SerializeTupleStruct for Elements<'_> {
    type Ok = bool;
    type Error = Error;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }

    fn end(self) -> Result<bool, Error> {
        self.finish()
    }
}

impl ser::SerializeTupleVariant for Elements<'_> {
    type Ok = bool;
    type Error = Error;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }

    fn end(self) -> Result<bool, Error> {
        self.finish()
    }
}

/// An object being written: fields go out in the order they come and are
/// reordered by key at the end.
struct Entries<'a> {
    depth: usize,
    out: &'a mut Vec<u8>,
    start: usize,
    /// Each field's key and where its value lies in `out`.
    fields: Vec<(Cow<'static, str>, usize, usize)>,
    /// A map key waiting for its value.
    key: Option<String>,
    /// For a struct variant: the depth of the wrapping object.
    variant: Option<usize>,
}

impl Entries<'_> {
    fn push<T: Serialize + ?Sized>(
        &mut self,
        key: Cow<'static, str>,
        value: &T,
    ) -> Result<(), Error> {
        let begin = self.out.len();
        value.serialize(Writer {
            depth: self.depth + 1,
            out: &mut *self.out,
        })?;
        self.fields.push((key, begin, self.out.len()));
        checked(self.out)
    }

    fn finish(mut self) -> Result<bool, Error> {
        let region = self.out.split_off(self.start);
        let empty = self.fields.is_empty();
        if empty {
            self.out.extend_from_slice(b"{}");
        } else {
            // Sorted as `serde_json::Map` keeps keys; of a repeated key
            // the last value stays, as a map insert would leave it.
            self.fields.sort_by(|a, b| a.0.cmp(&b.0));
            self.fields.reverse();
            self.fields.dedup_by(|later, earlier| later.0 == earlier.0);
            self.fields.reverse();
            self.out.extend_from_slice(b"{\n");
            let last = self.fields.len() - 1;
            for (index, (key, begin, end)) in self.fields.iter().enumerate() {
                indent(self.depth + 1, self.out);
                json(self.out, key.as_ref())?;
                self.out.extend_from_slice(b": ");
                self.out
                    .extend_from_slice(&region[begin - self.start..end - self.start]);
                if index < last {
                    self.out.push(b',');
                }
                self.out.push(b'\n');
            }
            indent(self.depth, self.out);
            self.out.push(b'}');
        }
        close_variant(self.out, self.variant);
        checked(self.out)?;
        Ok(empty && self.variant.is_none())
    }
}

impl ser::SerializeMap for Entries<'_> {
    type Ok = bool;
    type Error = Error;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), Error> {
        self.key = Some(key.serialize(KeyText)?);
        Ok(())
    }

    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        let key = self
            .key
            .take()
            .ok_or_else(|| Error::Message("map value without a key".to_owned()))?;
        self.push(Cow::Owned(key), value)
    }

    fn end(self) -> Result<bool, Error> {
        self.finish()
    }
}

impl ser::SerializeStruct for Entries<'_> {
    type Ok = bool;
    type Error = Error;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.push(Cow::Borrowed(key), value)
    }

    fn end(self) -> Result<bool, Error> {
        self.finish()
    }
}

impl ser::SerializeStructVariant for Entries<'_> {
    type Ok = bool;
    type Error = Error;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.push(Cow::Borrowed(key), value)
    }

    fn end(self) -> Result<bool, Error> {
        self.finish()
    }
}

/// A map key as the text serde_json gives it: strings as they are,
/// numbers, booleans, and characters as their text, unit variants by name.
struct KeyText;

fn key_error() -> Error {
    Error::Message("key must be a string".to_owned())
}

macro_rules! key_numbers {
    ($($method:ident: $type:ty),* $(,)?) => {
        $(fn $method(self, value: $type) -> Result<String, Error> {
            Ok(value.to_string())
        })*
    };
}

impl ser::Serializer for KeyText {
    type Ok = String;
    type Error = Error;
    type SerializeSeq = Impossible<String, Error>;
    type SerializeTuple = Impossible<String, Error>;
    type SerializeTupleStruct = Impossible<String, Error>;
    type SerializeTupleVariant = Impossible<String, Error>;
    type SerializeMap = Impossible<String, Error>;
    type SerializeStruct = Impossible<String, Error>;
    type SerializeStructVariant = Impossible<String, Error>;

    key_numbers! {
        serialize_bool: bool, serialize_i8: i8, serialize_i16: i16, serialize_i32: i32,
        serialize_i64: i64, serialize_i128: i128, serialize_u8: u8, serialize_u16: u16,
        serialize_u32: u32, serialize_u64: u64, serialize_u128: u128, serialize_char: char,
    }

    fn serialize_f32(self, _: f32) -> Result<String, Error> {
        Err(key_error())
    }

    fn serialize_f64(self, _: f64) -> Result<String, Error> {
        Err(key_error())
    }

    fn serialize_str(self, value: &str) -> Result<String, Error> {
        Ok(value.to_owned())
    }

    fn serialize_bytes(self, _: &[u8]) -> Result<String, Error> {
        Err(key_error())
    }

    fn serialize_none(self) -> Result<String, Error> {
        Err(key_error())
    }

    fn serialize_some<T: Serialize + ?Sized>(self, _: &T) -> Result<String, Error> {
        Err(key_error())
    }

    fn serialize_unit(self) -> Result<String, Error> {
        Err(key_error())
    }

    fn serialize_unit_struct(self, _: &'static str) -> Result<String, Error> {
        Err(key_error())
    }

    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<String, Error> {
        Ok(variant.to_owned())
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<String, Error> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: &T,
    ) -> Result<String, Error> {
        Err(key_error())
    }

    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, Error> {
        Err(key_error())
    }

    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, Error> {
        Err(key_error())
    }

    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleStruct, Error> {
        Err(key_error())
    }

    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant, Error> {
        Err(key_error())
    }

    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, Error> {
        Err(key_error())
    }

    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self::SerializeStruct, Error> {
        Err(key_error())
    }

    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant, Error> {
        Err(key_error())
    }
}
