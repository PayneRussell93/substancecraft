//! ISO 10303-21 (STEP physical file) reader: every entity instance of the
//! DATA sections, simple or complex, with its parameters. Header entities
//! are skipped; nothing is interpreted beyond the file syntax.

use std::collections::HashMap;

use thiserror::Error;

const MAX_PARAMETER_DEPTH: usize = 64;
pub(crate) const MAX_STEP_INSTANCES: usize = 8_000_000;

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum StepSyntaxError {
    #[error("line {line}: {message}")]
    At { line: usize, message: String },
    #[error("the file is not ISO 10303-21 (STEP) data")]
    NotStep,
    #[error("the file has more than {MAX_STEP_INSTANCES} entity instances")]
    TooManyInstances,
}

/// One parameter value.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Param {
    Integer(i64),
    Real(f64),
    String(String),
    /// An enumeration or logical without its dots, e.g. `T` or `UNSPECIFIED`.
    Enum(String),
    Reference(u64),
    /// `$`
    Unset,
    /// `*`
    Derived,
    List(Vec<Param>),
    /// A typed value such as `LENGTH_MEASURE(1.)`.
    Typed(String, Box<Param>),
    Binary(String),
}

impl Param {
    /// The number this parameter holds, looking through typed values.
    pub(crate) fn number(&self) -> Option<f64> {
        match self {
            Self::Real(value) => Some(*value),
            Self::Integer(value) => Some(*value as f64),
            Self::Typed(_, inner) => inner.number(),
            _ => None,
        }
    }

    pub(crate) fn reference(&self) -> Option<u64> {
        match self {
            Self::Reference(id) => Some(*id),
            Self::Typed(_, inner) => inner.reference(),
            _ => None,
        }
    }

    pub(crate) fn list(&self) -> Option<&[Param]> {
        match self {
            Self::List(items) => Some(items),
            _ => None,
        }
    }

    pub(crate) fn enumeration(&self) -> Option<&str> {
        match self {
            Self::Enum(value) => Some(value),
            _ => None,
        }
    }

    /// `.T.` is true, `.F.` false; anything else (`.U.`, `$`) is `None`.
    pub(crate) fn logical(&self) -> Option<bool> {
        match self.enumeration()? {
            "T" | "TRUE" => Some(true),
            "F" | "FALSE" => Some(false),
            _ => None,
        }
    }

    pub(crate) fn string(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    /// References in a list parameter, skipping anything else.
    pub(crate) fn references(&self) -> Vec<u64> {
        self.list()
            .map(|items| items.iter().filter_map(Param::reference).collect())
            .unwrap_or_default()
    }

    pub(crate) fn numbers(&self) -> Option<Vec<f64>> {
        self.list()?.iter().map(Param::number).collect()
    }
}

/// One `NAME(parameters)` record. Names are upper case.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Record {
    pub name: String,
    pub params: Vec<Param>,
}

impl Record {
    pub(crate) fn param(&self, index: usize) -> Option<&Param> {
        self.params.get(index)
    }

    pub(crate) fn reference(&self, index: usize) -> Option<u64> {
        self.param(index)?.reference()
    }

    pub(crate) fn number(&self, index: usize) -> Option<f64> {
        self.param(index)?.number()
    }

    pub(crate) fn logical(&self, index: usize) -> Option<bool> {
        self.param(index)?.logical()
    }

    pub(crate) fn references(&self, index: usize) -> Vec<u64> {
        self.param(index).map(Param::references).unwrap_or_default()
    }

    pub(crate) fn string(&self, index: usize) -> Option<&str> {
        self.param(index)?.string()
    }
}

/// An entity instance: one record when simple, one per partial entity (in
/// file order) when complex.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Instance {
    pub records: Vec<Record>,
    pub complex: bool,
}

impl Instance {
    /// The record named `name`.
    pub(crate) fn record(&self, name: &str) -> Option<&Record> {
        self.records.iter().find(|record| record.name == name)
    }

    /// The record of a simple instance, if it is named `name`.
    pub(crate) fn simple(&self, name: &str) -> Option<&Record> {
        if self.complex {
            return None;
        }
        self.records.first().filter(|record| record.name == name)
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        self.records.iter().any(|record| record.name == name)
    }

    /// Name of a simple instance.
    pub(crate) fn name(&self) -> Option<&str> {
        if self.complex {
            return None;
        }
        self.records.first().map(|record| record.name.as_str())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct StepFile {
    pub instances: HashMap<u64, Instance>,
}

impl StepFile {
    pub(crate) fn get(&self, id: u64) -> Option<&Instance> {
        self.instances.get(&id)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Keyword(String),
    Instance(u64),
    Integer(i64),
    Real(f64),
    String(String),
    Enum(String),
    Binary(String),
    Open,
    Close,
    Comma,
    Semicolon,
    Equals,
    Dollar,
    Star,
    End,
}

struct Lexer<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Lexer<'a> {
    fn error(&self, message: impl Into<String>) -> StepSyntaxError {
        let end = self.position.min(self.bytes.len());
        let line = 1 + self.bytes[..end]
            .iter()
            .filter(|byte| **byte == b'\n')
            .count();
        StepSyntaxError::At {
            line,
            message: message.into(),
        }
    }

    fn skip_space(&mut self) -> Result<(), StepSyntaxError> {
        loop {
            match self.bytes.get(self.position) {
                Some(byte) if byte.is_ascii_whitespace() => self.position += 1,
                Some(b'/') if self.bytes.get(self.position + 1) == Some(&b'*') => {
                    let start = self.position;
                    self.position += 2;
                    loop {
                        match self.bytes.get(self.position) {
                            None => {
                                self.position = start;
                                return Err(self.error("unterminated comment"));
                            }
                            Some(b'*') if self.bytes.get(self.position + 1) == Some(&b'/') => {
                                self.position += 2;
                                break;
                            }
                            Some(_) => self.position += 1,
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn next(&mut self) -> Result<Token, StepSyntaxError> {
        self.skip_space()?;
        let Some(&byte) = self.bytes.get(self.position) else {
            return Ok(Token::End);
        };
        let single = match byte {
            b'(' => Some(Token::Open),
            b')' => Some(Token::Close),
            b',' => Some(Token::Comma),
            b';' => Some(Token::Semicolon),
            b'=' => Some(Token::Equals),
            b'$' => Some(Token::Dollar),
            b'*' => Some(Token::Star),
            _ => None,
        };
        if let Some(token) = single {
            self.position += 1;
            return Ok(token);
        }
        match byte {
            b'#' => {
                self.position += 1;
                let start = self.position;
                while self
                    .bytes
                    .get(self.position)
                    .is_some_and(u8::is_ascii_digit)
                {
                    self.position += 1;
                }
                let text = std::str::from_utf8(&self.bytes[start..self.position]).unwrap_or("");
                text.parse()
                    .map(Token::Instance)
                    .map_err(|_| self.error("invalid instance name"))
            }
            b'\'' => self.string(),
            b'"' => {
                self.position += 1;
                let start = self.position;
                while self
                    .bytes
                    .get(self.position)
                    .is_some_and(|byte| *byte != b'"')
                {
                    self.position += 1;
                }
                if self.position >= self.bytes.len() {
                    return Err(self.error("unterminated binary value"));
                }
                let text = String::from_utf8_lossy(&self.bytes[start..self.position]).into_owned();
                self.position += 1;
                Ok(Token::Binary(text))
            }
            b'.' if self
                .bytes
                .get(self.position + 1)
                .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_') =>
            {
                self.position += 1;
                let start = self.position;
                while self
                    .bytes
                    .get(self.position)
                    .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
                {
                    self.position += 1;
                }
                let text =
                    String::from_utf8_lossy(&self.bytes[start..self.position]).to_ascii_uppercase();
                if self.bytes.get(self.position) != Some(&b'.') {
                    return Err(self.error("unterminated enumeration"));
                }
                self.position += 1;
                Ok(Token::Enum(text))
            }
            b'+' | b'-' | b'.' | b'0'..=b'9' => self.number(),
            byte if byte.is_ascii_alphabetic() || byte == b'_' || byte == b'!' => {
                let start = self.position;
                self.position += 1;
                while self.bytes.get(self.position).is_some_and(|byte| {
                    byte.is_ascii_alphanumeric() || *byte == b'_' || *byte == b'-'
                }) {
                    self.position += 1;
                }
                Ok(Token::Keyword(
                    String::from_utf8_lossy(&self.bytes[start..self.position]).to_ascii_uppercase(),
                ))
            }
            other => Err(self.error(format!("unexpected character `{}`", other as char))),
        }
    }

    fn number(&mut self) -> Result<Token, StepSyntaxError> {
        let start = self.position;
        if matches!(self.bytes.get(self.position), Some(b'+' | b'-')) {
            self.position += 1;
        }
        let mut real = false;
        while let Some(&byte) = self.bytes.get(self.position) {
            match byte {
                b'0'..=b'9' => self.position += 1,
                b'.' => {
                    real = true;
                    self.position += 1;
                }
                b'E' | b'e' => {
                    real = true;
                    self.position += 1;
                    if matches!(self.bytes.get(self.position), Some(b'+' | b'-')) {
                        self.position += 1;
                    }
                }
                _ => break,
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.position]).unwrap_or("");
        if real {
            // `1.` and `1.E-5` are valid STEP reals; Rust needs a digit
            // after the point.
            let fixed = text.replace(".E", ".0E").replace(".e", ".0e");
            let fixed = if fixed.ends_with('.') {
                format!("{fixed}0")
            } else {
                fixed
            };
            fixed
                .parse()
                .map(Token::Real)
                .map_err(|_| self.error(format!("invalid real `{text}`")))
        } else {
            text.parse()
                .map(Token::Integer)
                .map_err(|_| self.error(format!("invalid integer `{text}`")))
        }
    }

    fn string(&mut self) -> Result<Token, StepSyntaxError> {
        let start = self.position;
        self.position += 1;
        let mut raw = Vec::new();
        loop {
            match self.bytes.get(self.position) {
                None => {
                    self.position = start;
                    return Err(self.error("unterminated string"));
                }
                Some(b'\'') if self.bytes.get(self.position + 1) == Some(&b'\'') => {
                    raw.push(b'\'');
                    self.position += 2;
                }
                Some(b'\'') => {
                    self.position += 1;
                    break;
                }
                Some(&byte) => {
                    raw.push(byte);
                    self.position += 1;
                }
            }
        }
        Ok(Token::String(decode_string(&raw)))
    }
}

/// Decodes the `\X\`, `\X2\`, `\X4\`, `\S\`, and `\\` control directives of
/// a STEP string. Page selections (`\P?\`) are dropped; other bytes are read
/// as UTF-8 when valid, else ISO 8859-1.
fn decode_string(raw: &[u8]) -> String {
    let mut out = String::new();
    let mut plain = Vec::new();
    let flush = |plain: &mut Vec<u8>, out: &mut String| {
        match std::str::from_utf8(plain) {
            Ok(text) => out.push_str(text),
            Err(_) => out.extend(plain.iter().map(|byte| *byte as char)),
        }
        plain.clear();
    };
    let hex = |bytes: &[u8]| -> Option<u32> {
        u32::from_str_radix(std::str::from_utf8(bytes).ok()?, 16).ok()
    };
    let mut index = 0;
    while index < raw.len() {
        let rest = &raw[index..];
        if rest.starts_with(b"\\\\") {
            plain.push(b'\\');
            index += 2;
        } else if rest.starts_with(b"\\X2\\") || rest.starts_with(b"\\X4\\") {
            let width = if rest[2] == b'2' { 4 } else { 8 };
            let body_start = index + 4;
            let Some(end) = raw[body_start..]
                .windows(4)
                .position(|window| window == b"\\X0\\")
            else {
                plain.extend_from_slice(rest);
                break;
            };
            flush(&mut plain, &mut out);
            let body = &raw[body_start..body_start + end];
            let units: Vec<u32> = body.chunks(width).filter_map(hex).collect();
            if width == 4 {
                let units: Vec<u16> = units.iter().map(|unit| *unit as u16).collect();
                out.extend(char::decode_utf16(units).map(|c| c.unwrap_or('\u{fffd}')));
            } else {
                out.extend(
                    units
                        .iter()
                        .map(|unit| char::from_u32(*unit).unwrap_or('\u{fffd}')),
                );
            }
            index = body_start + end + 4;
        } else if rest.starts_with(b"\\X\\") && rest.len() >= 5 {
            flush(&mut plain, &mut out);
            if let Some(value) = hex(&rest[3..5]) {
                out.push(char::from_u32(value).unwrap_or('\u{fffd}'));
            }
            index += 5;
        } else if rest.starts_with(b"\\S\\") && rest.len() >= 4 {
            flush(&mut plain, &mut out);
            out.push(char::from_u32(u32::from(rest[3]) + 128).unwrap_or('\u{fffd}'));
            index += 4;
        } else if rest.starts_with(b"\\P") && rest.len() >= 4 && rest[3] == b'\\' {
            index += 4;
        } else {
            plain.push(raw[index]);
            index += 1;
        }
    }
    flush(&mut plain, &mut out);
    out
}

struct Parser<'a> {
    lexer: Lexer<'a>,
    peeked: Option<Token>,
}

impl<'a> Parser<'a> {
    fn peek(&mut self) -> Result<&Token, StepSyntaxError> {
        if self.peeked.is_none() {
            self.peeked = Some(self.lexer.next()?);
        }
        Ok(self.peeked.as_ref().unwrap_or(&Token::End))
    }

    fn take(&mut self) -> Result<Token, StepSyntaxError> {
        match self.peeked.take() {
            Some(token) => Ok(token),
            None => self.lexer.next(),
        }
    }

    fn expect(&mut self, expected: Token, what: &str) -> Result<(), StepSyntaxError> {
        let token = self.take()?;
        if token == expected {
            Ok(())
        } else {
            Err(self.lexer.error(format!("expected {what}")))
        }
    }

    /// `( param, ... )`, the opening parenthesis already consumed.
    fn params(&mut self, depth: usize) -> Result<Vec<Param>, StepSyntaxError> {
        if depth > MAX_PARAMETER_DEPTH {
            return Err(self.lexer.error("parameters are nested too deeply"));
        }
        let mut params = Vec::new();
        if *self.peek()? == Token::Close {
            self.take()?;
            return Ok(params);
        }
        loop {
            params.push(self.param(depth)?);
            match self.take()? {
                Token::Comma => {}
                Token::Close => return Ok(params),
                _ => return Err(self.lexer.error("expected `,` or `)` in a parameter list")),
            }
        }
    }

    fn param(&mut self, depth: usize) -> Result<Param, StepSyntaxError> {
        Ok(match self.take()? {
            Token::Integer(value) => Param::Integer(value),
            Token::Real(value) => Param::Real(value),
            Token::String(value) => Param::String(value),
            Token::Enum(value) => Param::Enum(value),
            Token::Binary(value) => Param::Binary(value),
            Token::Instance(id) => Param::Reference(id),
            Token::Dollar => Param::Unset,
            Token::Star => Param::Derived,
            Token::Open => Param::List(self.params(depth + 1)?),
            Token::Keyword(name) => {
                self.expect(Token::Open, "`(` after a typed parameter")?;
                let mut inner = self.params(depth + 1)?;
                let value = if inner.len() == 1 {
                    inner.pop().unwrap_or(Param::Unset)
                } else {
                    Param::List(inner)
                };
                Param::Typed(name, Box::new(value))
            }
            _ => return Err(self.lexer.error("expected a parameter")),
        })
    }

    fn record(&mut self, name: String) -> Result<Record, StepSyntaxError> {
        self.expect(Token::Open, "`(` after an entity name")?;
        Ok(Record {
            name,
            params: self.params(0)?,
        })
    }

    /// Skips to just past the next `;` outside strings and comments.
    fn skip_statement(&mut self) -> Result<(), StepSyntaxError> {
        loop {
            match self.take()? {
                Token::Semicolon => return Ok(()),
                Token::End => return Err(self.lexer.error("unexpected end of file")),
                _ => {}
            }
        }
    }
}

/// Parses a STEP physical file.
pub(crate) fn parse_step(bytes: &[u8]) -> Result<StepFile, StepSyntaxError> {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    let mut parser = Parser {
        lexer: Lexer { bytes, position: 0 },
        peeked: None,
    };
    match parser.take() {
        Ok(Token::Keyword(keyword)) if keyword == "ISO-10303-21" => {}
        _ => return Err(StepSyntaxError::NotStep),
    }
    parser.expect(Token::Semicolon, "`;` after ISO-10303-21")?;

    let mut file = StepFile::default();
    let mut in_data = false;
    loop {
        match parser.take()? {
            Token::End => {
                return Err(parser
                    .lexer
                    .error("unexpected end of file before END-ISO-10303-21"));
            }
            Token::Keyword(keyword) => match keyword.as_str() {
                "END-ISO-10303-21" => break,
                "DATA" => {
                    // `DATA;` or `DATA(name, schema);`
                    in_data = true;
                    parser.skip_statement()?;
                }
                "ENDSEC" => {
                    in_data = false;
                    parser.expect(Token::Semicolon, "`;` after ENDSEC")?;
                }
                _ => parser.skip_statement()?,
            },
            Token::Instance(id) if in_data => {
                parser.expect(Token::Equals, "`=` after an instance name")?;
                let instance = match parser.take()? {
                    Token::Keyword(name) => Instance {
                        records: vec![parser.record(name)?],
                        complex: false,
                    },
                    Token::Open => {
                        let mut records = Vec::new();
                        loop {
                            match parser.take()? {
                                Token::Keyword(name) => records.push(parser.record(name)?),
                                Token::Close => break,
                                _ => {
                                    return Err(parser
                                        .lexer
                                        .error("expected an entity name in a complex instance"));
                                }
                            }
                        }
                        Instance {
                            records,
                            complex: true,
                        }
                    }
                    _ => return Err(parser.lexer.error("expected an entity name")),
                };
                parser.expect(Token::Semicolon, "`;` after an entity instance")?;
                if file.instances.len() >= MAX_STEP_INSTANCES {
                    return Err(StepSyntaxError::TooManyInstances);
                }
                file.instances.insert(id, instance);
            }
            _ => return Err(parser.lexer.error("unexpected token outside an instance")),
        }
    }
    Ok(file)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_and_complex_instances_with_every_parameter_kind() {
        let text = "ISO-10303-21;\nHEADER;\nFILE_NAME('a.stp','2026',(''),(''),'','','');\nENDSEC;\nDATA;\n\
            /* comment */ #1=CARTESIAN_POINT('it''s',(0.,1.5,-2.E-1));\n\
            #2 = DIRECTION('',(1,0,0));\n\
            #3=( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );\n\
            #4=MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#3);\n\
            #5=FOO($,\"0F\",.T.);\nENDSEC;\nEND-ISO-10303-21;\n";
        let file = parse_step(text.as_bytes()).unwrap();
        let point = file.get(1).unwrap().simple("CARTESIAN_POINT").unwrap();
        assert_eq!(point.string(0), Some("it's"));
        assert_eq!(
            point.param(1).unwrap().numbers().unwrap(),
            vec![0.0, 1.5, -0.2]
        );
        assert_eq!(
            file.get(2).unwrap().records[0]
                .param(1)
                .unwrap()
                .numbers()
                .unwrap(),
            vec![1.0, 0.0, 0.0]
        );
        let unit = file.get(3).unwrap();
        assert!(unit.complex && unit.has("LENGTH_UNIT"));
        let si = unit.record("SI_UNIT").unwrap();
        assert_eq!(si.param(0).unwrap().enumeration(), Some("MILLI"));
        assert_eq!(file.get(4).unwrap().records[0].number(0), Some(25.4));
        let foo = &file.get(5).unwrap().records[0];
        assert_eq!(foo.params[0], Param::Unset);
        assert_eq!(foo.params[1], Param::Binary("0F".to_owned()));
        assert_eq!(foo.logical(2), Some(true));
    }

    #[test]
    fn decodes_string_directives() {
        assert_eq!(decode_string(b"caf\\X\\E9"), "café");
        assert_eq!(decode_string(b"\\X2\\00E9004B\\X0\\!"), "éK!");
        assert_eq!(decode_string(b"a\\\\b"), "a\\b");
        assert_eq!(decode_string("ü".as_bytes()), "ü");
    }

    #[test]
    fn rejects_non_step_and_reports_the_failing_line() {
        assert_eq!(parse_step(b"solid cube"), Err(StepSyntaxError::NotStep));
        let broken = b"ISO-10303-21;\nDATA;\n#1=POINT((0.,0.);\nENDSEC;\nEND-ISO-10303-21;";
        assert!(matches!(
            parse_step(broken),
            Err(StepSyntaxError::At { line: 3, .. })
        ));
        let truncated = b"ISO-10303-21;\nDATA;\n#1=POINT('',(0.,0.));\n";
        assert!(parse_step(truncated).is_err());
    }
}
