//! The YAML subset OpenColorIO configs are written in: block mappings and
//! sequences, flow `[…]` and `{…}` collections (which may span lines),
//! `!<Tag>` local tags, quoted and plain scalars, `|`/`>` block scalars,
//! and comments. Anchors, aliases, and multi-document streams are not
//! used by OCIO configs and are not supported.

/// A parsed node: an optional `!<Tag>` and a value.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub tag: Option<String>,
    pub value: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Scalar(String),
    Seq(Vec<Node>),
    Map(Vec<(String, Node)>),
}

impl Node {
    fn plain(value: Value) -> Self {
        Self { tag: None, value }
    }

    pub fn get(&self, key: &str) -> Option<&Node> {
        match &self.value {
            Value::Map(entries) => entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, node)| node),
            _ => None,
        }
    }

    pub fn str(&self) -> Option<&str> {
        match &self.value {
            Value::Scalar(text) => Some(text),
            _ => None,
        }
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Node::str)
    }

    pub fn seq(&self) -> &[Node] {
        match &self.value {
            Value::Seq(items) => items,
            _ => &[],
        }
    }

    pub fn entries(&self) -> &[(String, Node)] {
        match &self.value {
            Value::Map(entries) => entries,
            _ => &[],
        }
    }

    /// A scalar as a number.
    pub fn number(&self) -> Option<f64> {
        self.str()?
            .trim()
            .parse()
            .ok()
            .filter(|value: &f64| value.is_finite())
    }

    /// A sequence of numbers, or one number.
    pub fn numbers(&self) -> Option<Vec<f64>> {
        match &self.value {
            Value::Seq(items) => items.iter().map(Node::number).collect(),
            Value::Scalar(_) => self.number().map(|value| vec![value]),
            _ => None,
        }
    }

    pub fn boolean(&self) -> Option<bool> {
        match self.str()?.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "on" => Some(true),
            "false" | "no" | "off" => Some(false),
            _ => None,
        }
    }
}

struct Line {
    number: usize,
    indent: usize,
    text: String,
    /// The text of a block scalar (`|` or `>`) that this line introduces.
    block: Option<String>,
}

/// Parses a YAML document of the subset described in the module docs.
pub fn parse(text: &str) -> Result<Node, String> {
    let lines = logical_lines(text)?;
    let mut parser = Parser { lines, pos: 0 };
    if parser.lines.is_empty() {
        return Ok(Node::plain(Value::Null));
    }
    let indent = parser.lines[0].indent;
    let node = parser.block(indent)?;
    if let Some(line) = parser.lines.get(parser.pos) {
        return Err(format!("line {}: unexpected indentation", line.number));
    }
    Ok(node)
}

/// Removes a comment (`#` at the start or after whitespace, outside
/// quotes) from a line.
fn strip_comment(line: &str) -> &str {
    let mut quote = None;
    let mut previous = ' ';
    for (index, c) in line.char_indices() {
        match quote {
            Some(open) if c == open => quote = None,
            Some(_) => {}
            None if c == '"' || c == '\'' => quote = Some(c),
            None if c == '#' && previous.is_whitespace() => return &line[..index],
            None => {}
        }
        previous = c;
    }
    line
}

/// How many flow brackets `text` leaves open, outside quotes.
fn open_brackets(text: &str) -> i32 {
    let mut depth = 0;
    let mut quote = None;
    for c in text.chars() {
        match quote {
            Some(open) if c == open => quote = None,
            Some(_) => {}
            None => match c {
                '"' | '\'' => quote = Some(c),
                '[' | '{' => depth += 1,
                ']' | '}' => depth -= 1,
                _ => {}
            },
        }
    }
    depth
}

/// Splits the text into indented logical lines: comments and blank lines
/// dropped, flow collections joined across lines, and block scalars
/// gathered onto the line that introduces them.
fn logical_lines(text: &str) -> Result<Vec<Line>, String> {
    let raw: Vec<&str> = text.lines().collect();
    let mut lines = Vec::new();
    let mut index = 0;
    while index < raw.len() {
        let number = index + 1;
        let line = raw[index].trim_end_matches('\r');
        index += 1;
        if line.contains('\t') && line.trim_start().starts_with('\t') {
            return Err(format!("line {number}: tabs cannot indent YAML"));
        }
        let content = strip_comment(line);
        let trimmed = content.trim_end();
        if trimmed.trim().is_empty()
            || trimmed.trim() == "---"
            || trimmed.trim_start().starts_with('%')
        {
            continue;
        }
        let indent = trimmed.len() - trimmed.trim_start().len();
        let mut text = trimmed.trim_start().to_owned();
        let mut depth = open_brackets(&text);
        while depth > 0 && index < raw.len() {
            let next = strip_comment(raw[index].trim_end_matches('\r')).trim();
            index += 1;
            text.push(' ');
            text.push_str(next);
            depth = open_brackets(&text);
        }
        if depth != 0 {
            return Err(format!("line {number}: unbalanced brackets"));
        }
        let mut block = None;
        let indicator = text
            .rsplit_once(": ")
            .map(|(_, value)| value.trim())
            .or_else(|| text.strip_prefix("- ").map(str::trim));
        if let Some(indicator) = indicator
            && matches!(indicator, "|" | "|-" | "|+" | ">" | ">-" | ">+")
        {
            let folded = indicator.starts_with('>');
            let mut gathered: Vec<String> = Vec::new();
            let mut block_indent = None;
            while index < raw.len() {
                let candidate = raw[index].trim_end_matches('\r');
                if candidate.trim().is_empty() {
                    gathered.push(String::new());
                    index += 1;
                    continue;
                }
                let candidate_indent = candidate.len() - candidate.trim_start().len();
                if candidate_indent <= indent {
                    break;
                }
                let block_indent = *block_indent.get_or_insert(candidate_indent);
                gathered.push(
                    candidate
                        .get(block_indent..)
                        .unwrap_or(candidate.trim())
                        .to_owned(),
                );
                index += 1;
            }
            while gathered.last().is_some_and(String::is_empty) {
                gathered.pop();
            }
            block = Some(gathered.join(if folded { " " } else { "\n" }));
        }
        lines.push(Line {
            number,
            indent,
            text,
            block,
        });
    }
    Ok(lines)
}

struct Parser {
    lines: Vec<Line>,
    pos: usize,
}

/// Splits `key: value` at the first `:` followed by a space or the end,
/// outside quotes and brackets.
fn split_key(text: &str) -> Option<(String, &str)> {
    let mut quote = None;
    let mut depth = 0;
    let bytes = text.as_bytes();
    for (index, c) in text.char_indices() {
        match quote {
            Some(open) if c == open => quote = None,
            Some(_) => {}
            None => match c {
                '"' | '\'' if index == 0 => quote = Some(c),
                '[' | '{' => depth += 1,
                ']' | '}' => depth -= 1,
                ':' if depth == 0 && bytes.get(index + 1).is_none_or(|next| *next == b' ') => {
                    let key = unquote(text[..index].trim());
                    return Some((key, text[index + 1..].trim()));
                }
                _ => {}
            },
        }
    }
    None
}

fn unquote(text: &str) -> String {
    let bytes = text.as_bytes();
    if text.len() >= 2 && bytes[0] == b'"' && bytes[text.len() - 1] == b'"' {
        let inner = &text[1..text.len() - 1];
        let mut out = String::new();
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some(other) => out.push(other),
                    None => {}
                }
            } else {
                out.push(c);
            }
        }
        out
    } else if text.len() >= 2 && bytes[0] == b'\'' && bytes[text.len() - 1] == b'\'' {
        text[1..text.len() - 1].replace("''", "'")
    } else {
        text.to_owned()
    }
}

/// A `!<Tag>` (or `!Tag`) at the start of `text`, and the rest.
fn split_tag(text: &str) -> (Option<String>, &str) {
    if let Some(rest) = text.strip_prefix("!<")
        && let Some(end) = rest.find('>')
    {
        return (Some(rest[..end].to_owned()), rest[end + 1..].trim());
    }
    if let Some(rest) = text.strip_prefix('!') {
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        return (Some(rest[..end].to_owned()), rest[end..].trim());
    }
    (None, text)
}

impl Parser {
    fn block(&mut self, indent: usize) -> Result<Node, String> {
        let Some(line) = self.lines.get(self.pos) else {
            return Ok(Node::plain(Value::Null));
        };
        if line.text == "-" || line.text.starts_with("- ") {
            self.sequence(indent)
        } else if split_key(&line.text).is_some() {
            self.mapping(indent)
        } else {
            // A lone scalar or flow collection.
            let text = line.text.clone();
            self.pos += 1;
            inline(&text, line_number(&self.lines, self.pos - 1))
        }
    }

    /// The value after `key:` or `-` whose inline text is `rest`: a block
    /// scalar, an inline value, or a nested block on the following lines.
    fn value(&mut self, rest: &str, block: Option<String>, parent: usize) -> Result<Node, String> {
        let (tag, rest) = split_tag(rest);
        if let Some(block) = block {
            return Ok(Node {
                tag,
                value: Value::Scalar(block),
            });
        }
        if !rest.is_empty() {
            let mut node = inline(rest, line_number(&self.lines, self.pos - 1))?;
            if tag.is_some() {
                node.tag = tag;
            }
            return Ok(node);
        }
        let nested = match self.lines.get(self.pos) {
            Some(next) if next.indent > parent => {
                let indent = next.indent;
                self.block(indent)?
            }
            // A sequence may sit at its key's own indentation.
            Some(next)
                if next.indent == parent && (next.text == "-" || next.text.starts_with("- ")) =>
            {
                self.sequence(parent)?
            }
            _ => Node::plain(if tag.is_some() {
                Value::Map(Vec::new())
            } else {
                Value::Null
            }),
        };
        Ok(Node {
            tag: tag.or(nested.tag),
            value: nested.value,
        })
    }

    fn sequence(&mut self, indent: usize) -> Result<Node, String> {
        let mut items = Vec::new();
        while let Some(line) = self.lines.get(self.pos) {
            if line.indent != indent || !(line.text == "-" || line.text.starts_with("- ")) {
                break;
            }
            let rest = line.text[1..].trim_start().to_owned();
            let offset = line.text.len() - rest.len();
            let (tag, after_tag) = split_tag(&rest);
            let after_tag = after_tag.to_owned();
            if !after_tag.is_empty()
                && !after_tag.starts_with(['[', '{', '"', '\''])
                && split_key(&after_tag).is_some()
                && line.block.is_none()
            {
                // `- key: value` starts a mapping indented like `key`.
                let line = &mut self.lines[self.pos];
                line.indent = indent + offset;
                line.text = after_tag;
                let mut node = self.mapping(indent + offset)?;
                node.tag = tag;
                items.push(node);
                continue;
            }
            let block = line.block.clone();
            self.pos += 1;
            items.push(self.value(&rest, block, indent)?);
        }
        Ok(Node::plain(Value::Seq(items)))
    }

    fn mapping(&mut self, indent: usize) -> Result<Node, String> {
        let mut entries = Vec::new();
        while let Some(line) = self.lines.get(self.pos) {
            if line.indent < indent {
                break;
            }
            if line.indent > indent {
                return Err(format!("line {}: unexpected indentation", line.number));
            }
            if line.text == "-" || line.text.starts_with("- ") {
                break;
            }
            let number = line.number;
            let (key, rest) = split_key(&line.text)
                .map(|(key, rest)| (key, rest.to_owned()))
                .ok_or_else(|| format!("line {number}: expected `key: value`"))?;
            let block = line.block.clone();
            self.pos += 1;
            let node = self.value(&rest, block, indent)?;
            entries.push((key, node));
        }
        Ok(Node::plain(Value::Map(entries)))
    }
}

fn line_number(lines: &[Line], index: usize) -> usize {
    lines.get(index).map_or(0, |line| line.number)
}

/// Parses an inline value: a flow collection, a quoted scalar, or a plain
/// scalar, optionally tagged.
fn inline(text: &str, number: usize) -> Result<Node, String> {
    let mut flow = Flow {
        chars: text.chars().collect(),
        pos: 0,
        number,
    };
    let node = flow.node(false)?;
    flow.skip_space();
    if flow.pos < flow.chars.len() {
        return Err(format!("line {number}: unexpected `{}`", flow.rest()));
    }
    Ok(node)
}

struct Flow {
    chars: Vec<char>,
    pos: usize,
    number: usize,
}

impl Flow {
    fn rest(&self) -> String {
        self.chars[self.pos..].iter().collect()
    }

    fn skip_space(&mut self) {
        while self.chars.get(self.pos).is_some_and(|c| c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn error(&self, what: &str) -> String {
        format!("line {}: {what}", self.number)
    }

    /// One node; `in_flow` stops plain scalars at `,`, `]`, and `}`.
    fn node(&mut self, in_flow: bool) -> Result<Node, String> {
        self.skip_space();
        let mut tag = None;
        if self.peek() == Some('!') {
            let start = self.pos;
            if self.chars.get(self.pos + 1) == Some(&'<') {
                let end = self.chars[self.pos..]
                    .iter()
                    .position(|c| *c == '>')
                    .ok_or_else(|| self.error("unclosed tag"))?;
                tag = Some(self.chars[start + 2..self.pos + end].iter().collect());
                self.pos += end + 1;
            } else {
                while self.peek().is_some_and(|c| !c.is_whitespace()) {
                    self.pos += 1;
                }
                tag = Some(self.chars[start + 1..self.pos].iter().collect());
            }
            self.skip_space();
        }
        let value = match self.peek() {
            Some('[') => self.sequence()?,
            Some('{') => self.mapping()?,
            Some('"' | '\'') => Value::Scalar(self.quoted()?),
            None if tag.is_some() => Value::Map(Vec::new()),
            Some(',' | ']' | '}') if tag.is_some() => Value::Map(Vec::new()),
            _ => {
                let text = self.plain(in_flow, false);
                if text.is_empty() || text == "~" || text == "null" {
                    Value::Null
                } else {
                    Value::Scalar(text)
                }
            }
        };
        Ok(Node { tag, value })
    }

    fn quoted(&mut self) -> Result<String, String> {
        let open = self.peek().unwrap_or('"');
        let start = self.pos;
        self.pos += 1;
        loop {
            match self.peek() {
                None => return Err(self.error("unclosed quote")),
                Some('\\') if open == '"' => self.pos += 2,
                Some(c) if c == open => {
                    if open == '\'' && self.chars.get(self.pos + 1) == Some(&'\'') {
                        self.pos += 2;
                        continue;
                    }
                    self.pos += 1;
                    break;
                }
                Some(_) => self.pos += 1,
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        Ok(unquote(&text))
    }

    /// A plain scalar; keys stop at `: `.
    fn plain(&mut self, in_flow: bool, key: bool) -> String {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if in_flow && matches!(c, ',' | ']' | '}') {
                break;
            }
            if key
                && c == ':'
                && self
                    .chars
                    .get(self.pos + 1)
                    .is_none_or(|next| next.is_whitespace() || matches!(next, ',' | '}' | ']'))
            {
                break;
            }
            self.pos += 1;
        }
        self.chars[start..self.pos]
            .iter()
            .collect::<String>()
            .trim()
            .to_owned()
    }

    fn sequence(&mut self) -> Result<Value, String> {
        self.pos += 1;
        let mut items = Vec::new();
        loop {
            self.skip_space();
            match self.peek() {
                Some(']') => {
                    self.pos += 1;
                    return Ok(Value::Seq(items));
                }
                None => return Err(self.error("unclosed `[`")),
                _ => {}
            }
            items.push(self.node(true)?);
            self.skip_space();
            match self.peek() {
                Some(',') => self.pos += 1,
                Some(']') => {}
                _ => return Err(self.error("expected `,` or `]`")),
            }
        }
    }

    fn mapping(&mut self) -> Result<Value, String> {
        self.pos += 1;
        let mut entries = Vec::new();
        loop {
            self.skip_space();
            match self.peek() {
                Some('}') => {
                    self.pos += 1;
                    return Ok(Value::Map(entries));
                }
                None => return Err(self.error("unclosed `{`")),
                _ => {}
            }
            let key = if matches!(self.peek(), Some('"' | '\'')) {
                self.quoted()?
            } else {
                self.plain(true, true)
            };
            self.skip_space();
            let value = if self.peek() == Some(':') {
                self.pos += 1;
                self.node(true)?
            } else {
                Node::plain(Value::Null)
            };
            entries.push((key, value));
            self.skip_space();
            match self.peek() {
                Some(',') => self.pos += 1,
                Some('}') => {}
                _ => return Err(self.error("expected `,` or `}`")),
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn ocio_style_yaml_parses() {
        let text = "\
ocio_profile_version: 2   # comment
search_path: \"luts:more\"
roles:
  scene_linear: Linear Rec.709
description: |
  First line
  # not a comment

  Third line
displays:
  sRGB:
    - !<View> {name: Standard, colorspace: sRGB}
    - !<View> {name: \"Filmic: Base\", view_transform: Filmic, looks: +Punchy}
active_displays: [sRGB, 'Rec.1886']
colorspaces:
  - !<ColorSpace>
    name: Linear
    isdata: false
    to_reference: !<GroupTransform>
      children:
        - !<MatrixTransform> {matrix: [1, 0, 0, 0,
                                        0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}
        - !<FileTransform> {src: a.spi1d, interpolation: linear}
  - !<ColorSpace>
    name: Raw
    from_reference: !<ExponentTransform> {value: 2.2, direction: inverse}
list:
- a
- b
";
        let root = parse(text).unwrap();
        assert_eq!(root.get_str("ocio_profile_version"), Some("2"));
        assert_eq!(root.get_str("search_path"), Some("luts:more"));
        assert_eq!(
            root.get("roles").unwrap().get_str("scene_linear"),
            Some("Linear Rec.709")
        );
        assert_eq!(
            root.get_str("description"),
            Some("First line\n# not a comment\n\nThird line")
        );
        let views = root.get("displays").unwrap().get("sRGB").unwrap().seq();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].tag.as_deref(), Some("View"));
        assert_eq!(views[1].get_str("name"), Some("Filmic: Base"));
        assert_eq!(views[1].get_str("looks"), Some("+Punchy"));
        let active: Vec<&str> = root
            .get("active_displays")
            .unwrap()
            .seq()
            .iter()
            .filter_map(Node::str)
            .collect();
        assert_eq!(active, ["sRGB", "Rec.1886"]);
        let spaces = root.get("colorspaces").unwrap().seq();
        assert_eq!(spaces.len(), 2);
        assert_eq!(spaces[0].tag.as_deref(), Some("ColorSpace"));
        assert_eq!(spaces[0].get("isdata").unwrap().boolean(), Some(false));
        let group = spaces[0].get("to_reference").unwrap();
        assert_eq!(group.tag.as_deref(), Some("GroupTransform"));
        let children = group.get("children").unwrap().seq();
        assert_eq!(
            children[0].get("matrix").unwrap().numbers().unwrap().len(),
            16
        );
        assert_eq!(children[1].get_str("src"), Some("a.spi1d"));
        let exponent = spaces[1].get("from_reference").unwrap();
        assert_eq!(exponent.get("value").unwrap().numbers(), Some(vec![2.2]));
        let list: Vec<&str> = root
            .get("list")
            .unwrap()
            .seq()
            .iter()
            .filter_map(Node::str)
            .collect();
        assert_eq!(list, ["a", "b"]);

        assert!(parse("a: [1, 2").is_err());
        assert!(parse("a: b\n    c: d\n").is_err());
    }
}
