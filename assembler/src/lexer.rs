/// Lexer for NeuroFlow Assembly (.s)

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Ident(String),
    ScalarReg(usize),    // x0-x31
    VectorReg(usize),    // v0-v7
    TileDescReg(usize),  // td0-td7
    AccReg(usize),       // acc0-acc3
    Number(i64),
    Colon,
    Comma,
    LBracket,
    RBracket,
    Newline,
}

#[derive(Debug, Clone)]
pub struct LexerError {
    pub line: usize,
    pub message: String,
}

pub struct Lexer<'a> {
    _input: &'a str,
    chars: Vec<(usize, char)>,
    pos: usize,
    line: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let chars: Vec<(usize, char)> = input.char_indices().collect();
        Self {
            _input: input,
            chars,
            pos: 0,
            line: 1,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).map(|&(_, c)| c)
    }

    fn advance(&mut self) -> Option<char> {
        if self.pos < self.chars.len() {
            let (_, c) = self.chars[self.pos];
            self.pos += 1;
            if c == '\n' {
                self.line += 1;
            }
            Some(c)
        } else {
            None
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<(Token, usize)>, LexerError> {
        let mut tokens = Vec::new();

        while let Some(c) = self.peek() {
            match c {
                ' ' | '\t' | '\r' => {
                    self.advance();
                }
                '\n' => {
                    let cur_line = self.line;
                    self.advance();
                    // Avoid duplicate consecutive newlines
                    if let Some((Token::Newline, _)) = tokens.last() {
                        continue;
                    }
                    tokens.push((Token::Newline, cur_line));
                }
                '#' | '/' => {
                    // Comment until newline
                    if c == '/' {
                        self.advance();
                        if self.peek() == Some('/') {
                            while let Some(nc) = self.peek() {
                                if nc == '\n' { break; }
                                self.advance();
                            }
                        } else {
                            return Err(LexerError {
                                line: self.line,
                                message: "Unexpected single '/'".to_string(),
                            });
                        }
                    } else {
                        while let Some(nc) = self.peek() {
                            if nc == '\n' { break; }
                            self.advance();
                        }
                    }
                }
                ':' => {
                    let cur_line = self.line;
                    self.advance();
                    tokens.push((Token::Colon, cur_line));
                }
                ',' => {
                    let cur_line = self.line;
                    self.advance();
                    tokens.push((Token::Comma, cur_line));
                }
                '[' => {
                    let cur_line = self.line;
                    self.advance();
                    tokens.push((Token::LBracket, cur_line));
                }
                ']' => {
                    let cur_line = self.line;
                    self.advance();
                    tokens.push((Token::RBracket, cur_line));
                }
                '-' | '0'..='9' => {
                    let cur_line = self.line;
                    let num = self.read_number()?;
                    tokens.push((Token::Number(num), cur_line));
                }
                'a'..='z' | 'A'..='Z' | '_' | '.' => {
                    let cur_line = self.line;
                    let ident = self.read_ident();
                    let tok = self.classify_ident(&ident);
                    tokens.push((tok, cur_line));
                }
                _ => {
                    return Err(LexerError {
                        line: self.line,
                        message: format!("Unexpected character: '{}'", c),
                    });
                }
            }
        }

        // Final newline if not present
        if let Some((tok, _)) = tokens.last() {
            if *tok != Token::Newline {
                tokens.push((Token::Newline, self.line));
            }
        }

        Ok(tokens)
    }

    fn read_ident(&mut self) -> String {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' || c == '.' {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }
        s
    }

    fn read_number(&mut self) -> Result<i64, LexerError> {
        let mut s = String::new();
        let is_negative = if self.peek() == Some('-') {
            s.push('-');
            self.advance();
            true
        } else {
            false
        };

        if self.peek() == Some('0') {
            s.push('0');
            self.advance();
            if self.peek() == Some('x') || self.peek() == Some('X') {
                s.push('x');
                self.advance();
                while let Some(c) = self.peek() {
                    if c.is_ascii_hexdigit() {
                        s.push(c);
                        self.advance();
                    } else {
                        break;
                    }
                }
                let hex_str = s.trim_start_matches('-').trim_start_matches("0x").trim_start_matches("0X");
                let val = i64::from_str_radix(hex_str, 16).map_err(|_| LexerError {
                    line: self.line,
                    message: format!("Invalid hex literal: {}", s),
                })?;
                return Ok(if is_negative { -val } else { val });
            }
        }

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }

        s.parse::<i64>().map_err(|_| LexerError {
            line: self.line,
            message: format!("Invalid number literal: {}", s),
        })
    }

    fn classify_ident(&self, ident: &str) -> Token {
        let lower = ident.to_lowercase();

        // Scalar Registers
        if lower == "zero" { return Token::ScalarReg(0); }
        if lower == "ra" { return Token::ScalarReg(1); }
        if lower == "sp" { return Token::ScalarReg(2); }
        if lower == "tp" { return Token::ScalarReg(3); }

        if let Some(rest) = lower.strip_prefix('x') {
            if let Ok(num) = rest.parse::<usize>() {
                if num < 32 { return Token::ScalarReg(num); }
            }
        }
        if let Some(rest) = lower.strip_prefix('a') {
            if let Ok(num) = rest.parse::<usize>() {
                if num <= 7 { return Token::ScalarReg(4 + num); } // a0-a7 -> x4-x11
            }
        }
        if let Some(rest) = lower.strip_prefix('t') {
            if let Ok(num) = rest.parse::<usize>() {
                if num <= 7 { return Token::ScalarReg(12 + num); } // t0-t7 -> x12-x19
            }
        }
        if let Some(rest) = lower.strip_prefix('s') {
            if let Ok(num) = rest.parse::<usize>() {
                if num <= 11 { return Token::ScalarReg(20 + num); } // s0-s11 -> x20-x31
            }
        }

        // Vector Registers (v0-v7)
        if let Some(rest) = lower.strip_prefix('v') {
            if let Ok(num) = rest.parse::<usize>() {
                if num < 8 { return Token::VectorReg(num); }
            }
        }

        // Tile Descriptors (td0-td7)
        if let Some(rest) = lower.strip_prefix("td") {
            if let Ok(num) = rest.parse::<usize>() {
                if num < 8 { return Token::TileDescReg(num); }
            }
        }

        // Accumulators (acc0-acc3)
        if let Some(rest) = lower.strip_prefix("acc") {
            if let Ok(num) = rest.parse::<usize>() {
                if num < 4 { return Token::AccReg(num); }
            }
        }

        Token::Ident(ident.to_string())
    }
}
