/// AST and Parser for NeuroFlow Assembly

use crate::lexer::Token;

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    ScalarReg(usize),
    VectorReg(usize),
    TileDescReg(usize),
    AccReg(usize),
    Imm(i64),
    Label(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedInstruction {
    pub line: usize,
    pub mnemonic: String,
    pub operands: Vec<Operand>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Label(String),
    Instruction(ParsedInstruction),
    Directive { name: String, args: Vec<Operand> },
}

#[derive(Debug, Clone)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

pub struct Parser {
    tokens: Vec<(Token, usize)>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<(Token, usize)>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&(Token, usize)> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<(Token, usize)> {
        if self.pos < self.tokens.len() {
            let item = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(item)
        } else {
            None
        }
    }

    pub fn parse(&mut self) -> Result<Vec<Statement>, ParseError> {
        let mut statements = Vec::new();

        while self.peek().is_some() {
            // Skip empty newlines
            if let Some((Token::Newline, _)) = self.peek() {
                self.advance();
                continue;
            }

            let stmt = self.parse_statement()?;
            statements.push(stmt);
        }

        Ok(statements)
    }

    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        let (tok, line) = self.advance().ok_or(ParseError {
            line: 0,
            message: "Unexpected end of input".to_string(),
        })?;

        match tok {
            Token::Ident(name) => {
                // Check if this is a label (followed by ':')
                if let Some((Token::Colon, _)) = self.peek() {
                    self.advance(); // consume ':'
                    return Ok(Statement::Label(name));
                }

                // Check if directive (starts with '.')
                if name.starts_with('.') {
                    let mut args = Vec::new();
                    while let Some((t, _)) = self.peek() {
                        if *t == Token::Newline { break; }
                        if *t == Token::Comma {
                            self.advance();
                            continue;
                        }
                        args.push(self.parse_operand()?);
                    }
                    if let Some((Token::Newline, _)) = self.peek() {
                        self.advance();
                    }
                    return Ok(Statement::Directive { name, args });
                }

                // Otherwise, an instruction
                let mut operands = Vec::new();
                while let Some((t, _)) = self.peek() {
                    if *t == Token::Newline { break; }
                    if *t == Token::Comma {
                        self.advance();
                        continue;
                    }
                    operands.push(self.parse_operand()?);
                }

                if let Some((Token::Newline, _)) = self.peek() {
                    self.advance();
                }

                Ok(Statement::Instruction(ParsedInstruction {
                    line,
                    mnemonic: name.to_lowercase(),
                    operands,
                }))
            }
            _ => Err(ParseError {
                line,
                message: format!("Expected identifier or label, found {:?}", tok),
            }),
        }
    }

    fn parse_operand(&mut self) -> Result<Operand, ParseError> {
        let (tok, line) = self.advance().ok_or(ParseError {
            line: 0,
            message: "Expected operand".to_string(),
        })?;

        match tok {
            Token::ScalarReg(r) => Ok(Operand::ScalarReg(r)),
            Token::VectorReg(v) => Ok(Operand::VectorReg(v)),
            Token::TileDescReg(td) => Ok(Operand::TileDescReg(td)),
            Token::AccReg(acc) => Ok(Operand::AccReg(acc)),
            Token::Number(n) => Ok(Operand::Imm(n)),
            Token::Ident(id) => Ok(Operand::Label(id)),
            Token::LBracket => {
                // Support [reg] or [reg + imm]
                let inner = self.parse_operand()?;
                if let Some((Token::RBracket, _)) = self.peek() {
                    self.advance();
                    Ok(inner)
                } else {
                    Err(ParseError {
                        line,
                        message: "Expected closing ']'".to_string(),
                    })
                }
            }
            _ => Err(ParseError {
                line,
                message: format!("Invalid operand token: {:?}", tok),
            }),
        }
    }
}
