pub mod encoder;
pub mod lexer;
pub mod parser;

pub use encoder::Assembler;
pub use lexer::Lexer;
pub use parser::Parser;

pub fn assemble(source: &str) -> Result<Vec<u32>, String> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer
        .tokenize()
        .map_err(|e| format!("Lexer error on line {}: {}", e.line, e.message))?;

    let mut parser = Parser::new(tokens);
    let statements = parser
        .parse()
        .map_err(|e| format!("Parser error on line {}: {}", e.line, e.message))?;

    let mut assembler = Assembler::new();
    assembler
        .assemble(&statements)
        .map_err(|e| format!("Encoder error on line {}: {}", e.line, e.message))
}
