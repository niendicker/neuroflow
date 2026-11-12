use assembler::assemble;
use emulator::isa::{decode, DecodedInstruction};

#[test]
fn test_assemble_scalar_and_labels() {
    let source = r#"
        addi x1, x0, 10
        addi x2, x0, 20
        add  x3, x1, x2
        halt
    "#;

    let words = assemble(source).expect("Assembly should succeed");
    assert_eq!(words.len(), 4);

    let d0 = decode(words[0]).unwrap();
    assert_eq!(d0, DecodedInstruction::Addi { rd: 1, rs1: 0, imm: 10 });

    let d2 = decode(words[2]).unwrap();
    assert_eq!(d2, DecodedInstruction::Add { rd: 3, rs1: 1, rs2: 2 });

    let d3 = decode(words[3]).unwrap();
    assert_eq!(d3, DecodedInstruction::Halt);
}

#[test]
fn test_assemble_tensor_and_vector() {
    let source = r#"
        tzero acc0
        tmatmul acc0, td0, td1, bf16
        tacc.row v0, acc0, 0
        vgelu v1, v0
        halt
    "#;

    let words = assemble(source).expect("Assembly should succeed");
    assert_eq!(words.len(), 5);

    let d0 = decode(words[0]).unwrap();
    assert_eq!(d0, DecodedInstruction::TZero { acc_idx: 0 });

    let d1 = decode(words[1]).unwrap();
    assert_eq!(d1, DecodedInstruction::TMatMul {
        acc_idx: 0,
        td_a: 0,
        td_b: 1,
        dtype: emulator::types::DataType::Bf16,
    });

    let d2 = decode(words[2]).unwrap();
    assert_eq!(d2, DecodedInstruction::TAccRow { vd: 0, acc_idx: 0, row_idx: 0 });

    let d3 = decode(words[3]).unwrap();
    assert_eq!(d3, DecodedInstruction::VGelu { vd: 1, vs: 0 });
}

#[test]
fn test_assemble_hardware_loop() {
    let source = r#"
        addi x1, x0, 8
        loop x1, 2
        addi x2, x2, 1
        addi x3, x3, 2
        halt
    "#;

    let words = assemble(source).expect("Assembly should succeed");
    assert_eq!(words.len(), 5);

    let d1 = decode(words[1]).unwrap();
    assert_eq!(d1, DecodedInstruction::Loop { count_reg: 1, num_instructions: 2 });
}
