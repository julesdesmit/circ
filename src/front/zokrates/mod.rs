//! The ZoKrates front-end

mod parser;
mod term;

use super::FrontEnd;
use crate::circify::{Circify, Loc, Val};
use crate::ir::proof::{self, ConstraintMetadata};
use crate::ir::term::*;
use log::debug;
use rug::Integer;
use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use zokrates_pest_ast as ast;

use term::*;

/// The modulus for the ZoKrates language.
pub use term::ZOKRATES_MODULUS;
/// The modulus for the ZoKrates language.
pub use term::ZOKRATES_MODULUS_ARC;

/// The prover visibility
pub const PROVER_VIS: Option<PartyId> = Some(proof::PROVER_ID);
/// Public visibility
pub const PUBLIC_VIS: Option<PartyId> = None;

/// Inputs to the ZoKrates compilier
pub struct Inputs {
    /// The file to look for `main` in.
    pub file: PathBuf,
    /// The file to look for concrete arguments to main in. Optional.
    ///
    /// ## Examples
    ///
    /// If main takes `x: u64, y: field`, this file might contain
    ///
    /// ```ignore
    /// x 4
    /// y -1
    /// ```
    pub inputs: Option<PathBuf>,
    /// The mode to generate for (MPC or proof). Effects visibility.
    pub mode: Mode,
}

#[derive(Clone, Copy, Debug)]
/// Kind of circuit to generate. Effects privacy labels.
pub enum Mode {
    /// Generating an MPC circuit. Inputs are public or private (to a party in 1..N).
    Mpc(u8),
    /// Generating for a proof circuit. Inputs are public of private (to the prover).
    Proof,
    /// Generating for an optimization circuit. Inputs are existentially quantified.
    /// There should be only one output, which will be maximized.
    Opt,
    /// Find inputs that yeild an output at least this large,
    /// and then prove knowledge of them.
    ProofOfHighValue(u64),
}

impl Display for Mode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            &Mode::Mpc(n) => write!(f, "{}-pc", n),
            &Mode::Proof => write!(f, "proof"),
            &Mode::Opt => write!(f, "opt"),
            &Mode::ProofOfHighValue(v) => write!(f, "proof_of_high_value({})", v),
        }
    }
}

/// The ZoKrates front-end. Implements [FrontEnd].
pub struct Zokrates;

impl FrontEnd for Zokrates {
    type Inputs = Inputs;
    fn gen(i: Inputs) -> Computation {
        let loader = parser::ZLoad::new();
        let asts = loader.load(&i.file);
        let mut g = ZGen::new(i.inputs, asts, i.mode);
        g.visit_files();
        g.file_stack.push(i.file);
        g.entry_fn("main");
        g.file_stack.pop();
        g.circ.consume().borrow().clone()
    }
}

struct ZGen<'ast> {
    circ: Circify<ZoKrates>,
    stdlib: parser::ZStdLib,
    asts: HashMap<PathBuf, ast::File<'ast>>,
    file_stack: Vec<PathBuf>,
    functions: HashMap<(PathBuf, String), ast::FunctionDefinition<'ast>>,
    import_map: HashMap<(PathBuf, String), (PathBuf, String)>,
    mode: Mode,
}

enum ZLoc {
    Var(Loc),
    Member(Box<ZLoc>, String),
    Idx(Box<ZLoc>, T),
}

impl ZLoc {
    fn loc(&self) -> &Loc {
        match self {
            ZLoc::Var(l) => l,
            ZLoc::Member(i, _) => i.loc(),
            ZLoc::Idx(i, _) => i.loc(),
        }
    }
}

impl<'ast> ZGen<'ast> {
    fn new(inputs: Option<PathBuf>, asts: HashMap<PathBuf, ast::File<'ast>>, mode: Mode) -> Self {
        let this = Self {
            circ: Circify::new(ZoKrates::new(inputs.map(|i| parser::parse_inputs(i)))),
            asts,
            stdlib: parser::ZStdLib::new(),
            file_stack: vec![],
            functions: HashMap::default(),
            import_map: HashMap::default(),
            mode,
        };
        this.circ
            .cir_ctx()
            .cs
            .borrow_mut()
            .metadata
            .add_prover_and_verifier();
        this
    }

    /// Unwrap a result with a span-dependent error
    fn err<E: Display>(&self, e: E, s: &ast::Span) -> ! {
        println!("Error: {}", e);
        println!("In: {}", self.cur_path().display());
        for l in s.lines() {
            println!("  {}", l);
        }
        std::process::exit(1)
    }

    fn unwrap<T, E: Display>(&self, r: Result<T, E>, s: &ast::Span) -> T {
        r.unwrap_or_else(|e| self.err(e, s))
    }

    fn builtin_call(fn_name: &str, mut args: Vec<T>) -> Result<T, String> {
        match fn_name {
            "EMBED/u8_to_bits" if args.len() == 1 => uint_to_bits(args.pop().unwrap()),
            "EMBED/u16_to_bits" if args.len() == 1 => uint_to_bits(args.pop().unwrap()),
            "EMBED/u32_to_bits" if args.len() == 1 => uint_to_bits(args.pop().unwrap()),
            "EMBED/u8_from_bits" if args.len() == 1 => uint_from_bits(args.pop().unwrap()),
            "EMBED/u16_from_bits" if args.len() == 1 => uint_from_bits(args.pop().unwrap()),
            "EMBED/u32_from_bits" if args.len() == 1 => uint_from_bits(args.pop().unwrap()),
            "EMBED/unpack" if args.len() == 1 => field_to_bits(args.pop().unwrap()),
            _ => Err(format!("Unknown builtin '{}'", fn_name)),
        }
    }

    fn stmt(&mut self, s: &ast::Statement<'ast>) {
        debug!("Stmt: {}", s.span().as_str());
        match s {
            ast::Statement::Return(r) => {
                // XXX(rsw) multi-return unimplemented
                assert!(r.expressions.len() <= 1);
                if let Some(e) = r.expressions.first() {
                    let ret = self.expr(e);
                    let ret_res = self.circ.return_(Some(ret));
                    self.unwrap(ret_res, &r.span);
                } else {
                    let ret_res = self.circ.return_(None);
                    self.unwrap(ret_res, &r.span);
                }
            }
            ast::Statement::Assertion(e) => {
                let b = bool(self.expr(&e.expression));
                let e = self.unwrap(b, &e.span);
                self.circ.assert(e);
            }
            ast::Statement::Iteration(i) => {
                let ty = self.type_(&i.ty);
                let s = self.const_int(&i.from);
                let e = self.const_int(&i.to);
                let v_name = i.index.value.clone();
                self.circ.enter_scope();
                let decl_res = self.circ.declare(v_name.clone(), &ty, false, PROVER_VIS);
                self.unwrap(decl_res, &i.index.span);
                for j in s..e {
                    self.circ.enter_scope();
                    let ass_res = self
                        .circ
                        .assign(Loc::local(v_name.clone()), Val::Term(
                            match ty {
                                Ty::Uint(8) => T::Uint(8, bv_lit(j, 8)),
                                Ty::Uint(16) => T::Uint(16, bv_lit(j, 16)),
                                Ty::Uint(32) => T::Uint(32, bv_lit(j, 32)),
                                Ty::Field =>  T::Field(pf_lit(j)),
                                _ => panic!("Unexpected type for iteration: {:?}", ty),
                            }
                        ));
                    self.unwrap(ass_res, &i.index.span);
                    for s in &i.statements {
                        self.stmt(s);
                    }
                    self.circ.exit_scope();
                }
                self.circ.exit_scope();
            }
            ast::Statement::Definition(d) => {
                // XXX(rsw) multi-assignment unimplemented
                assert!(d.lhs.len() <= 1);
                let e = self.expr(&d.expression);
                if let Some(l) = d.lhs.first() {
                    let ty = e.type_();
                    match l {
                        ast::TypedIdentifierOrAssignee::Assignee(l) => {
                            let lval = self.lval(l);
                            let mod_res = self.mod_lval(lval, e);
                            self.unwrap(mod_res, &d.span);
                        }
                        ast::TypedIdentifierOrAssignee::TypedIdentifier(l) => {
                            let decl_ty = self.type_(&l.ty);
                            if decl_ty != ty {
                                self.err(
                                    format!(
                                        "Assignment type mismatch: {} annotatved vs {} actual",
                                        decl_ty,
                                        ty,
                                    ),
                                    &d.span,
                                );
                            }
                            let d_res = self
                                .circ
                                .declare_init(l.identifier.value.clone(), decl_ty, Val::Term(e));
                            self.unwrap(d_res, &d.span);
                        }
                    }
                }
            }
        }
    }

    fn apply_lval_mod(&mut self, base: T, loc: ZLoc, val: T) -> Result<T, String> {
        match loc {
            ZLoc::Var(_) => Ok(val),
            ZLoc::Member(inner_loc, field) => {
                let old_inner = field_select(&base, &field)?;
                let new_inner = self.apply_lval_mod(old_inner, *inner_loc, val)?;
                field_store(base, &field, new_inner)
            }
            ZLoc::Idx(inner_loc, idx) => {
                let old_inner = array_select(base.clone(), idx.clone())?;
                let new_inner = self.apply_lval_mod(old_inner, *inner_loc, val)?;
                array_store(base, idx, new_inner)
            }
        }
    }

    fn mod_lval(&mut self, l: ZLoc, t: T) -> Result<(), String> {
        let var = l.loc().clone();
        let old = self
            .circ
            .get_value(var.clone())
            .map_err(|e| format!("{}", e))?
            .unwrap_term();
        let new = self.apply_lval_mod(old, l, t)?;
        self.circ
            .assign(var, Val::Term(new))
            .map_err(|e| format!("{}", e))
            .map(|_| ())
    }

    fn lval(&mut self, l: &ast::Assignee<'ast>) -> ZLoc {
        l.accesses.iter().fold(
            ZLoc::Var(Loc::local(l.id.value.clone())),
            |inner, acc| match acc {
                ast::AssigneeAccess::Member(m) => ZLoc::Member(Box::new(inner), m.id.value.clone()),
                ast::AssigneeAccess::Select(m) => {
                    let i = if let ast::RangeOrExpression::Expression(e) = &m.expression {
                        self.expr(&e)
                    } else {
                        panic!("Cannot assign to slice")
                    };
                    ZLoc::Idx(Box::new(inner), i)
                }
            },
        )
    }

    fn literal_(&mut self, e: &ast::LiteralExpression<'ast>) -> T {
        match e {
            ast::LiteralExpression::DecimalLiteral(d) => {
                let vstr = &d.value.span.as_str();
                match &d.suffix {
                    Some(ast::DecimalSuffix::U8(_)) => {
                        T::Uint(8, bv_lit(u8::from_str_radix(vstr, 10).unwrap(), 8))
                    }
                    Some(ast::DecimalSuffix::U16(_)) => {
                        T::Uint(16, bv_lit(u16::from_str_radix(vstr, 10).unwrap(), 16))
                    }
                    Some(ast::DecimalSuffix::U32(_)) => {
                        T::Uint(32, bv_lit(u32::from_str_radix(vstr, 10).unwrap(), 32))
                    }
                    Some(ast::DecimalSuffix::U64(_)) => {
                        T::Uint(64, bv_lit(u64::from_str_radix(vstr, 10).unwrap(), 64))
                    }
                    Some(ast::DecimalSuffix::Field(_)) => {
                        T::Field(pf_lit(Integer::from_str_radix(vstr, 10).unwrap()))
                    }
                    // XXX(rsw) need to infer int size from context. yuck. error out.
                    _ => self.err("refusing to infer decimal literal type.", &d.span),
                }
            }
            ast::LiteralExpression::BooleanLiteral(b) => {
                Self::const_bool(bool::from_str(&b.value).unwrap())
            }
            ast::LiteralExpression::HexLiteral(h) => {
                match &h.value {
                    ast::HexNumberExpression::U8(h) => {
                        T::Uint(8, bv_lit(u8::from_str_radix(&h.value[2..], 16).unwrap(), 8))
                    }
                    ast::HexNumberExpression::U16(h) => {
                        T::Uint(16, bv_lit(u16::from_str_radix(&h.value[2..], 16).unwrap(), 16))
                    }
                    ast::HexNumberExpression::U32(h) => {
                        T::Uint(32, bv_lit(u32::from_str_radix(&h.value[2..], 16).unwrap(), 32))
                    }
                    ast::HexNumberExpression::U64(h) => {
                        T::Uint(64, bv_lit(u64::from_str_radix(&h.value[2..], 16).unwrap(), 64))
                    }
                }
            }
        }
    }

    fn const_bool(b: bool) -> T {
        T::Bool(leaf_term(Op::Const(Value::Bool(b))))
    }

    fn unary_op(&self, o: &ast::UnaryOperator) -> fn(T) -> Result<T, String> {
        match o {
            ast::UnaryOperator::Pos(_) => |x| Ok(x),
            ast::UnaryOperator::Neg(_) => neg,
            ast::UnaryOperator::Not(_) => not,
        }
    }

    fn bin_op(&self, o: &ast::BinaryOperator) -> fn(T, T) -> Result<T, String> {
        match o {
            ast::BinaryOperator::BitXor => bitxor,
            ast::BinaryOperator::BitAnd => bitand,
            ast::BinaryOperator::BitOr => bitor,
            ast::BinaryOperator::RightShift => shr,
            ast::BinaryOperator::LeftShift => shl,
            ast::BinaryOperator::Or => or,
            ast::BinaryOperator::And => and,
            ast::BinaryOperator::Add => add,
            ast::BinaryOperator::Sub => sub,
            ast::BinaryOperator::Mul => mul,
            ast::BinaryOperator::Div => div,
            ast::BinaryOperator::Rem => rem,
            ast::BinaryOperator::Eq => eq,
            ast::BinaryOperator::NotEq => neq,
            ast::BinaryOperator::Lt => ult,
            ast::BinaryOperator::Gt => ugt,
            ast::BinaryOperator::Lte => ule,
            ast::BinaryOperator::Gte => uge,
            ast::BinaryOperator::Pow => unimplemented!(),
        }
    }

    fn expr(&mut self, e: &ast::Expression<'ast>) -> T {
        debug!("Expr: {}", e.span().as_str());
        let res = match e {
            ast::Expression::Ternary(u) => {
                let c = self.expr(&u.first);
                let a = self.expr(&u.second);
                let b = self.expr(&u.third);
                cond(c, a, b)
            }
            ast::Expression::Binary(u) => {
                let f = self.bin_op(&u.op);
                let a = self.expr(&u.left);
                let b = self.expr(&u.right);
                f(a, b)
            }
            ast::Expression::Unary(u) => {
                let f = self.unary_op(&u.op);
                let a = self.expr(&u.expression);
                f(a)
            }
            ast::Expression::Identifier(u) => Ok(self
               .unwrap(self.circ.get_value(Loc::local(u.value.clone())), &u.span)
               .unwrap_term()),
            ast::Expression::InlineArray(u) => T::new_array(
                u.expressions
                    .iter()
                    .flat_map(|x| self.array_lit_elem(x))
                    .collect(),
            ),
            ast::Expression::Literal(l) => Ok(self.literal_(l)),
            ast::Expression::InlineStruct(u) => Ok(T::Struct(
                u.ty.value.clone(),
                u.members
                    .iter()
                    .map(|m| (m.id.value.clone(), self.expr(&m.expression)))
                    .collect(),
            )),
            ast::Expression::ArrayInitializer(a) => {
                let v = self.expr(&a.value);
                let ty = v.type_();
                let n = self.const_int(&a.count) as usize;
                /*
                let n = const_int(self.const_(&a.count))
                    .unwrap()
                    .to_usize()
                    .unwrap();
                */
                Ok(T::Array(ty, vec![v; n]))
            }
            ast::Expression::Postfix(p) => {
                // Assume no functions in arrays, etc.
                // XXX(rsw) is this a reasonable assumption? probably...
                let (base, accs) = if let Some(ast::Access::Call(c)) = p.accesses.first() {
                    debug!("Call: {}", p.id.value);
                    let (f_path, f_name) = self.deref_import(p.id.value.clone());
                    // XXX(rsw) no support for generics in calls yet
                    if c.explicit_generics.is_some() {
                        self.err("generic calls not yet supported", &c.span);
                    }
                    let args = c
                        .arguments
                        .expressions
                        .iter()
                        .map(|e| self.expr(e))
                        .collect::<Vec<_>>();
                    let res = if f_path.to_string_lossy().starts_with("EMBED") {
                        Self::builtin_call(f_path.to_str().unwrap(), args).unwrap()
                    } else {
                        let p = (f_path, f_name);
                        let f = self
                            .functions
                            .get(&p)
                            .unwrap_or_else(|| panic!("No function '{}'", p.1))
                            .clone();
                        self.file_stack.push(p.0);
                        assert!(f.returns.len() <= 1);
                        let ret_ty = f.returns.first().map(|r| self.type_(r));
                        self.circ.enter_fn(p.1, ret_ty);
                        assert_eq!(f.parameters.len(), args.len());
                        for (p, a) in f.parameters.iter().zip(args) {
                            let ty = self.type_(&p.ty);
                            let d_res =
                                self.circ.declare_init(p.id.value.clone(), ty, Val::Term(a));
                            self.unwrap(d_res, &c.span);
                        }
                        for s in &f.statements {
                            self.stmt(s);
                        }
                        let ret = self
                            .circ
                            .exit_fn()
                            .map(|a| a.unwrap_term())
                            .unwrap_or_else(|| Self::const_bool(false));
                        self.file_stack.pop();
                        ret
                    };
                    (res, &p.accesses[1..])
                } else {
                    // Assume no calls
                    (
                        self.unwrap(
                            self.circ.get_value(Loc::local(p.id.value.clone())),
                            &p.id.span,
                        )
                        .unwrap_term(),
                        &p.accesses[..],
                    )
                };
                accs.iter().fold(Ok(base), |b, acc| match acc {
                    ast::Access::Member(m) => field_select(&b?, &m.id.value),
                    ast::Access::Select(a) => match &a.expression {
                        ast::RangeOrExpression::Expression(e) => array_select(b?, self.expr(e)),
                        ast::RangeOrExpression::Range(r) => {
                            let s = r.from.as_ref().map(|s| self.const_int(&s.0) as usize);
                            let e = r.to.as_ref().map(|s| self.const_int(&s.0) as usize);
                            slice(b?, s, e)
                        }
                    },
                    ast::Access::Call(_) => unreachable!("stray call"),
                })
            }
        };
        self.unwrap(res, e.span())
    }
    fn array_lit_elem(&mut self, e: &ast::SpreadOrExpression<'ast>) -> Vec<T> {
        match e {
            ast::SpreadOrExpression::Expression(e) => vec![self.expr(e)],
            ast::SpreadOrExpression::Spread(s) => self.expr(&s.expression).unwrap_array().unwrap(),
        }
    }
    fn entry_fn(&mut self, n: &str) {
        debug!("Entry: {}", n);
        // find the entry function
        let (f_path, f_name) = self.deref_import(n.to_owned());
        let p = (f_path, f_name);
        let f = self
            .functions
            .get(&p)
            .unwrap_or_else(|| panic!("No function '{}'", p.1))
            .clone();
        assert!(f.returns.len() <= 1);
        // get return type
        let ret_ty = f.returns.first().map(|r| self.type_(r));
        // setup stack frame for entry function
        self.circ.enter_fn(n.to_owned(), ret_ty.clone());
        for p in f.parameters.iter() {
            let ty = self.type_(&p.ty);
            debug!("Entry param: {}: {}", p.id.value, ty);
            let vis = self.interpret_visibility(&p.visibility);
            let r = self.circ.declare(p.id.value.clone(), &ty, true, vis);
            self.unwrap(r, &p.span);
        }
        for s in &f.statements {
            self.stmt(s);
        }
        if let Some(r) = self.circ.exit_fn() {
            match self.mode {
                Mode::Mpc(_) => {
                    let ret_term = r.unwrap_term();
                    let ret_terms = ret_term.terms();
                    self.circ
                        .cir_ctx()
                        .cs
                        .borrow_mut()
                        .outputs
                        .extend(ret_terms);
                }
                Mode::Proof => {
                    let ty = ret_ty.as_ref().unwrap();
                    let name = "return".to_owned();
                    let term = r.unwrap_term();
                    let _r = self.circ.declare(name.clone(), &ty, false, PROVER_VIS);
                    self.circ
                        .assign_with_assertions(name, term, &ty, PUBLIC_VIS);
                }
                Mode::Opt => {
                    let ret_term = r.unwrap_term();
                    let ret_terms = ret_term.terms();
                    assert!(
                        ret_terms.len() == 1,
                        "When compiling to optimize, there can only be one output"
                    );
                    let t = ret_terms.into_iter().next().unwrap();
                    match check(&t) {
                        Sort::BitVector(_) => {}
                        s => {
                            panic!("Cannot maximize output of type {}", s)
                        }
                    }
                    self.circ.cir_ctx().cs.borrow_mut().outputs.push(t);
                }
                Mode::ProofOfHighValue(v) => {
                    let ret_term = r.unwrap_term();
                    let ret_terms = ret_term.terms();
                    assert!(
                        ret_terms.len() == 1,
                        "When compiling to optimize, there can only be one output"
                    );
                    let t = ret_terms.into_iter().next().unwrap();
                    let cmp = match check(&t) {
                        Sort::BitVector(w) => {
                            term![BV_UGE; t, bv_lit(v, w)]
                        }
                        s => {
                            panic!("Cannot maximize output of type {}", s)
                        }
                    };
                    self.circ.cir_ctx().cs.borrow_mut().outputs.push(cmp);
                }
            }
        }
    }
    fn interpret_visibility(&self, visibility: &Option<ast::Visibility<'ast>>) -> Option<PartyId> {
        match visibility {
            None | Some(ast::Visibility::Public(_)) => PUBLIC_VIS.clone(),
            Some(ast::Visibility::Private(private)) => match self.mode {
                Mode::Proof | Mode::Opt | Mode::ProofOfHighValue(_) => {
                    if private.number.is_some() {
                        self.err(
                            format!(
                                "Party number found, but we're generating a {} circuit",
                                self.mode
                            ),
                            &private.span,
                        );
                    }
                    PROVER_VIS.clone()
                }
                Mode::Mpc(n_parties) => {
                    let num_str = private
                        .number
                        .as_ref()
                        .unwrap_or_else(|| self.err("No party number", &private.span));
                    let num_val =
                        u8::from_str_radix(&num_str.value[1..num_str.value.len() - 1], 10)
                            .unwrap_or_else(|e| {
                                self.err(format!("Bad party number: {}", e), &private.span)
                            });
                    if num_val <= n_parties {
                        Some(num_val - 1)
                    } else {
                        self.err(
                            format!(
                                "Party number {} greater than the number of parties ({})",
                                num_val, n_parties
                            ),
                            &private.span,
                        )
                    }
                }
            },
        }
    }
    fn cur_path(&self) -> &Path {
        self.file_stack.last().unwrap()
    }
    fn cur_dir(&self) -> PathBuf {
        let mut p = self.file_stack.last().unwrap().to_path_buf();
        p.pop();
        p
    }
    fn deref_import(&self, s: String) -> (PathBuf, String) {
        let r = (self.cur_path().to_path_buf(), s);
        self.import_map.get(&r).cloned().unwrap_or(r)
    }

    fn const_int(&mut self, e: &ast::Expression<'ast>) -> isize {
        let i = const_int(self.expr(e));
        self.unwrap(i, e.span()).to_isize().unwrap()
    }

    fn type_(&mut self, t: &ast::Type<'ast>) -> Ty {
        fn lift<'ast>(t: &ast::BasicOrStructType<'ast>) -> ast::Type<'ast> {
            match t {
                ast::BasicOrStructType::Basic(b) => ast::Type::Basic(b.clone()),
                ast::BasicOrStructType::Struct(b) => ast::Type::Struct(b.clone()),
            }
        }
        match t {
            ast::Type::Basic(ast::BasicType::U8(_)) => Ty::Uint(8),
            ast::Type::Basic(ast::BasicType::U16(_)) => Ty::Uint(16),
            ast::Type::Basic(ast::BasicType::U32(_)) => Ty::Uint(32),
            ast::Type::Basic(ast::BasicType::Boolean(_)) => Ty::Bool,
            ast::Type::Basic(ast::BasicType::Field(_)) => Ty::Field,
            ast::Type::Array(a) => {
                let b = self.type_(&lift(&a.ty));
                a.dimensions
                    .iter()
                    .map(|d| self.const_int(d))
                    .fold(b, |b, d| Ty::Array(d as usize, Box::new(b)))
            }
            ast::Type::Struct(s) => self.circ.get_type(&s.id.value).clone(),
        }
    }

    fn visit_files(&mut self) {
        // first, go through includes and return a toposorted visit order for remaining processing
        let files = self.visit_includes();
        let t = std::mem::take(&mut self.asts);
        for (p, f) in files.iter().map(|p| (p, t.get(p).unwrap())) {
            self.file_stack.push(p.to_owned());
            for d in f.declarations.iter() {
                match d {
                    ast::SymbolDeclaration::Import(_) => {}
                    ast::SymbolDeclaration::Constant(c) => {
                        // constant defns: need to add global defs (in self.circ?) and use to eval
                    }
                    ast::SymbolDeclaration::Struct(s) => {
                        // XXX(rsw) need to define structs in a way that handles generics!
                        /*
                        let ty = Ty::Struct(
                            s.id.value.clone(),
                            s.fields
                                .clone()
                                .iter()
                                .map(|f| (f.id.value.clone(), self.type_(&f.ty)))
                                .collect(),
                        );
                        debug!("struct {}", s.id.value);
                        self.circ.def_type(&s.id.value, ty);
                        */
                    }
                    ast::SymbolDeclaration::Function(f) => {
                        debug!("fn {} in {}", f.id.value, self.cur_path().display());
                        self.functions.insert(
                            (self.cur_path().to_owned(), f.id.value.clone()),
                            f.clone(),
                        );
                    }
                }
            }
            self.file_stack.pop();
        }
        self.asts = t;
    }

    fn visit_includes(&mut self) -> Vec<PathBuf> {
        use petgraph::graph::{DiGraph, NodeIndex, DefaultIx};
        use petgraph::algo::toposort;
        use petgraph::dot::{Dot, Config};
        let asts = std::mem::take(&mut self.asts);

        // we use the graph to toposort the includes and the map to go from PathBuf to NodeIdx
        let mut ig = DiGraph::<PathBuf, ()>::with_capacity(asts.len(), asts.len());
        let mut gn = HashMap::<PathBuf, NodeIndex<DefaultIx>>::with_capacity(asts.len());

        for (p, f) in asts.iter() {
            self.file_stack.push(p.to_owned());
            if !gn.contains_key(p) {
                gn.insert(p.to_owned(), ig.add_node(p.to_owned()));
            }

            for d in f.declarations.iter() {
                if let ast::SymbolDeclaration::Import(i) = d {
                    let (src_path, src_names, dst_names) = match i {
                        ast::ImportDirective::Main(m) => (
                            m.source.value.clone(),
                            vec!["main".to_owned()],
                            vec![m.alias
                                .as_ref()
                                .map(|a| a.value.clone())
                                .unwrap_or_else(|| {
                                    PathBuf::from(m.source.value.clone())
                                        .file_stem()
                                        .unwrap_or_else(|| panic!("Bad import: {}", m.source.value))
                                        .to_string_lossy()
                                        .to_string()
                                })],
                        ),
                        ast::ImportDirective::From(m) => (
                            m.source.value.clone(),
                            m.symbols.iter().map(|s| s.id.value.clone()).collect(),
                            m.symbols.iter().map(|s| {
                                s.alias
                                    .as_ref()
                                    .map(|a| a.value.clone())
                                    .unwrap_or_else(|| s.id.value.clone())
                            }).collect(),
                        ),
                    };
                    assert!(src_names.len() > 0);
                    let abs_src_path = self.stdlib.canonicalize(&self.cur_dir(), src_path.as_str());
                    debug!(
                        "Import of {:?} from {} as {:?}",
                        src_names,
                        abs_src_path.display(),
                        dst_names
                    );
                    src_names.into_iter().zip(dst_names.into_iter())
                        .map(|(sn, dn)| {
                            self.import_map.insert(
                                (self.cur_path().to_path_buf(), dn),
                                (abs_src_path.clone(), sn),
                            );
                        });

                    // add included -> includer edge for later toposort
                    if !gn.contains_key(&abs_src_path) {
                        gn.insert(abs_src_path.clone(), ig.add_node(abs_src_path.clone()));
                    }
                    ig.add_edge(*gn.get(&abs_src_path).unwrap(), *gn.get(p).unwrap(), ());
                }
            }

            self.file_stack.pop();
        }
        self.asts = asts;

        toposort(&ig, None)
            .unwrap_or_else(|e| {
                panic!("Import graph is cyclic!: {:?}\n{:?}\n",
                       e,
                       Dot::with_config(&ig, &[Config::EdgeNoLabel]))
            })
            .iter().map(|idx| std::mem::take(ig.node_weight_mut(*idx).unwrap())).collect()
    }
}
