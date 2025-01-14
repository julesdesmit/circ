//! Symbolic ZoKrates terms
use std::collections::{BTreeMap, HashMap};
use std::fmt::{self, Display, Formatter};
use std::sync::Arc;

use lazy_static::lazy_static;
use rug::Integer;

use crate::circify::{CirCtx, Embeddable};
use crate::ir::term::*;

lazy_static! {
    // TODO: handle this better
    /// The modulus for ZoKrates.
    pub static ref ZOKRATES_MODULUS: Integer = Integer::from_str_radix(
        "52435875175126190479447740508185965837690552500527637822603658699938581184513",
        10
    )
    .unwrap();
    /// The modulus for ZoKrates, as an ARC
    pub static ref ZOKRATES_MODULUS_ARC: Arc<Integer> = Arc::new(ZOKRATES_MODULUS.clone());
    /// The modulus for ZoKrates, as an IR sort
    pub static ref ZOK_FIELD_SORT: Sort = Sort::Field(ZOKRATES_MODULUS_ARC.clone());
}

#[derive(Clone, PartialEq, Eq)]
pub enum Ty {
    Uint(usize),
    Bool,
    Field,
    Struct(String, FieldList<Ty>),
    Array(usize, Box<Ty>),
}

pub use field_list::FieldList;

/// This module contains [FieldList].
///
/// It gets its own module so that its member can be private.
mod field_list {

    #[derive(Clone, PartialEq, Eq)]
    pub struct FieldList<T> {
        // must be kept in sorted order
        list: Vec<(String, T)>,
    }

    impl<T> FieldList<T> {
        pub fn new(mut list: Vec<(String, T)>) -> Self {
            list.sort_by_cached_key(|p| p.0.clone());
            FieldList { list }
        }
        pub fn search(&self, key: &str) -> Option<(usize, &T)> {
            let idx = self
                .list
                .binary_search_by_key(&key, |p| p.0.as_str())
                .ok()?;
            Some((idx, &self.list[idx].1))
        }
        pub fn get(&self, idx: usize) -> (&str, &T) {
            (&self.list[idx].0, &self.list[idx].1)
        }
        pub fn fields(&self) -> impl Iterator<Item = &(String, T)> {
            self.list.iter()
        }
    }
}

impl Display for Ty {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Ty::Bool => write!(f, "bool"),
            Ty::Uint(w) => write!(f, "u{}", w),
            Ty::Field => write!(f, "field"),
            Ty::Struct(n, fields) => {
                let mut o = f.debug_struct(n);
                for (f_name, f_ty) in fields.fields() {
                    o.field(f_name, f_ty);
                }
                o.finish()
            }
            Ty::Array(n, b) => write!(f, "{}[{}]", b, n),
        }
    }
}

impl fmt::Debug for Ty {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl Ty {
    fn sort(&self) -> Sort {
        match self {
            Self::Bool => Sort::Bool,
            Self::Uint(w) => Sort::BitVector(*w),
            Self::Field => ZOK_FIELD_SORT.clone(),
            Self::Array(n, b) => {
                Sort::Array(Box::new(ZOK_FIELD_SORT.clone()), Box::new(b.sort()), *n)
            }
            Self::Struct(_name, fs) => {
                Sort::Tuple(fs.fields().map(|(_f_name, f_ty)| f_ty.sort()).collect())
            }
        }
    }
    fn default_ir_term(&self) -> Term {
        self.sort().default_term()
    }
    fn default(&self) -> T {
        T {
            term: self.default_ir_term(),
            ty: self.clone(),
        }
    }
    /// Creates a new structure type, sorting the keys.
    pub fn new_struct<I: IntoIterator<Item = (String, Ty)>>(name: String, fields: I) -> Self {
        Self::Struct(name, FieldList::new(fields.into_iter().collect()))
    }
}

#[derive(Clone, Debug)]
pub struct T {
    pub ty: Ty,
    pub term: Term,
}

impl T {
    pub fn new(ty: Ty, term: Term) -> Self {
        Self { ty, term }
    }
    pub fn type_(&self) -> &Ty {
        &self.ty
    }
    /// Get all IR terms inside this value, as a list.
    pub fn terms(&self) -> Vec<Term> {
        let mut output: Vec<Term> = Vec::new();
        fn terms_tail(term: &Term, output: &mut Vec<Term>) {
            match check(term) {
                Sort::Bool | Sort::BitVector(_) | Sort::Field(_) => output.push(term.clone()),
                Sort::Array(_k, _v, size) => {
                    for i in 0..size {
                        terms_tail(&term![Op::Select; term.clone(), pf_lit_ir(i)], output)
                    }
                }
                Sort::Tuple(sorts) => {
                    for i in 0..sorts.len() {
                        terms_tail(&term![Op::Field(i); term.clone()], output)
                    }
                }
                s => unreachable!("Unreachable IR sort {} in ZoK", s),
            }
        }
        terms_tail(&self.term, &mut output);
        output
    }
    fn unwrap_array_ir(self) -> Result<Vec<Term>, String> {
        match &self.ty {
            Ty::Array(size, _sort) => Ok((0..*size)
                .map(|i| term![Op::Select; self.term.clone(), pf_lit_ir(i)])
                .collect()),
            s => Err(format!("Not an array: {}", s)),
        }
    }
    pub fn unwrap_array(self) -> Result<Vec<T>, String> {
        match &self.ty {
            Ty::Array(_size, sort) => {
                let sort = (**sort).clone();
                Ok(self
                    .unwrap_array_ir()?
                    .into_iter()
                    .map(|t| T::new(sort.clone(), t))
                    .collect())
            }
            s => Err(format!("Not an array: {}", s)),
        }
    }
    pub fn new_array(v: Vec<T>) -> Result<T, String> {
        array(v)
    }
    pub fn new_struct(name: String, fields: Vec<(String, T)>) -> T {
        let (field_tys, ir_terms): (Vec<_>, Vec<_>) = fields
            .into_iter()
            .map(|(name, t)| ((name.clone(), t.ty), (name, t.term)))
            .unzip();
        let field_ty_list = FieldList::new(field_tys);
        let ir_term = term(Op::Tuple, {
            let with_indices: BTreeMap<usize, Term> = ir_terms
                .into_iter()
                .map(|(name, t)| (field_ty_list.search(&name).unwrap().0, t))
                .collect();
            with_indices.into_iter().map(|(_i, t)| t).collect()
        });
        T::new(Ty::Struct(name, field_ty_list), ir_term)
    }
}

impl Display for T {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.term)
    }
}

fn wrap_bin_op(
    name: &str,
    fu: Option<fn(Term, Term) -> Term>,
    ff: Option<fn(Term, Term) -> Term>,
    fb: Option<fn(Term, Term) -> Term>,
    a: T,
    b: T,
) -> Result<T, String> {
    match (&a.ty, &b.ty, fu, ff, fb) {
        (Ty::Uint(na), Ty::Uint(nb), Some(fu), _, _) if na == nb => {
            Ok(T::new(Ty::Uint(*na), fu(a.term.clone(), b.term.clone())))
        }
        (Ty::Bool, Ty::Bool, _, _, Some(fb)) => {
            Ok(T::new(Ty::Bool, fb(a.term.clone(), b.term.clone())))
        }
        (Ty::Field, Ty::Field, _, Some(ff), _) => {
            Ok(T::new(Ty::Field, ff(a.term.clone(), b.term.clone())))
        }
        (x, y, _, _, _) => Err(format!("Cannot perform op '{}' on {} and {}", name, x, y)),
    }
}

fn wrap_bin_pred(
    name: &str,
    fu: Option<fn(Term, Term) -> Term>,
    ff: Option<fn(Term, Term) -> Term>,
    fb: Option<fn(Term, Term) -> Term>,
    a: T,
    b: T,
) -> Result<T, String> {
    match (&a.ty, &b.ty, fu, ff, fb) {
        (Ty::Uint(na), Ty::Uint(nb), Some(fu), _, _) if na == nb => {
            Ok(T::new(Ty::Bool, fu(a.term.clone(), b.term.clone())))
        }
        (Ty::Bool, Ty::Bool, _, _, Some(fb)) => {
            Ok(T::new(Ty::Bool, fb(a.term.clone(), b.term.clone())))
        }
        (Ty::Field, Ty::Field, _, Some(ff), _) => {
            Ok(T::new(Ty::Bool, ff(a.term.clone(), b.term.clone())))
        }
        (x, y, _, _, _) => Err(format!("Cannot perform op '{}' on {} and {}", name, x, y)),
    }
}

fn add_uint(a: Term, b: Term) -> Term {
    term![Op::BvNaryOp(BvNaryOp::Add); a, b]
}

fn add_field(a: Term, b: Term) -> Term {
    term![Op::PfNaryOp(PfNaryOp::Add); a, b]
}

pub fn add(a: T, b: T) -> Result<T, String> {
    wrap_bin_op("+", Some(add_uint), Some(add_field), None, a, b)
}

fn sub_uint(a: Term, b: Term) -> Term {
    term![Op::BvBinOp(BvBinOp::Sub); a, b]
}

fn sub_field(a: Term, b: Term) -> Term {
    term![Op::PfNaryOp(PfNaryOp::Add); a, term![Op::PfUnOp(PfUnOp::Neg); b]]
}

pub fn sub(a: T, b: T) -> Result<T, String> {
    wrap_bin_op("-", Some(sub_uint), Some(sub_field), None, a, b)
}

fn mul_uint(a: Term, b: Term) -> Term {
    term![Op::BvNaryOp(BvNaryOp::Mul); a, b]
}

fn mul_field(a: Term, b: Term) -> Term {
    term![Op::PfNaryOp(PfNaryOp::Mul); a, b]
}

pub fn mul(a: T, b: T) -> Result<T, String> {
    wrap_bin_op("*", Some(mul_uint), Some(mul_field), None, a, b)
}

fn div_uint(a: Term, b: Term) -> Term {
    term![Op::BvBinOp(BvBinOp::Udiv); a, b]
}

fn div_field(a: Term, b: Term) -> Term {
    term![Op::PfNaryOp(PfNaryOp::Mul); a, term![Op::PfUnOp(PfUnOp::Recip); b]]
}

pub fn div(a: T, b: T) -> Result<T, String> {
    wrap_bin_op("/", Some(div_uint), Some(div_field), None, a, b)
}

fn rem_uint(a: Term, b: Term) -> Term {
    term![Op::BvBinOp(BvBinOp::Urem); a, b]
}

pub fn rem(a: T, b: T) -> Result<T, String> {
    wrap_bin_op("%", Some(rem_uint), None, None, a, b)
}

fn bitand_uint(a: Term, b: Term) -> Term {
    term![Op::BvNaryOp(BvNaryOp::And); a, b]
}

pub fn bitand(a: T, b: T) -> Result<T, String> {
    wrap_bin_op("&", Some(bitand_uint), None, None, a, b)
}

fn bitor_uint(a: Term, b: Term) -> Term {
    term![Op::BvNaryOp(BvNaryOp::Or); a, b]
}

pub fn bitor(a: T, b: T) -> Result<T, String> {
    wrap_bin_op("|", Some(bitor_uint), None, None, a, b)
}

fn bitxor_uint(a: Term, b: Term) -> Term {
    term![Op::BvNaryOp(BvNaryOp::Xor); a, b]
}

pub fn bitxor(a: T, b: T) -> Result<T, String> {
    wrap_bin_op("^", Some(bitxor_uint), None, None, a, b)
}

fn or_bool(a: Term, b: Term) -> Term {
    term![Op::BoolNaryOp(BoolNaryOp::Or); a, b]
}

pub fn or(a: T, b: T) -> Result<T, String> {
    wrap_bin_op("||", None, None, Some(or_bool), a, b)
}

fn and_bool(a: Term, b: Term) -> Term {
    term![Op::BoolNaryOp(BoolNaryOp::And); a, b]
}

pub fn and(a: T, b: T) -> Result<T, String> {
    wrap_bin_op("&&", None, None, Some(and_bool), a, b)
}

fn eq_base(a: Term, b: Term) -> Term {
    term![Op::Eq; a, b]
}

pub fn eq(a: T, b: T) -> Result<T, String> {
    wrap_bin_pred("==", Some(eq_base), Some(eq_base), Some(eq_base), a, b)
}

fn neq_base(a: Term, b: Term) -> Term {
    term![Op::Not; term![Op::Eq; a, b]]
}

pub fn neq(a: T, b: T) -> Result<T, String> {
    wrap_bin_pred("!=", Some(neq_base), Some(neq_base), Some(neq_base), a, b)
}

fn ult_uint(a: Term, b: Term) -> Term {
    term![Op::BvBinPred(BvBinPred::Ult); a, b]
}

pub fn ult(a: T, b: T) -> Result<T, String> {
    wrap_bin_pred("<", Some(ult_uint), None, None, a, b)
}

fn ule_uint(a: Term, b: Term) -> Term {
    term![Op::BvBinPred(BvBinPred::Ule); a, b]
}

pub fn ule(a: T, b: T) -> Result<T, String> {
    wrap_bin_pred("<=", Some(ule_uint), None, None, a, b)
}

fn ugt_uint(a: Term, b: Term) -> Term {
    term![Op::BvBinPred(BvBinPred::Ugt); a, b]
}

pub fn ugt(a: T, b: T) -> Result<T, String> {
    wrap_bin_pred(">", Some(ugt_uint), None, None, a, b)
}

fn uge_uint(a: Term, b: Term) -> Term {
    term![Op::BvBinPred(BvBinPred::Uge); a, b]
}

pub fn uge(a: T, b: T) -> Result<T, String> {
    wrap_bin_pred(">=", Some(uge_uint), None, None, a, b)
}

fn wrap_un_op(
    name: &str,
    fu: Option<fn(Term) -> Term>,
    ff: Option<fn(Term) -> Term>,
    fb: Option<fn(Term) -> Term>,
    a: T,
) -> Result<T, String> {
    match (&a.ty, fu, ff, fb) {
        (Ty::Uint(_), Some(fu), _, _) => Ok(T::new(a.ty.clone(), fu(a.term.clone()))),
        (Ty::Bool, _, _, Some(fb)) => Ok(T::new(Ty::Bool, fb(a.term.clone()))),
        (Ty::Field, _, Some(ff), _) => Ok(T::new(Ty::Field, ff(a.term.clone()))),
        (x, _, _, _) => Err(format!("Cannot perform op '{}' on {}", name, x)),
    }
}

fn neg_field(a: Term) -> Term {
    term![Op::PfUnOp(PfUnOp::Neg); a]
}

fn neg_uint(a: Term) -> Term {
    term![Op::BvUnOp(BvUnOp::Neg); a]
}

#[allow(dead_code)]
// Missing from ZoKrates.
pub fn neg(a: T) -> Result<T, String> {
    wrap_un_op("unary-", Some(neg_uint), Some(neg_field), None, a)
}

fn not_bool(a: Term) -> Term {
    term![Op::Not; a]
}

fn not_uint(a: Term) -> Term {
    term![Op::BvUnOp(BvUnOp::Not); a]
}

pub fn not(a: T) -> Result<T, String> {
    wrap_un_op("!", Some(not_uint), None, Some(not_bool), a)
}

pub fn const_int(a: T) -> Result<Integer, String> {
    match &a.term.op {
        Op::Const(Value::Field(f)) => Some(f.i().clone()),
        Op::Const(Value::BitVector(f)) => Some(f.uint().clone()),
        _ => None,
    }
    .ok_or_else(|| format!("{} is not a constant integer", a))
}

pub fn bool(a: T) -> Result<Term, String> {
    match &a.ty {
        Ty::Bool => Ok(a.term),
        a => Err(format!("{} is not a boolean", a)),
    }
}

fn wrap_shift(name: &str, op: BvBinOp, a: T, b: T) -> Result<T, String> {
    let bc = const_int(b)?;
    match &a.ty {
        &Ty::Uint(na) => Ok(T::new(a.ty, term![Op::BvBinOp(op); a.term, bv_lit(bc, na)])),
        x => Err(format!("Cannot perform op '{}' on {} and {}", name, x, bc)),
    }
}

pub fn shl(a: T, b: T) -> Result<T, String> {
    wrap_shift("<<", BvBinOp::Shl, a, b)
}

pub fn shr(a: T, b: T) -> Result<T, String> {
    wrap_shift(">>", BvBinOp::Lshr, a, b)
}

fn ite(c: Term, a: T, b: T) -> Result<T, String> {
    if &a.ty != &b.ty {
        Err(format!("Cannot perform ITE on {} and {}", a, b))
    } else {
        Ok(T::new(a.ty.clone(), term![Op::Ite; c, a.term, b.term]))
    }
}

pub fn cond(c: T, a: T, b: T) -> Result<T, String> {
    ite(bool(c)?, a, b)
}

pub fn pf_lit_ir<I>(i: I) -> Term
where
    Integer: From<I>,
{
    leaf_term(Op::Const(Value::Field(FieldElem::new(
        Integer::from(i),
        ZOKRATES_MODULUS_ARC.clone(),
    ))))
}

pub fn field_lit<I>(i: I) -> T
where
    Integer: From<I>,
{
    T::new(Ty::Field, pf_lit_ir(i))
}

pub fn z_bool_lit(v: bool) -> T {
    T::new(Ty::Bool, leaf_term(Op::Const(Value::Bool(v))))
}

pub fn uint_lit<I>(v: I, bits: usize) -> T
where
    Integer: From<I>,
{
    T::new(Ty::Uint(bits), bv_lit(v, bits))
}

pub fn slice(arr: T, start: Option<usize>, end: Option<usize>) -> Result<T, String> {
    match &arr.ty {
        Ty::Array(size, _) => {
            let start = start.unwrap_or(0);
            let end = end.unwrap_or(*size);
            array(arr.unwrap_array()?.drain(start..end))
        }
        a => Err(format!("Cannot slice {}", a)),
    }
}

pub fn field_select(struct_: &T, field: &str) -> Result<T, String> {
    match &struct_.ty {
        Ty::Struct(_, map) => {
            if let Some((idx, ty)) = map.search(field) {
                Ok(T::new(
                    ty.clone(),
                    term![Op::Field(idx); struct_.term.clone()],
                ))
            } else {
                Err(format!("No field '{}'", field))
            }
        }
        a => Err(format!("{} is not a struct", a)),
    }
}

pub fn field_store(struct_: T, field: &str, val: T) -> Result<T, String> {
    match &struct_.ty {
        Ty::Struct(_, map) => {
            if let Some((idx, ty)) = map.search(field) {
                if ty == &val.ty {
                    Ok(T::new(
                        struct_.ty.clone(),
                        term![Op::Update(idx); struct_.term.clone(), val.term],
                    ))
                } else {
                    Err(format!(
                        "term {} assigned to field {} of type {}",
                        val,
                        field,
                        map.get(idx).1
                    ))
                }
            } else {
                Err(format!("No field '{}'", field))
            }
        }
        a => Err(format!("{} is not a struct", a)),
    }
}

pub fn array_select(array: T, idx: T) -> Result<T, String> {
    match (array.ty, idx.ty) {
        (Ty::Array(_size, elem_ty), Ty::Field) => {
            Ok(T::new(*elem_ty, term![Op::Select; array.term, idx.term]))
        }
        (a, b) => Err(format!("Cannot index {} by {}", b, a)),
    }
}

pub fn array_store(array: T, idx: T, val: T) -> Result<T, String> {
    match (&array.ty, idx.ty) {
        (Ty::Array(_, _), Ty::Field) => Ok(T::new(
            array.ty,
            term![Op::Store; array.term, idx.term, val.term],
        )),
        (a, b) => Err(format!("Cannot index {} by {}", b, a)),
    }
}

fn ir_array<I: IntoIterator<Item = Term>>(sort: Sort, elems: I) -> Term {
    make_array(ZOK_FIELD_SORT.clone(), sort, elems.into_iter().collect())
}

pub fn array<I: IntoIterator<Item = T>>(elems: I) -> Result<T, String> {
    let v: Vec<T> = elems.into_iter().collect();
    if let Some(e) = v.first() {
        let ty = e.type_();
        if v.iter().skip(1).any(|a| a.type_() != ty) {
            Err(format!("Inconsistent types in array"))
        } else {
            let sort = check(&e.term);
            Ok(T::new(
                Ty::Array(v.len(), Box::new(ty.clone())),
                ir_array(sort, v.into_iter().map(|t| t.term)),
            ))
        }
    } else {
        Err(format!("Empty array"))
    }
}

pub fn uint_to_bits(u: T) -> Result<T, String> {
    match &u.ty {
        Ty::Uint(n) => Ok(T::new(
            Ty::Array(*n, Box::new(Ty::Bool)),
            ir_array(
                Sort::Bool,
                (0..*n).map(|i| term![Op::BvBit(i); u.term.clone()]),
            ),
        )),
        u => Err(format!("Cannot do uint-to-bits on {}", u)),
    }
}

pub fn uint_from_bits(u: T) -> Result<T, String> {
    match &u.ty {
        Ty::Array(bits, elem_ty) if &**elem_ty == &Ty::Bool => match bits {
            8 | 16 | 32 => Ok(T::new(
                Ty::Uint(*bits),
                term(
                    Op::BvConcat,
                    u.unwrap_array_ir()?
                        .into_iter()
                        .map(|z: Term| -> Term { term![Op::BoolToBv; z] })
                        .collect(),
                ),
            )),
            l => Err(format!("Cannot do uint-from-bits on len {} array", l,)),
        },
        u => Err(format!("Cannot do uint-from-bits on {}", u)),
    }
}

pub fn field_to_bits(f: T) -> Result<T, String> {
    match &f.ty {
        Ty::Field => uint_to_bits(T::new(Ty::Uint(254), term![Op::PfToBv(254); f.term])),
        u => Err(format!("Cannot do uint-to-bits on {}", u)),
    }
}

pub struct ZoKrates {
    values: Option<HashMap<String, Integer>>,
    modulus: Arc<Integer>,
}

fn field_name(struct_name: &str, field_name: &str) -> String {
    format!("{}.{}", struct_name, field_name)
}

fn idx_name(struct_name: &str, idx: usize) -> String {
    format!("{}.{}", struct_name, idx)
}

impl ZoKrates {
    pub fn new(values: Option<HashMap<String, Integer>>) -> Self {
        Self {
            values,
            modulus: ZOKRATES_MODULUS_ARC.clone(),
        }
    }
}

impl Embeddable for ZoKrates {
    type T = T;
    type Ty = Ty;
    fn declare(
        &self,
        ctx: &mut CirCtx,
        ty: &Self::Ty,
        raw_name: String,
        user_name: Option<String>,
        visibility: Option<PartyId>,
    ) -> Self::T {
        let get_int_val = || -> Integer {
            self.values
                .as_ref()
                .and_then(|vs| {
                    user_name
                        .as_ref()
                        .and_then(|n| vs.get(n))
                        .or_else(|| vs.get(&raw_name))
                })
                .cloned()
                .unwrap_or_else(|| Integer::from(0))
        };
        match ty {
            Ty::Bool => T::new(
                Ty::Bool,
                ctx.cs.borrow_mut().new_var(
                    &raw_name,
                    Sort::Bool,
                    || Value::Bool(get_int_val() != 0),
                    visibility,
                ),
            ),
            Ty::Field => T::new(
                Ty::Field,
                ctx.cs.borrow_mut().new_var(
                    &raw_name,
                    Sort::Field(self.modulus.clone()),
                    || Value::Field(FieldElem::new(get_int_val(), self.modulus.clone())),
                    visibility,
                ),
            ),
            Ty::Uint(w) => T::new(
                Ty::Uint(*w),
                ctx.cs.borrow_mut().new_var(
                    &raw_name,
                    Sort::BitVector(*w),
                    || Value::BitVector(BitVector::new(get_int_val(), *w)),
                    visibility,
                ),
            ),
            Ty::Array(n, ty) => array((0..*n).map(|i| {
                self.declare(
                    ctx,
                    &*ty,
                    idx_name(&raw_name, i),
                    user_name.as_ref().map(|u| idx_name(u, i)),
                    visibility.clone(),
                )
            }))
            .unwrap(),
            Ty::Struct(n, fs) => T::new_struct(
                n.clone(),
                fs.fields()
                    .map(|(f_name, f_ty)| {
                        (
                            f_name.clone(),
                            self.declare(
                                ctx,
                                f_ty,
                                field_name(&raw_name, f_name),
                                user_name.as_ref().map(|u| field_name(u, f_name)),
                                visibility.clone(),
                            ),
                        )
                    })
                    .collect(),
            ),
        }
    }
    fn ite(&self, _ctx: &mut CirCtx, cond: Term, t: Self::T, f: Self::T) -> Self::T {
        ite(cond, t, f).unwrap()
    }
    fn assign(
        &self,
        ctx: &mut CirCtx,
        ty: &Self::Ty,
        name: String,
        t: Self::T,
        visibility: Option<PartyId>,
    ) -> Self::T {
        assert!(t.type_() == ty);
        T::new(t.ty, ctx.cs.borrow_mut().assign(&name, t.term, visibility))
    }
    fn values(&self) -> bool {
        self.values.is_some()
    }

    fn type_of(&self, term: &Self::T) -> Self::Ty {
        term.type_().clone()
    }

    fn initialize_return(&self, ty: &Self::Ty, _ssa_name: &String) -> Self::T {
        ty.default()
    }
}
