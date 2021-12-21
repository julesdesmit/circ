//! Generic parameter inference


use crate::ir::term::{bv_lit, leaf_term, term, Op, Sort, Term};
use super::super::{ZGen, span_to_string};
use super::super::term::{Ty, T, cond, const_int};

use rand::{distributions::Alphanumeric, Rng};
use std::collections::HashMap;
use zokrates_pest_ast as ast;

pub(in super::super) struct ZGenericInf<'ast, 'gen> {
    zgen: &'gen ZGen<'ast>,
    fdef: &'gen ast::FunctionDefinition<'ast>,
    gens: &'gen [ast::IdentifierExpression<'ast>],
    sfx: String,
    terms: Vec<Term>,
}

impl<'ast, 'gen> ZGenericInf<'ast, 'gen> {
    pub fn new(zgen: &'gen ZGen<'ast>, fdef: &'gen ast::FunctionDefinition<'ast>) -> Self {
        let gens = fdef.generics.as_ref();
        let sfx = Self::make_sfx(
            (&mut rand::thread_rng())
                .sample_iter(Alphanumeric)
                .map(char::from)
                .take(8)
                .collect(),
            &fdef.id.value,
        );
        Self {
            zgen,
            fdef,
            gens,
            sfx,
            terms: Vec::new(),
        }
    }

    fn make_sfx(mut base: String, sfx: &str) -> String {
        base.push('_');
        base.push_str(sfx);
        base
    }

    fn is_generic_var(&self, var: &str) -> bool {
        self.gens.iter().any(|id| &id.value == var)
    }

    fn add_constraint(&mut self, lhs: Term, rhs: Term) {
        self.terms.push(term![Op::Eq; lhs, rhs]);
    }

    fn make_varname(id: &str, sfx: &str) -> Term {
        let mut tmp = String::from(id);
        tmp.push('_');
        tmp.push_str(sfx);
        term![Op::Var(tmp, Sort::BitVector(32))]
    }
    
    pub fn unify_generic(
        &mut self,
        call: &ast::CallAccess<'ast>,
        rty: Ty,
        args: &Vec<T>,
    ) -> Result<HashMap<String, T>, String> {
        use ast::ConstantGenericValue as CGV;

        // 1. build up the already-known generics
        if let Some(eg) = call.explicit_generics.as_ref() {
            for (cgv, id) in eg.values.iter().zip(self.fdef.generics.iter()) {
                if let Some(v) = match cgv {
                    CGV::Underscore(_) => None,
                    CGV::Value(v) => Some(self.zgen.literal_(v)),
                    CGV::Identifier(i) => Some(self.zgen.const_identifier_(i)),
                } {
                    let var = Self::make_varname(&id.value, &self.sfx);
                    let val = match v? {
                        T::Uint(32, val) => Ok(val),
                        v => Err(format!("ZGenericInf: ConstantGenericValue for {} had type {}, expected u32", &id.value, v.type_())),
                    }?;
                    self.add_constraint(var, val);
                }
            }
        }

        // 2. for each argument, update the const generic values
        for (pty, arg) in self.fdef.parameters.iter().map(|p| &p.ty).zip(args.iter()) {
            let aty = arg.type_();
            self.fdef_gen_ty(aty, pty)?;
        }

        // 3. unify the return type
        if let Some(ret) = self.fdef.returns.first() {
            self.fdef_gen_ty(rty, ret)?;
        } else if rty != Ty::Bool {
            Err(format!("Function {} expected implicit Bool ret, but got {}", &self.fdef.id.value, rty))?;
        }

        // 4. run the solver on the term stack
        // XXX(TODO)

        // 5. extract the assignments from the solver result
        Ok(HashMap::new())
    }

    fn fdef_gen_ty(
        &mut self,
        arg_ty: Ty,
        def_ty: &ast::Type<'ast>,
    ) -> Result<(), String> {
        use ast::Type as TT;
        match def_ty {
            TT::Basic(dty_b) => self.fdef_gen_ty_basic(arg_ty, dty_b),
            TT::Array(dty_a) => self.fdef_gen_ty_array(arg_ty, dty_a),
            TT::Struct(dty_s) => self.fdef_gen_ty_struct(arg_ty, dty_s),
        }
    }

    fn fdef_gen_ty_basic(
        &self,
        arg_ty: Ty,
        bas_ty: &ast::BasicType<'ast>,
    ) -> Result<(), String> {
        if arg_ty != self.zgen.type_(&ast::Type::Basic(bas_ty.clone())) {
            Err(format!("Type mismatch unifying generics: got {}, decl was {:?}", arg_ty, bas_ty))
        } else {
            Ok(())
        }
    }

    fn fdef_gen_ty_array(
        &mut self,
        mut arg_ty: Ty,
        def_ty: &ast::ArrayType<'ast>,
    ) -> Result<(), String> {
        if !matches!(arg_ty, Ty::Array(_, _)) {
            Err(format!("Type mismatch unifying generics: got {}, decl was Array", arg_ty))?;
        }

        // iterate through array dimensions, unifying each with fn decl
        let mut dim_off = 0;
        loop {
            match arg_ty {
                Ty::Array(arg_dim, nty) => {
                    // make sure that we expect at least one more array dim
                    if dim_off >= def_ty.dimensions.len() {
                        Err(format!(
                            "Type mismatch: got >={}-dim array, decl was {} dims",
                            dim_off,
                            def_ty.dimensions.len(),
                        ))?;
                    }

                    // unify actual dimension with dim expression
                    self.fdef_gen_ty_expr(arg_dim, &def_ty.dimensions[dim_off])?;

                    // iterate
                    dim_off += 1;
                    arg_ty = *nty;
                }
                nty => {
                    // make sure we didn't expect any more array dims!
                    if dim_off != def_ty.dimensions.len() {
                        Err(format!(
                            "Type mismatch: got {}-dim array, decl had {} dims",
                            dim_off,
                            def_ty.dimensions.len(),
                        ))?;
                    }

                    arg_ty = nty;
                    break;
                }
            };
        }

        use ast::BasicOrStructType as BoST;
        match &def_ty.ty {
            BoST::Struct(dty_s) => self.fdef_gen_ty_struct(arg_ty, dty_s),
            BoST::Basic(dty_b) => self.fdef_gen_ty_basic(arg_ty, dty_b),
        }
    }

    fn fdef_gen_ty_struct(
        &mut self,
        arg_ty: Ty,
        def_ty: &ast::StructType<'ast>,
    ) -> Result<(), String> {
        // check type and struct name
        let aty_map = match arg_ty {
            Ty::Struct(aty_n, aty_map) if &aty_n == &def_ty.id.value => Ok(aty_map),
            Ty::Struct(aty_n, _) => Err(format!("Type mismatch: got struct {}, decl was struct {}", &aty_n, &def_ty.id.value)),
            arg_ty => Err(format!("Type mismatch unifying generics: got {}, decl was Struct", arg_ty)),
        }?;

        // short-circuit if there are no generics in this struct
        let strdef = self.zgen.get_struct(&def_ty.id.value)
            .ok_or_else(|| format!("ZGenericInf: no such struct {}", &def_ty.id.value))?;

        if strdef.generics.is_empty() {
            return if def_ty.explicit_generics.is_some() {
                Err(format!(
                    "Unifying generics: got explicit generics for non-generic struct type {}:\n{}",
                    &def_ty.id.value,
                    span_to_string(&def_ty.span),
                ))
            } else {
                Ok(())
            }
        }

        // struct type in fn defn must provide explicit generics
        use ast::ConstantGenericValue as CGV;
        if def_ty.explicit_generics
            .as_ref()
            .map(|eg| eg.values.iter().any(|eg| matches!(eg, CGV::Underscore(_))))
            .unwrap_or(true)
        {
            Err(format!(
                "Cannot infer generic values for struct {} arg to function {}\nGeneric structs in fn defns must have explicit generics (in terms of fn generic vars)",
                &def_ty.id.value,
                &self.fdef.id.value,
            ))?;
        }

        // 1. set up mapping from outer explicit generics to inner explicit generics
        let new_sfx = Self::make_sfx(self.sfx.clone(), &def_ty.id.value);
        def_ty.explicit_generics.as_ref().unwrap().values.iter()
            .zip(strdef.generics.iter())
            .try_for_each::<_,Result<(),String>>(|(cgv, id)| {
                let sgid = Self::make_varname(&id.value, &new_sfx);
                let val = match cgv {
                    CGV::Underscore(_) => unreachable!(),
                    CGV::Value(le) => {
                        Self::u32_term(self.zgen.literal_(le)?)?
                    }
                    CGV::Identifier(id) => {
                        if self.is_generic_var(&id.value) {
                            Self::make_varname(&id.value, &self.sfx)
                        } else {
                            Self::u32_term(self.zgen.const_identifier_(&id)?)?
                        }
                    }
                };
                self.add_constraint(sgid, val);
                Ok(())
            })?;

        // 2. walk through struct def to generate constraints on inner explicit generics
        let old_sfx = std::mem::replace(&mut self.sfx, new_sfx);
        let old_gens = std::mem::replace(&mut self.gens, &strdef.generics[..]);
        // XXX(TODO)

        // 3. pop stack and continue
        self.gens = old_gens;
        self.sfx = old_sfx;
        Ok(())
    }

    // turn an expr into a set of terms and assert equality
    fn fdef_gen_ty_expr(
        &mut self,
        arg_dim: usize,
        def_exp: &ast::Expression<'ast>,
    ) -> Result<(), String> {
        let t = Self::u32_term(self.expr(def_exp)?)?;
        self.add_constraint(bv_lit(arg_dim, 32), t);
        Ok(())
    }

    fn expr(
        &self,
        expr: &ast::Expression<'ast>,
    ) -> Result<T, String> {
        use ast::Expression::*;
        match expr {
            Ternary(te) => {
                let cnd = self.expr(&te.first)?;
                let csq = self.expr(&te.second)?;
                let alt = self.expr(&te.third)?;
                cond(cnd, csq, alt)
            }
            Binary(be) => {
                let lhs = self.expr(&be.left)?;
                let rhs = self.expr(&be.right)?;
                let op = self.zgen.bin_op(&be.op);
                op(lhs, rhs)
            }
            Unary(ue) => {
                let exp = self.expr(&ue.expression)?;
                let op = self.zgen.unary_op(&ue.op);
                op(exp)
            }
            Identifier(id) => {
                if self.is_generic_var(&id.value) {
                    Ok(T::Uint(32, Self::make_varname(&id.value, &self.sfx)))
                } else {
                    self.zgen.const_identifier_(&id)
                }
            }
            Literal(le) => self.zgen.literal_(le),
            Postfix(_) => Err("ZGenericInf: got Postfix in array dim expr (unimpl)".into()),
            InlineArray(_) => Err("ZGenericInf: got InlineArray in array dim expr (unimpl)".into()),
            InlineStruct(_) => Err("ZGenericInf: got InlineStruct in array dim expr (unimpl)".into()),
            ArrayInitializer(_) => Err("ZGenericInf: got ArrayInitializer in array dim expr (unimpl)".into()),
        }
    }

    fn u32_term(t: T) -> Result<Term, String> {
        match t {
            T::Uint(32, t) => Ok(t),
            e => Err(format!("ZGenericInf: got {} for expr, expected T::Uint(32)", e.type_())),
        }
    }
}
    /*
        use ast::{Expression::*, ConstantGenericValue as CGV};
        match (dexp, rexp) {
            (Binary(dbin), Binary(rbin)) if dbin.op == rbin.op => {
                // XXX(unimpl) improve support for complex const expression inference?
                self.fdef_gen_ty_expr(dbin.left.as_ref(), rbin.left.as_ref(), egv, gid_map)?;
                self.fdef_gen_ty_expr(dbin.right.as_ref(), rbin.right.as_ref(), egv, gid_map)
            }
            (Identifier(did), _) if matches!(rexp, Identifier(_) | Literal(_)) => {
                if let Some(&doff) = gid_map.get(did.value.as_str()) {
                    if matches!(&egv[doff], CGV::Underscore(_)) {
                        egv[doff] = match rexp {
                            Identifier(rid) => CGV::Identifier(rid.clone()),
                            Literal(rle) => CGV::Value(rle.clone()),
                            _ => unreachable!(),
                        };
                        Ok(())
                    } else {
                        match (&egv[doff], rexp) {
                            (CGV::Identifier(did), Identifier(rid)) => self.fdef_gen_id_id(did, rid),
                            (CGV::Identifier(did), Literal(rle)) => self.fdef_gen_id_le(did, rle),
                            (CGV::Value(dle), Identifier(rid)) => self.fdef_gen_id_le(rid, dle),
                            (CGV::Value(dle), Literal(rle)) => self.fdef_gen_le_le(dle, rle),
                            _ => unreachable!(),
                        }
                    }
                } else {
                    match rexp {
                        Identifier(rid) => self.fdef_gen_id_id(did, rid),
                        Literal(rle) => self.fdef_gen_id_le(did, rle),
                        _ => unreachable!(),
                    }
                }
            }
            (Identifier(did), _) => {
                if let Some(&doff) = gid_map.get(did.value.as_str()) {
                    if matches!(&egv[doff], CGV::Underscore(_)) {
                        const_int(self.zgen.const_expr_(rexp)?)
                            .map_err(|e| ZVisitorError(format!(
                                "Inferring fn call generics: cannot constify expression {:?}: {}",
                                rexp,
                                e
                            )))
                            .and_then(|rval| match rval.to_u32() {
                                Some(rval) => {
                                    let span = rexp.span().clone();
                                    let hne = ast::HexNumberExpression::U32(
                                        ast::U32NumberExpression {
                                            value: format!("0x{:08x}", rval),
                                            span: span.clone(),
                                        }
                                    );
                                    let hle = ast::HexLiteralExpression {
                                        value: hne,
                                        span: span.clone(),
                                    };
                                    egv[doff] = CGV::Value(
                                        ast::LiteralExpression::HexLiteral(hle)
                                    );
                                    Ok(())
                                }
                                None => Err(ZVisitorError(format!(
                                    "Inferring fn call generics: got generic value {} out of u32 range",
                                    rval,
                                ))),
                            })
                    } else {
                        self.fdef_gen_expr_check(dexp, rexp)
                    }
                } else {
                    self.fdef_gen_expr_check(dexp, rexp)
                }
            }
            _ => self.fdef_gen_expr_check(dexp, rexp),
        }
    */

    /*
        // invariant: rty is LHS, therefore must have explicit generics
        let dty_egvs = &dty.explicit_generics.as_ref().unwrap().values;
        let rty_egvs = &rty.explicit_generics.as_ref().unwrap().values;
        assert_eq!(dty_egvs.len(), rty_egvs.len());

        // unify generic args to structs
        dty_egvs
            .iter()
            .zip(rty_egvs.iter())
            .try_for_each(|(dv, rv)| self.fdef_gen_ty_cgv(dv, rv, egv, gid_map))
    */

    /*

        if call
            .explicit_generics
            .as_ref()
            .map(|eg| eg.values.iter().any(|eg| matches!(eg, Underscore(_))))
            .unwrap_or_else(|| !fdef.generics.is_empty())
        {
            // step 1: construct mutable vector of constant generic values, plus a Name->Posn map
            let (gen, par, ret) = (&fdef.generics, &fdef.parameters, &fdef.returns);
            let gid_map = gen.iter().enumerate().map(|(i,v)| (v.value.as_ref(),i)).collect::<HashMap<&str,usize>>();
            let egv = {
                let (sp, eg) = (&call.span, &mut call.explicit_generics);
                &mut eg.get_or_insert_with(|| ast::ExplicitGenerics {
                    values: vec![Underscore( ast::Underscore { span: sp.clone() } ); gen.len()],
                    span: sp.clone(),
                }).values
            };
            assert_eq!(egv.len(), gen.len());

            // step 2: for each function argument unify type and update cgvs
            let mut zty = ZExpressionTyper::new(self);
            for (exp, pty) in call.arguments.expressions.iter_mut().zip(par.iter().map(|p| &p.ty)) {
                let aty = self.type_expression(exp, &mut zty)?;
                if let Some(aty) = aty {
                    self.fdef_gen_ty(pty, &aty, egv, &gid_map)?;
                } else {
                    debug!("Could not type expression {:?} while inferring generic fn call", exp);
                }
            }

            // step 3: optionally unify return type and update cgvs
            if let Some(rty) = rty {
                // XXX(unimpl) multi-return statements not supported
                self.fdef_gen_ty(&ret[0], rty, egv, &gid_map)?;
            }

            // step 4: if we've determined the explicit generic values, write them back to the call
            // otherwise return an error
            if egv.iter().any(|eg| matches!(eg, Underscore(_))) {
                return Err(ZVisitorError(format!(
                    "ZStatementWalker: failed to infer generics in fn call:\n{}\n\n{:?}\n{:?}",
                    span_to_string(&call.span),
                    egv,
                    gid_map,
                )));
            }
        }

        // rewrite return type and return it
        // XXX(perf) do this without so much cloning? probably changes ZExpressionRewriter
        let egv = call
            .explicit_generics
            .as_ref()
            .map(|eg| {
                {
                    eg.values.iter().map(|cgv| match cgv {
                        Underscore(_) => unreachable!(),
                        Value(l) => ast::Expression::Literal(l.clone()),
                        Identifier(i) => ast::Expression::Identifier(i.clone()),
                    })
                }
                .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let gvmap = fdef
            .generics
            .iter()
            .map(|ie| ie.value.clone())
            .zip(egv.into_iter())
            .collect::<HashMap<String, ast::Expression<'ast>>>();

        let mut ret_rewriter = ZExpressionRewriter::new(gvmap);
        let mut ret_ty = fdef.returns.first().unwrap().clone();
        ret_rewriter.visit_type(&mut ret_ty).map(|_| ret_ty)

        Ok(HashMap::new());
    */


    /*
    fn fdef_gen_ty_cgv(
        &self,
        dv: &ast::ConstantGenericValue<'ast>,   // declared type (from fn defn)
        rv: &ast::ConstantGenericValue<'ast>,   // required type (from call context)
        egv: &mut Vec<ast::ConstantGenericValue<'ast>>,
        gid_map: &HashMap<&str, usize>,
    ) -> ZVisitorResult {
        use ast::ConstantGenericValue::*;
        match (dv, rv) {
            (Identifier(did), _) => {
                if let Some(&doff) = gid_map.get(did.value.as_str()) {
                    if matches!(&egv[doff], Underscore(_)) {
                        egv[doff] = rv.clone();
                        Ok(())
                    } else {
                        self.fdef_gen_cgv_check(&egv[doff], rv)
                    }
                } else {
                    self.fdef_gen_cgv_check(dv, rv)
                }
            }
            (dv, rv) => self.fdef_gen_cgv_check(dv, rv),
        }
    }

    fn fdef_gen_cgv_check(
        &self,
        dv: &ast::ConstantGenericValue<'ast>,
        rv: &ast::ConstantGenericValue<'ast>,
    ) -> ZVisitorResult {
        use ast::ConstantGenericValue::*;
        match (dv, rv) {
            (Underscore(_), _) | (_, Underscore(_)) => unreachable!(),
            (Value(dle), Value(rle)) => self.fdef_gen_le_le(dle, rle),
            (Identifier(did), Identifier(rid)) => self.fdef_gen_id_id(did, rid),
            (Identifier(did), Value(rle)) => self.fdef_gen_id_le(did, rle),
            (Value(dle), Identifier(rid)) => self.fdef_gen_id_le(rid, dle),
        }
    }

    fn fdef_gen_id_id(
        &self,
        did: &ast::IdentifierExpression<'ast>,
        rid: &ast::IdentifierExpression<'ast>,
    ) -> ZVisitorResult {
        // did must be either generic id in enclosing scope or const
        if self.generic_defined(did.value.as_str()) {
            if &did.value == &rid.value {
                Ok(())
            } else {
                Err(ZVisitorError(format!(
                    "Inconsistent generic args detected: wanted {}, got {}",
                    &rid.value,
                    &did.value,
                )))
            }
        } else if self.generic_defined(rid.value.as_str()) {
            // did is a const, but rid is a generic arg
            Err(ZVisitorError(format!(
                "Generic identifier {} is not identically const identifier {}",
                &rid.value,
                &did.value,
            )))
        } else {
            match (self.zgen.const_lookup_(did.value.as_str()),
                   self.zgen.const_lookup_(rid.value.as_str())) {
                (None, _) => Err(ZVisitorError(format!(
                    "Constant {} undefined when inferring generics",
                    &did.value,
                ))),
                (_, None) => Err(ZVisitorError(format!(
                    "Constant {} undefined when inferring generics",
                    &rid.value,
                ))),
                (Some(dc), Some(rc)) => {
                    let dval = const_int_ref(dc)?;
                    let rval = const_int_ref(rc)?;
                    if dval != rval {
                        Err(ZVisitorError(format!(
                            "Mismatch in struct generic: expected {}, got {}",
                            rval,
                            dval,
                        )))
                    } else {
                        Ok(())
                    }
                }
            }
        }
    }

    fn fdef_gen_id_le(
        &self,
        id: &ast::IdentifierExpression<'ast>,
        le: &ast::LiteralExpression<'ast>,
    ) -> ZVisitorResult {
        if self.generic_defined(id.value.as_str()) {
            Err(ZVisitorError(format!(
                "Inconsistent generic args detected: wanted {:?}, got local generic identifier {}",
                le,
                &id.value,
            )))
        } else if let Some(dc) = self.zgen.const_lookup_(id.value.as_str()) {
            let dval = const_int_ref(dc)?;
            let rval = const_int(self.zgen.literal_(le)?)?;
            if dval != rval {
                Err(ZVisitorError(format!(
                    "Mismatch in struct generic: expected {}, got {}",
                    rval,
                    dval,
                )))
            } else {
                Ok(())
            }
        } else {
            Err(ZVisitorError(format!(
                "Constant {} undefined when inferring generics",
                &id.value,
            )))
        }
    }

    fn fdef_gen_le_le(
        &self,
        dle: &ast::LiteralExpression<'ast>,
        rle: &ast::LiteralExpression<'ast>,
    ) -> ZVisitorResult {
        let dval = const_int(self.zgen.literal_(dle)?)?;
        let rval = const_int(self.zgen.literal_(rle)?)?;
        if dval != rval {
            Err(ZVisitorError(format!(
                "Mismatch in struct generic: expected {}, got {}",
                rval,
                dval,
            )))
        } else {
            Ok(())
        }
    }

    fn fdef_gen_expr_check(
        &self,
        dexp: &ast::Expression<'ast>,
        rexp: &ast::Expression<'ast>,
    ) -> ZVisitorResult {
        match (const_int(self.zgen.const_expr_(dexp)?), const_int(self.zgen.const_expr_(rexp)?)) {
            (Ok(dci), Ok(rci)) => {
                if dci == rci {
                    Ok(())
                } else {
                    Err(ZVisitorError(format!(
                        "Mismatch in struct generic: expected {}, got {}",
                        rci,
                        dci,
                    )))
                }
            }
            _ => Err(ZVisitorError(format!(
                "Inferring fn call generics: unsupported array dimension expr {:?}, expected {:?}",
                dexp,
                rexp,
            )))
        }
    }
    */
