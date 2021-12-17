//! Generic inference


use super::eqtype::*;
use super::{ZResult, ZVisitorError, ZVisitorResult};
use super::super::{ZGen, span_to_string};
use super::super::term::{Ty, T, const_int, const_int_ref};

use std::collections::HashMap;
use zokrates_pest_ast as ast;

pub(in super::super) struct ZGenericInf<'ast, 'gen> {
    zgen: &'gen ZGen<'ast>,
}

impl<'ast, 'gen> ZGenericInf<'ast, 'gen> {
    pub fn new(zgen: &'gen ZGen<'ast>) -> Self {
        Self { zgen }
    }
    
    fn unify_generic(
        &self,
        fdef: &ast::FunctionDefinition<'ast>,
        call: &ast::CallAccess<'ast>,
        rty: Option<&ast::Type<'ast>>,
    ) -> ZResult<HashMap<String, T>> {
        use ast::ConstantGenericValue as CGV;

        // build up the already-known generics
        let mut gens = HashMap::new();
        if let Some(eg) = call.explicit_generics.as_ref() {
            for (cgv, id) in eg.values.iter().zip(fdef.generics.iter()) {
                if let Some(v) = match cgv {
                    CGV::Underscore(_) => None,
                    CGV::Value(v) => Some(self.zgen.literal_(v)),
                    CGV::Identifier(i) => Some(self.zgen.const_identifier_(i)),
                } {
                    gens.insert(id.value.clone(), v?);
                }
            }
        }

        Ok(gens)
    }

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

    fn fdef_gen_ty(
        &self,
        dty: &ast::Type<'ast>,      // declared type (from fn defn)
        rty: &ast::Type<'ast>,      // required type (from call context)
        egv: &mut Vec<ast::ConstantGenericValue<'ast>>,
        gid_map: &HashMap<&str, usize>,
    ) -> ZVisitorResult {
        use ast::Type::*;
        match (dty, rty) {
            (Basic(dty_b), Basic(rty_b)) => eq_basic_type(dty_b, rty_b)
                .map_err(|e| ZVisitorError(format!("Inferring generic fn call: {}", e.0))),
            (Array(dty_a), Array(rty_a)) => self.fdef_gen_ty_array(dty_a, rty_a, egv, gid_map),
            (Struct(dty_s), Struct(rty_s)) => self.fdef_gen_ty_struct(dty_s, rty_s, egv, gid_map),
            _ => Err(ZVisitorError(format!(
                "Inferring generic fn call: type mismatch: expected {:?}, got {:?}",
                rty,
                dty,
            ))),
        }
    }

    fn fdef_gen_ty_array(
        &self,
        dty: &ast::ArrayType<'ast>,     // declared type (from fn defn)
        rty: &ast::ArrayType<'ast>,     // required type (from call context)
        egv: &mut Vec<ast::ConstantGenericValue<'ast>>,
        gid_map: &HashMap<&str, usize>,
    ) -> ZVisitorResult {
        // check dimensions
        if dty.dimensions.len() != rty.dimensions.len() {
            return Err(ZVisitorError(format!(
                "Inferring generic fn call: Array #dimensions mismatch: expected {}, got {}",
                rty.dimensions.len(),
                dty.dimensions.len(),
            )));
        }

        // unify the type contained in the array
        use ast::BasicOrStructType as BoST;
        match (&dty.ty, &rty.ty) {
            (BoST::Struct(dty_s), BoST::Struct(rty_s)) => self.fdef_gen_ty_struct(dty_s, rty_s, egv, gid_map),
            (BoST::Basic(dty_b), BoST::Basic(rty_b)) => eq_basic_type(dty_b, rty_b)
                .map_err(|e| ZVisitorError(format!("Inferring generic fn call: {}", e.0))),
            _ => Err(ZVisitorError(format!(
                "Inferring generic fn call: Array type mismatch: expected {:?}, got {:?}",
                &rty.ty,
                &dty.ty,
            ))),
        }?;

        // unify the dimensions
        dty.dimensions
            .iter()
            .zip(rty.dimensions.iter())
            .try_for_each(|(dexp, rexp)| self.fdef_gen_ty_expr(dexp, rexp, egv, gid_map))
    }

    fn fdef_gen_ty_struct(
        &self,
        dty: &ast::StructType<'ast>,    // declared type (from fn defn)
        rty: &ast::StructType<'ast>,    // required type (from call context)
        egv: &mut Vec<ast::ConstantGenericValue<'ast>>,
        gid_map: &HashMap<&str, usize>,
    ) -> ZVisitorResult {
        if &dty.id.value != &rty.id.value {
            return Err(ZVisitorError(format!(
                "Inferring generic in fn call: wanted struct {}, found struct {}",
                &rty.id.value,
                &dty.id.value,
            )));
        }
        // make sure struct exists and short-circuit if it's not generic
        if self.get_struct(&dty.id.value)?.generics.is_empty() {
            return if dty.explicit_generics.is_some() {
                Err(ZVisitorError(format!(
                    "Inferring generic in fn call: got explicit generics for non-generic struct type {}:\n{}",
                    &dty.id.value,
                    span_to_string(&dty.span),
                )))
            } else {
                Ok(())
            };
        }

        // declared type in fn defn must provide explicit generics
        use ast::ConstantGenericValue::*;
        if dty.explicit_generics
            .as_ref()
            .map(|eg| eg.values.iter().any(|eg| matches!(eg, Underscore(_))))
            .unwrap_or(true)
        {
            return Err(ZVisitorError(format!(
                "Cannot infer generic values for struct {}\nGeneric structs in fn defns must have explicit generics (possibly in terms of fn generics)",
                &dty.id.value,
            )));
        }

        // invariant: rty is LHS, therefore must have explicit generics
        let dty_egvs = &dty.explicit_generics.as_ref().unwrap().values;
        let rty_egvs = &rty.explicit_generics.as_ref().unwrap().values;
        assert_eq!(dty_egvs.len(), rty_egvs.len());

        // unify generic args to structs
        dty_egvs
            .iter()
            .zip(rty_egvs.iter())
            .try_for_each(|(dv, rv)| self.fdef_gen_ty_cgv(dv, rv, egv, gid_map))
    }

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

    fn fdef_gen_ty_expr(
        &self,
        dexp: &ast::Expression<'ast>,       // declared type (from fn defn)
        rexp: &ast::Expression<'ast>,       // required type (from call context)
        egv: &mut Vec<ast::ConstantGenericValue<'ast>>,
        gid_map: &HashMap<&str, usize>,
    ) -> ZVisitorResult {
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
    }

    fn get_struct(&self, id: &str) -> ZResult<&ast::StructDefinition<'ast>> { unimplemented!() }

    fn generic_defined(&self, id: &str) -> bool { unimplemented!() }
}
