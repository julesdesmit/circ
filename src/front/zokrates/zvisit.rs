//! AST Walker for zokrates_pest_ast
#![allow(missing_docs)]

use std::collections::HashMap;
use zokrates_pest_ast as ast;

use super::term::Ty;
use super::ZGen;

pub struct ZVisitorError(pub String);
pub type ZResult<T> = Result<T, ZVisitorError>;
pub type ZVisitorResult = ZResult<()>;

pub trait ZVisitorMut<'ast>: Sized {
    fn visit_file(&mut self, file: &mut ast::File<'ast>) -> ZVisitorResult {
        walk_file(self, file)
    }

    fn visit_pragma(&mut self, pragma: &mut ast::Pragma<'ast>) -> ZVisitorResult {
        walk_pragma(self, pragma)
    }

    fn visit_curve(&mut self, curve: &mut ast::Curve<'ast>) -> ZVisitorResult {
        walk_curve(self, curve)
    }

    fn visit_span(&mut self, _span: &mut ast::Span<'ast>) -> ZVisitorResult {
        Ok(())
    }

    fn visit_symbol_declaration(
        &mut self,
        sd: &mut ast::SymbolDeclaration<'ast>,
    ) -> ZVisitorResult {
        walk_symbol_declaration(self, sd)
    }

    fn visit_eoi(&mut self, _eoi: &mut ast::EOI) -> ZVisitorResult {
        Ok(())
    }

    fn visit_import_directive(
        &mut self,
        import: &mut ast::ImportDirective<'ast>,
    ) -> ZVisitorResult {
        walk_import_directive(self, import)
    }

    fn visit_main_import_directive(
        &mut self,
        mimport: &mut ast::MainImportDirective<'ast>,
    ) -> ZVisitorResult {
        walk_main_import_directive(self, mimport)
    }

    fn visit_from_import_directive(
        &mut self,
        fimport: &mut ast::FromImportDirective<'ast>,
    ) -> ZVisitorResult {
        walk_from_import_directive(self, fimport)
    }

    fn visit_import_source(&mut self, is: &mut ast::ImportSource<'ast>) -> ZVisitorResult {
        walk_import_source(self, is)
    }

    fn visit_import_symbol(&mut self, is: &mut ast::ImportSymbol<'ast>) -> ZVisitorResult {
        walk_import_symbol(self, is)
    }

    fn visit_identifier_expression(
        &mut self,
        ie: &mut ast::IdentifierExpression<'ast>,
    ) -> ZVisitorResult {
        walk_identifier_expression(self, ie)
    }

    fn visit_constant_definition(
        &mut self,
        cnstdef: &mut ast::ConstantDefinition<'ast>,
    ) -> ZVisitorResult {
        walk_constant_definition(self, cnstdef)
    }

    fn visit_struct_definition(
        &mut self,
        structdef: &mut ast::StructDefinition<'ast>,
    ) -> ZVisitorResult {
        walk_struct_definition(self, structdef)
    }

    fn visit_struct_field(&mut self, structfield: &mut ast::StructField<'ast>) -> ZVisitorResult {
        walk_struct_field(self, structfield)
    }

    fn visit_function_definition(
        &mut self,
        fundef: &mut ast::FunctionDefinition<'ast>,
    ) -> ZVisitorResult {
        walk_function_definition(self, fundef)
    }

    fn visit_parameter(&mut self, param: &mut ast::Parameter<'ast>) -> ZVisitorResult {
        walk_parameter(self, param)
    }

    fn visit_visibility(&mut self, vis: &mut ast::Visibility<'ast>) -> ZVisitorResult {
        walk_visibility(self, vis)
    }

    fn visit_public_visibility(&mut self, _pu: &mut ast::PublicVisibility) -> ZVisitorResult {
        Ok(())
    }

    fn visit_private_visibility(
        &mut self,
        pr: &mut ast::PrivateVisibility<'ast>,
    ) -> ZVisitorResult {
        walk_private_visibility(self, pr)
    }

    fn visit_private_number(&mut self, pn: &mut ast::PrivateNumber<'ast>) -> ZVisitorResult {
        walk_private_number(self, pn)
    }

    fn visit_type(&mut self, ty: &mut ast::Type<'ast>) -> ZVisitorResult {
        walk_type(self, ty)
    }

    fn visit_basic_type(&mut self, bty: &mut ast::BasicType<'ast>) -> ZVisitorResult {
        walk_basic_type(self, bty)
    }

    fn visit_field_type(&mut self, fty: &mut ast::FieldType<'ast>) -> ZVisitorResult {
        walk_field_type(self, fty)
    }

    fn visit_boolean_type(&mut self, bty: &mut ast::BooleanType<'ast>) -> ZVisitorResult {
        walk_boolean_type(self, bty)
    }

    fn visit_u8_type(&mut self, u8ty: &mut ast::U8Type<'ast>) -> ZVisitorResult {
        walk_u8_type(self, u8ty)
    }

    fn visit_u16_type(&mut self, u16ty: &mut ast::U16Type<'ast>) -> ZVisitorResult {
        walk_u16_type(self, u16ty)
    }

    fn visit_u32_type(&mut self, u32ty: &mut ast::U32Type<'ast>) -> ZVisitorResult {
        walk_u32_type(self, u32ty)
    }

    fn visit_u64_type(&mut self, u64ty: &mut ast::U64Type<'ast>) -> ZVisitorResult {
        walk_u64_type(self, u64ty)
    }

    fn visit_array_type(&mut self, aty: &mut ast::ArrayType<'ast>) -> ZVisitorResult {
        walk_array_type(self, aty)
    }

    fn visit_basic_or_struct_type(
        &mut self,
        bsty: &mut ast::BasicOrStructType<'ast>,
    ) -> ZVisitorResult {
        walk_basic_or_struct_type(self, bsty)
    }

    fn visit_struct_type(&mut self, sty: &mut ast::StructType<'ast>) -> ZVisitorResult {
        walk_struct_type(self, sty)
    }

    fn visit_explicit_generics(&mut self, eg: &mut ast::ExplicitGenerics<'ast>) -> ZVisitorResult {
        walk_explicit_generics(self, eg)
    }

    fn visit_constant_generic_value(
        &mut self,
        cgv: &mut ast::ConstantGenericValue<'ast>,
    ) -> ZVisitorResult {
        walk_constant_generic_value(self, cgv)
    }

    fn visit_literal_expression(
        &mut self,
        lexpr: &mut ast::LiteralExpression<'ast>,
    ) -> ZVisitorResult {
        walk_literal_expression(self, lexpr)
    }

    fn visit_decimal_literal_expression(
        &mut self,
        dle: &mut ast::DecimalLiteralExpression<'ast>,
    ) -> ZVisitorResult {
        walk_decimal_literal_expression(self, dle)
    }

    fn visit_decimal_number(&mut self, dn: &mut ast::DecimalNumber<'ast>) -> ZVisitorResult {
        walk_decimal_number(self, dn)
    }

    fn visit_decimal_suffix(&mut self, ds: &mut ast::DecimalSuffix<'ast>) -> ZVisitorResult {
        walk_decimal_suffix(self, ds)
    }

    fn visit_u8_suffix(&mut self, u8s: &mut ast::U8Suffix<'ast>) -> ZVisitorResult {
        walk_u8_suffix(self, u8s)
    }

    fn visit_u16_suffix(&mut self, u16s: &mut ast::U16Suffix<'ast>) -> ZVisitorResult {
        walk_u16_suffix(self, u16s)
    }

    fn visit_u32_suffix(&mut self, u32s: &mut ast::U32Suffix<'ast>) -> ZVisitorResult {
        walk_u32_suffix(self, u32s)
    }

    fn visit_u64_suffix(&mut self, u64s: &mut ast::U64Suffix<'ast>) -> ZVisitorResult {
        walk_u64_suffix(self, u64s)
    }

    fn visit_field_suffix(&mut self, fs: &mut ast::FieldSuffix<'ast>) -> ZVisitorResult {
        walk_field_suffix(self, fs)
    }

    fn visit_boolean_literal_expression(
        &mut self,
        ble: &mut ast::BooleanLiteralExpression<'ast>,
    ) -> ZVisitorResult {
        walk_boolean_literal_expression(self, ble)
    }

    fn visit_hex_literal_expression(
        &mut self,
        hle: &mut ast::HexLiteralExpression<'ast>,
    ) -> ZVisitorResult {
        walk_hex_literal_expression(self, hle)
    }

    fn visit_hex_number_expression(
        &mut self,
        hne: &mut ast::HexNumberExpression<'ast>,
    ) -> ZVisitorResult {
        walk_hex_number_expression(self, hne)
    }

    fn visit_u8_number_expression(
        &mut self,
        u8e: &mut ast::U8NumberExpression<'ast>,
    ) -> ZVisitorResult {
        walk_u8_number_expression(self, u8e)
    }

    fn visit_u16_number_expression(
        &mut self,
        u16e: &mut ast::U16NumberExpression<'ast>,
    ) -> ZVisitorResult {
        walk_u16_number_expression(self, u16e)
    }

    fn visit_u32_number_expression(
        &mut self,
        u32e: &mut ast::U32NumberExpression<'ast>,
    ) -> ZVisitorResult {
        walk_u32_number_expression(self, u32e)
    }

    fn visit_u64_number_expression(
        &mut self,
        u64e: &mut ast::U64NumberExpression<'ast>,
    ) -> ZVisitorResult {
        walk_u64_number_expression(self, u64e)
    }

    fn visit_underscore(&mut self, u: &mut ast::Underscore<'ast>) -> ZVisitorResult {
        walk_underscore(self, u)
    }

    fn visit_expression(&mut self, expr: &mut ast::Expression<'ast>) -> ZVisitorResult {
        walk_expression(self, expr)
    }

    fn visit_ternary_expression(
        &mut self,
        te: &mut ast::TernaryExpression<'ast>,
    ) -> ZVisitorResult {
        walk_ternary_expression(self, te)
    }

    fn visit_binary_expression(&mut self, be: &mut ast::BinaryExpression<'ast>) -> ZVisitorResult {
        walk_binary_expression(self, be)
    }

    fn visit_binary_operator(&mut self, _bo: &mut ast::BinaryOperator) -> ZVisitorResult {
        Ok(())
    }

    fn visit_unary_expression(&mut self, ue: &mut ast::UnaryExpression<'ast>) -> ZVisitorResult {
        walk_unary_expression(self, ue)
    }

    fn visit_unary_operator(&mut self, uo: &mut ast::UnaryOperator) -> ZVisitorResult {
        walk_unary_operator(self, uo)
    }

    fn visit_pos_operator(&mut self, _po: &mut ast::PosOperator) -> ZVisitorResult {
        Ok(())
    }

    fn visit_neg_operator(&mut self, _po: &mut ast::NegOperator) -> ZVisitorResult {
        Ok(())
    }

    fn visit_not_operator(&mut self, _po: &mut ast::NotOperator) -> ZVisitorResult {
        Ok(())
    }

    fn visit_postfix_expression(
        &mut self,
        pe: &mut ast::PostfixExpression<'ast>,
    ) -> ZVisitorResult {
        walk_postfix_expression(self, pe)
    }

    fn visit_access(&mut self, acc: &mut ast::Access<'ast>) -> ZVisitorResult {
        walk_access(self, acc)
    }

    fn visit_call_access(&mut self, ca: &mut ast::CallAccess<'ast>) -> ZVisitorResult {
        walk_call_access(self, ca)
    }

    fn visit_arguments(&mut self, args: &mut ast::Arguments<'ast>) -> ZVisitorResult {
        walk_arguments(self, args)
    }

    fn visit_array_access(&mut self, aa: &mut ast::ArrayAccess<'ast>) -> ZVisitorResult {
        walk_array_access(self, aa)
    }

    fn visit_range_or_expression(
        &mut self,
        roe: &mut ast::RangeOrExpression<'ast>,
    ) -> ZVisitorResult {
        walk_range_or_expression(self, roe)
    }

    fn visit_range(&mut self, rng: &mut ast::Range<'ast>) -> ZVisitorResult {
        walk_range(self, rng)
    }

    fn visit_from_expression(&mut self, from: &mut ast::FromExpression<'ast>) -> ZVisitorResult {
        walk_from_expression(self, from)
    }

    fn visit_to_expression(&mut self, to: &mut ast::ToExpression<'ast>) -> ZVisitorResult {
        walk_to_expression(self, to)
    }

    fn visit_member_access(&mut self, ma: &mut ast::MemberAccess<'ast>) -> ZVisitorResult {
        walk_member_access(self, ma)
    }

    fn visit_inline_array_expression(
        &mut self,
        iae: &mut ast::InlineArrayExpression<'ast>,
    ) -> ZVisitorResult {
        walk_inline_array_expression(self, iae)
    }

    fn visit_spread_or_expression(
        &mut self,
        soe: &mut ast::SpreadOrExpression<'ast>,
    ) -> ZVisitorResult {
        walk_spread_or_expression(self, soe)
    }

    fn visit_spread(&mut self, spread: &mut ast::Spread<'ast>) -> ZVisitorResult {
        walk_spread(self, spread)
    }

    fn visit_inline_struct_expression(
        &mut self,
        ise: &mut ast::InlineStructExpression<'ast>,
    ) -> ZVisitorResult {
        walk_inline_struct_expression(self, ise)
    }

    fn visit_inline_struct_member(
        &mut self,
        ism: &mut ast::InlineStructMember<'ast>,
    ) -> ZVisitorResult {
        walk_inline_struct_member(self, ism)
    }

    fn visit_array_initializer_expression(
        &mut self,
        aie: &mut ast::ArrayInitializerExpression<'ast>,
    ) -> ZVisitorResult {
        walk_array_initializer_expression(self, aie)
    }

    fn visit_statement(&mut self, stmt: &mut ast::Statement<'ast>) -> ZVisitorResult {
        walk_statement(self, stmt)
    }

    fn visit_return_statement(&mut self, ret: &mut ast::ReturnStatement<'ast>) -> ZVisitorResult {
        walk_return_statement(self, ret)
    }

    fn visit_definition_statement(
        &mut self,
        def: &mut ast::DefinitionStatement<'ast>,
    ) -> ZVisitorResult {
        walk_definition_statement(self, def)
    }

    fn visit_typed_identifier_or_assignee(
        &mut self,
        tioa: &mut ast::TypedIdentifierOrAssignee<'ast>,
    ) -> ZVisitorResult {
        walk_typed_identifier_or_assignee(self, tioa)
    }

    fn visit_typed_identifier(&mut self, ti: &mut ast::TypedIdentifier<'ast>) -> ZVisitorResult {
        walk_typed_identifier(self, ti)
    }

    fn visit_assignee(&mut self, asgn: &mut ast::Assignee<'ast>) -> ZVisitorResult {
        walk_assignee(self, asgn)
    }

    fn visit_assignee_access(&mut self, acc: &mut ast::AssigneeAccess<'ast>) -> ZVisitorResult {
        walk_assignee_access(self, acc)
    }

    fn visit_assertion_statement(
        &mut self,
        asrt: &mut ast::AssertionStatement<'ast>,
    ) -> ZVisitorResult {
        walk_assertion_statement(self, asrt)
    }

    fn visit_iteration_statement(
        &mut self,
        iter: &mut ast::IterationStatement<'ast>,
    ) -> ZVisitorResult {
        walk_iteration_statement(self, iter)
    }
}

pub fn walk_file<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    file: &mut ast::File<'ast>,
) -> ZVisitorResult {
    if let Some(p) = &mut file.pragma {
        visitor.visit_pragma(p)?;
    }
    file.declarations
        .iter_mut()
        .try_for_each(|d| visitor.visit_symbol_declaration(d))?;
    visitor.visit_eoi(&mut file.eoi)?;
    visitor.visit_span(&mut file.span)
}

pub fn walk_pragma<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    pragma: &mut ast::Pragma<'ast>,
) -> ZVisitorResult {
    visitor.visit_curve(&mut pragma.curve)?;
    visitor.visit_span(&mut pragma.span)
}

pub fn walk_curve<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    curve: &mut ast::Curve<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut curve.span)
}

pub fn walk_symbol_declaration<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    sd: &mut ast::SymbolDeclaration<'ast>,
) -> ZVisitorResult {
    use ast::SymbolDeclaration::*;
    match sd {
        Import(i) => visitor.visit_import_directive(i),
        Constant(c) => visitor.visit_constant_definition(c),
        Struct(s) => visitor.visit_struct_definition(s),
        Function(f) => visitor.visit_function_definition(f),
    }
}

pub fn walk_import_directive<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    import: &mut ast::ImportDirective<'ast>,
) -> ZVisitorResult {
    use ast::ImportDirective::*;
    match import {
        Main(m) => visitor.visit_main_import_directive(m),
        From(f) => visitor.visit_from_import_directive(f),
    }
}

pub fn walk_main_import_directive<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    mimport: &mut ast::MainImportDirective<'ast>,
) -> ZVisitorResult {
    visitor.visit_import_source(&mut mimport.source)?;
    if let Some(ie) = &mut mimport.alias {
        visitor.visit_identifier_expression(ie)?;
    }
    visitor.visit_span(&mut mimport.span)
}

pub fn walk_from_import_directive<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    fimport: &mut ast::FromImportDirective<'ast>,
) -> ZVisitorResult {
    visitor.visit_import_source(&mut fimport.source)?;
    fimport
        .symbols
        .iter_mut()
        .try_for_each(|s| visitor.visit_import_symbol(s))?;
    visitor.visit_span(&mut fimport.span)
}

pub fn walk_import_source<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    is: &mut ast::ImportSource<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut is.span)
}

pub fn walk_identifier_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    ie: &mut ast::IdentifierExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut ie.span)
}

pub fn walk_import_symbol<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    is: &mut ast::ImportSymbol<'ast>,
) -> ZVisitorResult {
    visitor.visit_identifier_expression(&mut is.id)?;
    if let Some(ie) = &mut is.alias {
        visitor.visit_identifier_expression(ie)?;
    }
    visitor.visit_span(&mut is.span)
}

pub fn walk_constant_definition<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    cnstdef: &mut ast::ConstantDefinition<'ast>,
) -> ZVisitorResult {
    visitor.visit_type(&mut cnstdef.ty)?;
    visitor.visit_identifier_expression(&mut cnstdef.id)?;
    visitor.visit_expression(&mut cnstdef.expression)?;
    visitor.visit_span(&mut cnstdef.span)
}

pub fn walk_struct_definition<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    structdef: &mut ast::StructDefinition<'ast>,
) -> ZVisitorResult {
    visitor.visit_identifier_expression(&mut structdef.id)?;
    structdef
        .generics
        .iter_mut()
        .try_for_each(|g| visitor.visit_identifier_expression(g))?;
    structdef
        .fields
        .iter_mut()
        .try_for_each(|f| visitor.visit_struct_field(f))?;
    visitor.visit_span(&mut structdef.span)
}

pub fn walk_struct_field<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    structfield: &mut ast::StructField<'ast>,
) -> ZVisitorResult {
    visitor.visit_type(&mut structfield.ty)?;
    visitor.visit_identifier_expression(&mut structfield.id)?;
    visitor.visit_span(&mut structfield.span)
}

pub fn walk_function_definition<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    fundef: &mut ast::FunctionDefinition<'ast>,
) -> ZVisitorResult {
    visitor.visit_identifier_expression(&mut fundef.id)?;
    fundef
        .generics
        .iter_mut()
        .try_for_each(|g| visitor.visit_identifier_expression(g))?;
    fundef
        .parameters
        .iter_mut()
        .try_for_each(|p| visitor.visit_parameter(p))?;
    fundef
        .returns
        .iter_mut()
        .try_for_each(|r| visitor.visit_type(r))?;
    fundef
        .statements
        .iter_mut()
        .try_for_each(|s| visitor.visit_statement(s))?;
    visitor.visit_span(&mut fundef.span)
}

pub fn walk_parameter<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    param: &mut ast::Parameter<'ast>,
) -> ZVisitorResult {
    if let Some(v) = &mut param.visibility {
        visitor.visit_visibility(v)?;
    }
    visitor.visit_type(&mut param.ty)?;
    visitor.visit_identifier_expression(&mut param.id)?;
    visitor.visit_span(&mut param.span)
}

pub fn walk_visibility<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    vis: &mut ast::Visibility<'ast>,
) -> ZVisitorResult {
    use ast::Visibility::*;
    match vis {
        Public(pu) => visitor.visit_public_visibility(pu),
        Private(pr) => visitor.visit_private_visibility(pr),
    }
}

pub fn walk_private_visibility<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    prv: &mut ast::PrivateVisibility<'ast>,
) -> ZVisitorResult {
    if let Some(pn) = &mut prv.number {
        visitor.visit_private_number(pn)?;
    }
    visitor.visit_span(&mut prv.span)
}

pub fn walk_private_number<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    pn: &mut ast::PrivateNumber<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut pn.span)
}

pub fn walk_type<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    ty: &mut ast::Type<'ast>,
) -> ZVisitorResult {
    use ast::Type::*;
    match ty {
        Basic(b) => visitor.visit_basic_type(b),
        Array(a) => visitor.visit_array_type(a),
        Struct(s) => visitor.visit_struct_type(s),
    }
}

pub fn walk_basic_type<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    bty: &mut ast::BasicType<'ast>,
) -> ZVisitorResult {
    use ast::BasicType::*;
    match bty {
        Field(f) => visitor.visit_field_type(f),
        Boolean(b) => visitor.visit_boolean_type(b),
        U8(u) => visitor.visit_u8_type(u),
        U16(u) => visitor.visit_u16_type(u),
        U32(u) => visitor.visit_u32_type(u),
        U64(u) => visitor.visit_u64_type(u),
    }
}

pub fn walk_field_type<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    fty: &mut ast::FieldType<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut fty.span)
}

pub fn walk_boolean_type<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    bty: &mut ast::BooleanType<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut bty.span)
}

pub fn walk_u8_type<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    u8ty: &mut ast::U8Type<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut u8ty.span)
}

pub fn walk_u16_type<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    u16ty: &mut ast::U16Type<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut u16ty.span)
}

pub fn walk_u32_type<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    u32ty: &mut ast::U32Type<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut u32ty.span)
}

pub fn walk_u64_type<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    u64ty: &mut ast::U64Type<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut u64ty.span)
}

pub fn walk_array_type<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    aty: &mut ast::ArrayType<'ast>,
) -> ZVisitorResult {
    visitor.visit_basic_or_struct_type(&mut aty.ty)?;
    aty.dimensions
        .iter_mut()
        .try_for_each(|d| visitor.visit_expression(d))?;
    visitor.visit_span(&mut aty.span)
}

pub fn walk_basic_or_struct_type<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    bsty: &mut ast::BasicOrStructType<'ast>,
) -> ZVisitorResult {
    use ast::BasicOrStructType::*;
    match bsty {
        Struct(s) => visitor.visit_struct_type(s),
        Basic(b) => visitor.visit_basic_type(b),
    }
}

pub fn walk_struct_type<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    sty: &mut ast::StructType<'ast>,
) -> ZVisitorResult {
    visitor.visit_identifier_expression(&mut sty.id)?;
    if let Some(eg) = &mut sty.explicit_generics {
        visitor.visit_explicit_generics(eg)?;
    }
    visitor.visit_span(&mut sty.span)
}

pub fn walk_explicit_generics<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    eg: &mut ast::ExplicitGenerics<'ast>,
) -> ZVisitorResult {
    eg.values
        .iter_mut()
        .try_for_each(|v| visitor.visit_constant_generic_value(v))?;
    visitor.visit_span(&mut eg.span)
}

pub fn walk_constant_generic_value<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    cgv: &mut ast::ConstantGenericValue<'ast>,
) -> ZVisitorResult {
    use ast::ConstantGenericValue::*;
    match cgv {
        Value(l) => visitor.visit_literal_expression(l),
        Identifier(i) => visitor.visit_identifier_expression(i),
        Underscore(u) => visitor.visit_underscore(u),
    }
}

pub fn walk_literal_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    lexpr: &mut ast::LiteralExpression<'ast>,
) -> ZVisitorResult {
    use ast::LiteralExpression::*;
    match lexpr {
        DecimalLiteral(d) => visitor.visit_decimal_literal_expression(d),
        BooleanLiteral(b) => visitor.visit_boolean_literal_expression(b),
        HexLiteral(h) => visitor.visit_hex_literal_expression(h),
    }
}

pub fn walk_decimal_literal_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    dle: &mut ast::DecimalLiteralExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_decimal_number(&mut dle.value)?;
    if let Some(s) = &mut dle.suffix {
        visitor.visit_decimal_suffix(s)?;
    }
    visitor.visit_span(&mut dle.span)
}

pub fn walk_decimal_number<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    dn: &mut ast::DecimalNumber<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut dn.span)
}

pub fn walk_decimal_suffix<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    ds: &mut ast::DecimalSuffix<'ast>,
) -> ZVisitorResult {
    use ast::DecimalSuffix::*;
    match ds {
        U8(u8s) => visitor.visit_u8_suffix(u8s),
        U16(u16s) => visitor.visit_u16_suffix(u16s),
        U32(u32s) => visitor.visit_u32_suffix(u32s),
        U64(u64s) => visitor.visit_u64_suffix(u64s),
        Field(fs) => visitor.visit_field_suffix(fs),
    }
}

pub fn walk_u8_suffix<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    u8s: &mut ast::U8Suffix<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut u8s.span)
}

pub fn walk_u16_suffix<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    u16s: &mut ast::U16Suffix<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut u16s.span)
}

pub fn walk_u32_suffix<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    u32s: &mut ast::U32Suffix<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut u32s.span)
}

pub fn walk_u64_suffix<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    u64s: &mut ast::U64Suffix<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut u64s.span)
}

pub fn walk_field_suffix<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    fs: &mut ast::FieldSuffix<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut fs.span)
}

pub fn walk_boolean_literal_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    ble: &mut ast::BooleanLiteralExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut ble.span)
}

pub fn walk_hex_literal_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    hle: &mut ast::HexLiteralExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_hex_number_expression(&mut hle.value)?;
    visitor.visit_span(&mut hle.span)
}

pub fn walk_hex_number_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    hne: &mut ast::HexNumberExpression<'ast>,
) -> ZVisitorResult {
    use ast::HexNumberExpression::*;
    match hne {
        U8(u8e) => visitor.visit_u8_number_expression(u8e),
        U16(u16e) => visitor.visit_u16_number_expression(u16e),
        U32(u32e) => visitor.visit_u32_number_expression(u32e),
        U64(u64e) => visitor.visit_u64_number_expression(u64e),
    }
}

pub fn walk_u8_number_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    u8e: &mut ast::U8NumberExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut u8e.span)
}

pub fn walk_u16_number_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    u16e: &mut ast::U16NumberExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut u16e.span)
}

pub fn walk_u32_number_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    u32e: &mut ast::U32NumberExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut u32e.span)
}

pub fn walk_u64_number_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    u64e: &mut ast::U64NumberExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut u64e.span)
}

pub fn walk_underscore<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    u: &mut ast::Underscore<'ast>,
) -> ZVisitorResult {
    visitor.visit_span(&mut u.span)
}

pub fn walk_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    expr: &mut ast::Expression<'ast>,
) -> ZVisitorResult {
    use ast::Expression::*;
    match expr {
        Ternary(te) => visitor.visit_ternary_expression(te),
        Binary(be) => visitor.visit_binary_expression(be),
        Unary(ue) => visitor.visit_unary_expression(ue),
        Postfix(pe) => visitor.visit_postfix_expression(pe),
        Identifier(ie) => visitor.visit_identifier_expression(ie),
        Literal(le) => visitor.visit_literal_expression(le),
        InlineArray(iae) => visitor.visit_inline_array_expression(iae),
        InlineStruct(ise) => visitor.visit_inline_struct_expression(ise),
        ArrayInitializer(aie) => visitor.visit_array_initializer_expression(aie),
    }
}

pub fn walk_ternary_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    te: &mut ast::TernaryExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_expression(&mut te.first)?;
    visitor.visit_expression(&mut te.second)?;
    visitor.visit_expression(&mut te.third)?;
    visitor.visit_span(&mut te.span)
}

pub fn walk_binary_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    be: &mut ast::BinaryExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_binary_operator(&mut be.op)?;
    visitor.visit_expression(&mut be.left)?;
    visitor.visit_expression(&mut be.right)?;
    visitor.visit_span(&mut be.span)
}

pub fn walk_unary_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    ue: &mut ast::UnaryExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_unary_operator(&mut ue.op)?;
    visitor.visit_expression(&mut ue.expression)?;
    visitor.visit_span(&mut ue.span)
}

pub fn walk_unary_operator<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    uo: &mut ast::UnaryOperator,
) -> ZVisitorResult {
    use ast::UnaryOperator::*;
    match uo {
        Pos(po) => visitor.visit_pos_operator(po),
        Neg(ne) => visitor.visit_neg_operator(ne),
        Not(no) => visitor.visit_not_operator(no),
    }
}

pub fn walk_postfix_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    pe: &mut ast::PostfixExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_identifier_expression(&mut pe.id)?;
    pe.accesses
        .iter_mut()
        .try_for_each(|a| visitor.visit_access(a))?;
    visitor.visit_span(&mut pe.span)
}

pub fn walk_access<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    acc: &mut ast::Access<'ast>,
) -> ZVisitorResult {
    use ast::Access::*;
    match acc {
        Call(ca) => visitor.visit_call_access(ca),
        Select(aa) => visitor.visit_array_access(aa),
        Member(ma) => visitor.visit_member_access(ma),
    }
}

pub fn walk_call_access<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    ca: &mut ast::CallAccess<'ast>,
) -> ZVisitorResult {
    if let Some(eg) = &mut ca.explicit_generics {
        visitor.visit_explicit_generics(eg)?;
    }
    visitor.visit_arguments(&mut ca.arguments)?;
    visitor.visit_span(&mut ca.span)
}

pub fn walk_arguments<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    args: &mut ast::Arguments<'ast>,
) -> ZVisitorResult {
    args.expressions
        .iter_mut()
        .try_for_each(|e| visitor.visit_expression(e))?;
    visitor.visit_span(&mut args.span)
}

pub fn walk_array_access<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    aa: &mut ast::ArrayAccess<'ast>,
) -> ZVisitorResult {
    visitor.visit_range_or_expression(&mut aa.expression)?;
    visitor.visit_span(&mut aa.span)
}

pub fn walk_range_or_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    roe: &mut ast::RangeOrExpression<'ast>,
) -> ZVisitorResult {
    use ast::RangeOrExpression::*;
    match roe {
        Range(r) => visitor.visit_range(r),
        Expression(e) => visitor.visit_expression(e),
    }
}

pub fn walk_range<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    rng: &mut ast::Range<'ast>,
) -> ZVisitorResult {
    if let Some(f) = &mut rng.from {
        visitor.visit_from_expression(f)?;
    }
    if let Some(t) = &mut rng.to {
        visitor.visit_to_expression(t)?;
    }
    visitor.visit_span(&mut rng.span)
}

pub fn walk_from_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    from: &mut ast::FromExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_expression(&mut from.0)
}

pub fn walk_to_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    to: &mut ast::ToExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_expression(&mut to.0)
}

pub fn walk_member_access<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    ma: &mut ast::MemberAccess<'ast>,
) -> ZVisitorResult {
    visitor.visit_identifier_expression(&mut ma.id)?;
    visitor.visit_span(&mut ma.span)
}

pub fn walk_inline_array_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    iae: &mut ast::InlineArrayExpression<'ast>,
) -> ZVisitorResult {
    iae.expressions
        .iter_mut()
        .try_for_each(|e| visitor.visit_spread_or_expression(e))?;
    visitor.visit_span(&mut iae.span)
}

pub fn walk_spread_or_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    soe: &mut ast::SpreadOrExpression<'ast>,
) -> ZVisitorResult {
    use ast::SpreadOrExpression::*;
    match soe {
        Spread(s) => visitor.visit_spread(s),
        Expression(e) => visitor.visit_expression(e),
    }
}

pub fn walk_spread<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    spread: &mut ast::Spread<'ast>,
) -> ZVisitorResult {
    visitor.visit_expression(&mut spread.expression)?;
    visitor.visit_span(&mut spread.span)
}

pub fn walk_inline_struct_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    ise: &mut ast::InlineStructExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_identifier_expression(&mut ise.ty)?;
    ise.members
        .iter_mut()
        .try_for_each(|m| visitor.visit_inline_struct_member(m))?;
    visitor.visit_span(&mut ise.span)
}

pub fn walk_inline_struct_member<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    ism: &mut ast::InlineStructMember<'ast>,
) -> ZVisitorResult {
    visitor.visit_identifier_expression(&mut ism.id)?;
    visitor.visit_expression(&mut ism.expression)?;
    visitor.visit_span(&mut ism.span)
}

pub fn walk_array_initializer_expression<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    aie: &mut ast::ArrayInitializerExpression<'ast>,
) -> ZVisitorResult {
    visitor.visit_expression(&mut aie.value)?;
    visitor.visit_expression(&mut aie.count)?;
    visitor.visit_span(&mut aie.span)
}

pub fn walk_statement<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    stmt: &mut ast::Statement<'ast>,
) -> ZVisitorResult {
    use ast::Statement::*;
    match stmt {
        Return(r) => visitor.visit_return_statement(r),
        Definition(d) => visitor.visit_definition_statement(d),
        Assertion(a) => visitor.visit_assertion_statement(a),
        Iteration(i) => visitor.visit_iteration_statement(i),
    }
}

pub fn walk_return_statement<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    ret: &mut ast::ReturnStatement<'ast>,
) -> ZVisitorResult {
    ret.expressions
        .iter_mut()
        .try_for_each(|e| visitor.visit_expression(e))?;
    visitor.visit_span(&mut ret.span)
}

pub fn walk_definition_statement<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    def: &mut ast::DefinitionStatement<'ast>,
) -> ZVisitorResult {
    def.lhs
        .iter_mut()
        .try_for_each(|l| visitor.visit_typed_identifier_or_assignee(l))?;
    visitor.visit_expression(&mut def.expression)?;
    visitor.visit_span(&mut def.span)
}

pub fn walk_typed_identifier_or_assignee<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    tioa: &mut ast::TypedIdentifierOrAssignee<'ast>,
) -> ZVisitorResult {
    use ast::TypedIdentifierOrAssignee::*;
    match tioa {
        Assignee(a) => visitor.visit_assignee(a),
        TypedIdentifier(ti) => visitor.visit_typed_identifier(ti),
    }
}

pub fn walk_typed_identifier<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    tid: &mut ast::TypedIdentifier<'ast>,
) -> ZVisitorResult {
    visitor.visit_type(&mut tid.ty)?;
    visitor.visit_identifier_expression(&mut tid.identifier)?;
    visitor.visit_span(&mut tid.span)
}

pub fn walk_assignee<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    asgn: &mut ast::Assignee<'ast>,
) -> ZVisitorResult {
    visitor.visit_identifier_expression(&mut asgn.id)?;
    asgn.accesses
        .iter_mut()
        .try_for_each(|a| visitor.visit_assignee_access(a))?;
    visitor.visit_span(&mut asgn.span)
}

pub fn walk_assignee_access<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    acc: &mut ast::AssigneeAccess<'ast>,
) -> ZVisitorResult {
    use ast::AssigneeAccess::*;
    match acc {
        Select(aa) => visitor.visit_array_access(aa),
        Member(ma) => visitor.visit_member_access(ma),
    }
}

pub fn walk_assertion_statement<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    asrt: &mut ast::AssertionStatement<'ast>,
) -> ZVisitorResult {
    visitor.visit_expression(&mut asrt.expression)?;
    visitor.visit_span(&mut asrt.span)
}

pub fn walk_iteration_statement<'ast, Z: ZVisitorMut<'ast>>(
    visitor: &mut Z,
    iter: &mut ast::IterationStatement<'ast>,
) -> ZVisitorResult {
    visitor.visit_type(&mut iter.ty)?;
    visitor.visit_identifier_expression(&mut iter.index)?;
    visitor.visit_expression(&mut iter.from)?;
    visitor.visit_expression(&mut iter.to)?;
    iter.statements
        .iter_mut()
        .try_for_each(|s| visitor.visit_statement(s))?;
    visitor.visit_span(&mut iter.span)
}

// *************************

fn eq_type<'ast>(ty: &ast::Type<'ast>, ty2: &ast::Type<'ast>) -> ZVisitorResult {
    use ast::Type::*;
    match (ty, ty2) {
        (Basic(bty), Basic(bty2)) => eq_basic_type(bty, bty2),
        (Array(aty), Array(aty2)) => eq_array_type(aty, aty2),
        (Struct(sty), Struct(sty2)) => eq_struct_type(sty, sty2),
        _ => Err(ZVisitorError(format!(
            "type mismatch: expected {:?}, found {:?}",
            ty, ty2,
        ))),
    }
}

fn eq_basic_type<'ast>(ty: &ast::BasicType<'ast>, ty2: &ast::BasicType<'ast>) -> ZVisitorResult {
    use ast::BasicType::*;
    match (ty, ty2) {
        (Field(_), Field(_)) => Ok(()),
        (Boolean(_), Boolean(_)) => Ok(()),
        (U8(_), U8(_)) => Ok(()),
        (U16(_), U16(_)) => Ok(()),
        (U32(_), U32(_)) => Ok(()),
        (U64(_), U64(_)) => Ok(()),
        _ => Err(ZVisitorError(format!(
            "basic type mismatch: expected {:?}, found {:?}",
            ty, ty2,
        ))),
    }
}

fn eq_array_type<'ast>(ty: &ast::ArrayType<'ast>, ty2: &ast::ArrayType<'ast>) -> ZVisitorResult {
    use ast::BasicOrStructType::*;
    if ty.dimensions.len() != ty2.dimensions.len() {
        return Err(ZVisitorError(format!(
            "array type mismatch: expected {}-dimensional array, found {}-dimensional array",
            ty.dimensions.len(),
            ty2.dimensions.len(),
        )));
    }
    match (&ty.ty, &ty2.ty) {
        (Basic(bty), Basic(bty2)) => eq_basic_type(bty, bty2),
        (Struct(sty), Struct(sty2)) => eq_struct_type(sty, sty2),
        _ => Err(ZVisitorError(format!(
            "array type mismatch: expected elms of type {:?}, found {:?}",
            &ty.ty, &ty2.ty,
        ))),
    }
}

fn eq_struct_type<'ast>(ty: &ast::StructType<'ast>, ty2: &ast::StructType<'ast>) -> ZVisitorResult {
    // XXX(unimpl) can monomorphization break this?
    if &ty.id.value != &ty2.id.value {
        Err(ZVisitorError(format!(
            "struct type mismatch: expected {:?}, found {:?}",
            &ty.id.value, &ty2.id.value,
        )))
    } else {
        Ok(())
    }
}

struct ZExpressionTyper<'ast, 'ret, 'wlk> {
    walker: &'wlk ZStatementWalker<'ast, 'ret>,
    ty: Option<ast::Type<'ast>>,
}

impl<'ast, 'ret, 'wlk> ZExpressionTyper<'ast, 'ret, 'wlk> {
    fn new(walker: &'wlk ZStatementWalker<'ast, 'ret>) -> Self {
        Self { walker, ty: None }
    }

    fn take(&mut self) -> Option<ast::Type<'ast>> {
        self.ty.take()
    }

    fn visit_identifier_expression_t(
        &mut self,
        ie: &mut ast::IdentifierExpression<'ast>,
    ) -> ZVisitorResult {
        assert!(self.ty.is_none());
        self.walker.lookup_type(ie).map(|t| {
            self.ty.replace(t);
        })
    }

    fn arrayize(
        &self,
        ty: ast::Type<'ast>,
        cnt: ast::Expression<'ast>,
        spn: &ast::Span<'ast>,
    ) -> ast::ArrayType<'ast> {
        use ast::Type::*;
        match ty {
            Array(mut aty) => {
                aty.dimensions.insert(0, cnt);
                aty
            }
            Basic(bty) => ast::ArrayType {
                ty: ast::BasicOrStructType::Basic(bty),
                dimensions: vec![cnt],
                span: spn.clone(),
            },
            Struct(sty) => ast::ArrayType {
                ty: ast::BasicOrStructType::Struct(sty),
                dimensions: vec![cnt],
                span: spn.clone(),
            },
        }
    }
}

impl<'ast, 'ret, 'wlk> ZVisitorMut<'ast> for ZExpressionTyper<'ast, 'ret, 'wlk> {
    fn visit_expression(&mut self, expr: &mut ast::Expression<'ast>) -> ZVisitorResult {
        use ast::Expression::*;
        if self.ty.is_some() {
            return Err(ZVisitorError("type found at expression entry?".to_string()));
        }
        match expr {
            Ternary(te) => self.visit_ternary_expression(te),
            Binary(be) => self.visit_binary_expression(be),
            Unary(ue) => self.visit_unary_expression(ue),
            Postfix(pe) => self.visit_postfix_expression(pe),
            Identifier(ie) => self.visit_identifier_expression_t(ie),
            Literal(le) => self.visit_literal_expression(le),
            InlineArray(iae) => self.visit_inline_array_expression(iae),
            InlineStruct(ise) => self.visit_inline_struct_expression(ise),
            ArrayInitializer(aie) => self.visit_array_initializer_expression(aie),
        }
    }

    fn visit_ternary_expression(
        &mut self,
        te: &mut ast::TernaryExpression<'ast>,
    ) -> ZVisitorResult {
        assert!(self.ty.is_none());
        self.visit_expression(&mut te.second)?;
        let ty2 = self.take();
        self.visit_expression(&mut te.third)?;
        let ty3 = self.take();
        match (ty2, ty3) {
            (Some(t), None) => self.ty.replace(t),
            (None, Some(t)) => self.ty.replace(t),
            (Some(t1), Some(t2)) => {
                eq_type(&t1, &t2)?;
                self.ty.replace(t2)
            }
            (None, None) => None,
        };
        Ok(())
    }

    fn visit_binary_expression(&mut self, be: &mut ast::BinaryExpression<'ast>) -> ZVisitorResult {
        use ast::{BasicType::*, BinaryOperator::*, Type::*};
        assert!(self.ty.is_none());
        match &be.op {
            Or | And | Eq | NotEq | Lt | Gt | Lte | Gte => {
                self.ty.replace(Basic(Boolean(ast::BooleanType {
                    span: be.span.clone(),
                })));
            }
            Pow => {
                self.ty.replace(Basic(Field(ast::FieldType {
                    span: be.span.clone(),
                })));
            }
            BitXor | BitAnd | BitOr | RightShift | LeftShift | Add | Sub | Mul | Div | Rem => {
                self.visit_expression(&mut be.left)?;
                let ty_l = self.take();
                self.visit_expression(&mut be.right)?;
                let ty_r = self.take();
                if let Some(ty) = match (ty_l, ty_r) {
                    (Some(t), None) => Some(t),
                    (None, Some(t)) => Some(t),
                    (Some(t1), Some(t2)) => {
                        eq_type(&t1, &t2)?;
                        Some(t2)
                    }
                    (None, None) => None,
                } {
                    if !matches!(&ty, Basic(_)) {
                        return Err(ZVisitorError("got non-Basic type for a binop".to_string()));
                    }
                    if matches!(&ty, Basic(Boolean(_))) {
                        return Err(ZVisitorError(
                            "got Bool for a binop that cannot support it".to_string(),
                        ));
                    }
                    if matches!(
                        &be.op,
                        BitXor | BitAnd | BitOr | RightShift | LeftShift | Rem
                    ) && matches!(&ty, Basic(Field(_)))
                    {
                        return Err(ZVisitorError(
                            "got Field for a binop that cannot support it".to_string(),
                        ));
                    }
                    self.ty.replace(ty);
                }
            }
        };
        Ok(())
    }

    fn visit_unary_expression(&mut self, ue: &mut ast::UnaryExpression<'ast>) -> ZVisitorResult {
        use ast::{BasicType::*, Type::*, UnaryOperator::*};
        assert!(self.ty.is_none());
        match &ue.op {
            Pos(_) | Neg(_) => {
                self.visit_expression(&mut ue.expression)?;
                if let Some(ty) = &self.ty {
                    if !matches!(ty, Basic(_)) || matches!(ty, Basic(Boolean(_))) {
                        return Err(ZVisitorError(
                            "got Bool or non-Basic for unary op".to_string(),
                        ));
                    }
                }
            }
            Not(_) => {
                self.ty.replace(Basic(Boolean(ast::BooleanType {
                    span: ue.span.clone(),
                })));
            }
        }
        Ok(())
    }

    fn visit_boolean_literal_expression(
        &mut self,
        ble: &mut ast::BooleanLiteralExpression<'ast>,
    ) -> ZVisitorResult {
        assert!(self.ty.is_none());
        self.ty.replace(ast::Type::Basic(ast::BasicType::Boolean(
            ast::BooleanType {
                span: ble.span.clone(),
            },
        )));
        Ok(())
    }

    fn visit_decimal_suffix(&mut self, ds: &mut ast::DecimalSuffix<'ast>) -> ZVisitorResult {
        assert!(self.ty.is_none());
        use ast::{BasicType::*, DecimalSuffix as DS, Type::*};
        match ds {
            DS::U8(s) => self.ty.replace(Basic(U8(ast::U8Type {
                span: s.span.clone(),
            }))),
            DS::U16(s) => self.ty.replace(Basic(U16(ast::U16Type {
                span: s.span.clone(),
            }))),
            DS::U32(s) => self.ty.replace(Basic(U32(ast::U32Type {
                span: s.span.clone(),
            }))),
            DS::U64(s) => self.ty.replace(Basic(U64(ast::U64Type {
                span: s.span.clone(),
            }))),
            DS::Field(s) => self.ty.replace(Basic(Field(ast::FieldType {
                span: s.span.clone(),
            }))),
        };
        Ok(())
    }

    fn visit_hex_number_expression(
        &mut self,
        hne: &mut ast::HexNumberExpression<'ast>,
    ) -> ZVisitorResult {
        assert!(self.ty.is_none());
        use ast::{BasicType::*, HexNumberExpression as HNE, Type::*};
        match hne {
            HNE::U8(s) => self.ty.replace(Basic(U8(ast::U8Type {
                span: s.span.clone(),
            }))),
            HNE::U16(s) => self.ty.replace(Basic(U16(ast::U16Type {
                span: s.span.clone(),
            }))),
            HNE::U32(s) => self.ty.replace(Basic(U32(ast::U32Type {
                span: s.span.clone(),
            }))),
            HNE::U64(s) => self.ty.replace(Basic(U64(ast::U64Type {
                span: s.span.clone(),
            }))),
        };
        Ok(())
    }

    fn visit_array_initializer_expression(
        &mut self,
        aie: &mut ast::ArrayInitializerExpression<'ast>,
    ) -> ZVisitorResult {
        assert!(self.ty.is_none());
        use ast::Type::*;

        self.visit_expression(&mut *aie.value)?;
        if let Some(ty) = self.take() {
            let ty = self.arrayize(ty, aie.count.as_ref().clone(), &aie.span);
            self.ty.replace(Array(ty));
        }
        Ok(())
    }

    fn visit_inline_struct_expression(
        &mut self,
        ise: &mut ast::InlineStructExpression<'ast>,
    ) -> ZVisitorResult {
        // XXX(unimpl) we don't monomorphize struct type here... OK?
        self.visit_identifier_expression_t(&mut ise.ty)
    }

    fn visit_inline_array_expression(
        &mut self,
        iae: &mut ast::InlineArrayExpression<'ast>,
    ) -> ZVisitorResult {
        assert!(self.ty.is_none());
        assert!(!iae.expressions.is_empty());

        // XXX(unimpl) does not check array lengths
        let (sp, ex) = (&iae.span, &mut iae.expressions);
        let mut acc_ty = None;
        ex.iter_mut().try_for_each(|soe| {
            self.visit_spread_or_expression(soe)?;
            if let Some(ty) = self.take() {
                let ty = if matches!(soe, ast::SpreadOrExpression::Expression(_)) {
                    ast::Type::Array(self.arrayize(
                        ty,
                        ast::Expression::Literal(ast::LiteralExpression::HexLiteral(
                            ast::HexLiteralExpression {
                                value: ast::HexNumberExpression::U32(ast::U32NumberExpression {
                                    value: "0x0000".to_string(),
                                    span: sp.clone(),
                                }),
                                span: sp.clone(),
                            },
                        )),
                        sp,
                    ))
                } else {
                    ty
                };

                if let Some(acc) = &acc_ty {
                    eq_type(acc, &ty)?;
                } else {
                    acc_ty.replace(ty);
                }
            }
            Ok(())
        })?;

        self.ty = acc_ty;
        Ok(())
    }

    fn visit_postfix_expression(
        &mut self,
        pfe: &mut ast::PostfixExpression<'ast>,
    ) -> ZVisitorResult {
        assert!(self.ty.is_none());
        self.ty.replace(self.walker.get_postfix_ty(pfe, None)?);
        Ok(())
    }
}

struct ZExpressionRewriter<'ast> {
    gvmap: HashMap<String, ast::Expression<'ast>>,
}

impl<'ast> ZVisitorMut<'ast> for ZExpressionRewriter<'ast> {
    fn visit_expression(&mut self, expr: &mut ast::Expression<'ast>) -> ZVisitorResult {
        use ast::Expression::*;
        match expr {
            Ternary(te) => self.visit_ternary_expression(te),
            Binary(be) => self.visit_binary_expression(be),
            Unary(ue) => self.visit_unary_expression(ue),
            Postfix(pe) => self.visit_postfix_expression(pe),
            Literal(le) => self.visit_literal_expression(le),
            InlineArray(iae) => self.visit_inline_array_expression(iae),
            InlineStruct(ise) => self.visit_inline_struct_expression(ise),
            ArrayInitializer(aie) => self.visit_array_initializer_expression(aie),
            Identifier(ie) => {
                if let Some(e) = self.gvmap.get(&ie.value) {
                    *expr = e.clone();
                    Ok(())
                } else {
                    self.visit_identifier_expression(ie)
                }
            }
        }
    }
}

pub(super) struct ZStatementWalker<'ast, 'ret> {
    rets: &'ret [ast::Type<'ast>],
    gens: &'ret [ast::IdentifierExpression<'ast>],
    zgen: &'ret mut ZGen<'ast>,
    vars: Vec<HashMap<String, ast::Type<'ast>>>,
}

impl<'ast, 'ret> ZStatementWalker<'ast, 'ret> {
    pub fn new(
        prms: &'ret [ast::Parameter<'ast>],
        rets: &'ret [ast::Type<'ast>],
        gens: &'ret [ast::IdentifierExpression<'ast>],
        zgen: &'ret mut ZGen<'ast>,
    ) -> Self {
        let vars = vec![prms
            .iter()
            .map(|p| (p.id.value.clone(), p.ty.clone()))
            .collect()];
        Self {
            rets,
            gens,
            zgen,
            vars,
        }
    }

    // XXX(opt) take ref to Type instead of owned?
    fn unify(
        &self,
        ty: Option<ast::Type<'ast>>,
        expr: &mut ast::Expression<'ast>,
    ) -> ZVisitorResult {
        let mut rewriter = ZConstLiteralRewriter::new(None);
        rewriter.visit_expression(expr)?;
        let ty = if let Some(ty) = ty { ty } else { return Ok(()) };
        self.unify_expression(ty, expr)
    }

    fn unify_expression(
        &self,
        ty: ast::Type<'ast>,
        expr: &mut ast::Expression<'ast>,
    ) -> ZVisitorResult {
        use ast::Expression::*;
        match expr {
            Ternary(te) => self.unify_ternary(ty, te),
            Binary(be) => self.unify_binary(ty, be),
            Unary(ue) => self.unify_unary(ty, ue),
            Postfix(pe) => self.unify_postfix(ty, pe),
            Identifier(ie) => self.unify_identifier(ty, ie),
            Literal(le) => self.unify_literal(ty, le),
            InlineArray(ia) => self.unify_inline_array(ty, ia),
            InlineStruct(is) => self.unify_inline_struct(ty, is),
            ArrayInitializer(ai) => self.unify_array_initializer(ty, ai),
        }
    }

    fn fdef_gen_arg(
        &self,
        pty: &ast::Type<'ast>,
        exp: &ast::Expression<'ast>,
        egv: &mut Vec<ast::ConstantGenericValue<'ast>>,
        gid_map: &HashMap<&str, usize>,
    ) -> ZVisitorResult {
        Ok(())
    }

    fn fdef_gen_ret(
        &self,
        dty: &ast::Type<'ast>,
        rty: &ast::Type<'ast>,
        egv: &mut Vec<ast::ConstantGenericValue<'ast>>,
        gid_map: &HashMap<&str, usize>,
    ) -> ZVisitorResult {
        Ok(())
    }

    fn unify_fdef_call(
        &self,
        fdef: &ast::FunctionDefinition<'ast>,
        call: &mut ast::CallAccess<'ast>,
        rty: Option<&ast::Type<'ast>>,
    ) -> ZResult<ast::Type<'ast>> {
        if call.arguments.expressions.len() != fdef.parameters.len() {
            return Err(ZVisitorError(format!(
                "ZStatementWalker: wrong number of arguments to fn {}:\n{}",
                &fdef.id.value,
                span_to_string(&call.span),
            )));
        }
        if call.explicit_generics.is_some()
            && call.explicit_generics.as_ref().unwrap().values.len() != fdef.generics.len()
        {
            return Err(ZVisitorError(format!(
                "ZStatementWalker: wrong number of generic args to fn {}:\n{}",
                &fdef.id.value,
                span_to_string(&call.span),
            )));
        }
        // early return if no generics in this function call
        if fdef.generics.is_empty() {
            return Ok(fdef.returns.first().unwrap().clone());
        }

        // XXX(unimpl) generic inference in fn calls not yet supported
        use ast::ConstantGenericValue::*;
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
            for (exp, pty) in call.arguments.expressions.iter().zip(par.iter().map(|p| &p.ty)) {
                self.fdef_gen_arg(pty, exp, egv, &gid_map)?;
            }

            // step 3: optionally unify return type and update cgvs
            if let Some(rty) = rty {
                // XXX(unimpl) multi-return statements not supported
                self.fdef_gen_ret(&ret[0], rty, egv, &gid_map)?;
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
        // XXX(perf) do this without so much cloning?
        let egv = call
            .explicit_generics
            .as_ref()
            .map(|eg| {
                {
                    eg.values.clone().into_iter().map(|cgv| match cgv {
                        Underscore(_) => unreachable!(),
                        Value(l) => ast::Expression::Literal(l),
                        Identifier(i) => ast::Expression::Identifier(i),
                    })
                }
                .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let gvmap = fdef
            .generics
            .clone()
            .into_iter()
            .map(|ie| ie.value)
            .zip(egv.into_iter())
            .collect::<HashMap<String, ast::Expression<'ast>>>();

        let mut ret_rewriter = ZExpressionRewriter { gvmap };
        let mut ret_ty = fdef.returns.first().unwrap().clone();
        ret_rewriter.visit_type(&mut ret_ty).map(|_| ret_ty)
    }

    fn get_postfix_ty(
        &self,
        pf: &mut ast::PostfixExpression<'ast>,
        rty: Option<&ast::Type<'ast>>,
    ) -> ZResult<ast::Type<'ast>> {
        use ast::Access::*;
        assert!(!pf.accesses.is_empty());

        // XXX(assume) no functions in arrays or structs
        // handle first access, which is special because only this one could be a Call()
        let (id, acc) = (&pf.id, &mut pf.accesses);
        let alen = acc.len();
        let (pf_id_ty, acc_offset) = if let Call(ca) = acc.first_mut().unwrap() {
            // look up function type
            // XXX(todo) handle EMBED/* functions
            self.get_function(&id.value).and_then(|fdef| {
                if fdef.returns.is_empty() {
                    // XXX(unimpl) fn without return type not supported
                    Err(ZVisitorError(format!(
                        "ZStatementWalker: fn {} has no return type",
                        &id.value,
                    )))
                } else if fdef.returns.len() > 1 {
                    // XXX(unimpl) multiple return types not implemented
                    Err(ZVisitorError(format!(
                        "ZStatementWalker: fn {} has multiple returns",
                        &id.value,
                    )))
                } else {
                    let rty = if alen == 1 {
                        rty
                    } else {
                        None
                    };
                    Ok((self.unify_fdef_call(fdef, ca, rty)?, 1))
                }
            })?
        } else {
            // just look up variable type
            (self.lookup_type(id)?, 0)
        };

        // typecheck the remaining accesses
        self.walk_accesses(pf_id_ty, &pf.accesses[acc_offset..], acc_to_msacc)
    }

    fn unify_postfix(
        &self,
        ty: ast::Type<'ast>,
        pf: &mut ast::PostfixExpression<'ast>,
    ) -> ZVisitorResult {
        let acc_ty = self.get_postfix_ty(pf, Some(&ty))?;
        eq_type(&ty, &acc_ty)
    }

    fn unify_array_initializer(
        &self,
        ty: ast::Type<'ast>,
        ai: &mut ast::ArrayInitializerExpression<'ast>,
    ) -> ZVisitorResult {
        use ast::Type::*;
        let mut at = if let Array(at) = ty {
            at
        } else {
            return Err(ZVisitorError(format!(
                "ZStatementWalker: array initializer expression wanted type {:?}:\n{}",
                &ty,
                span_to_string(&ai.span),
            )));
        };
        assert!(!at.dimensions.is_empty());

        // XXX(unimpl) does not check array lengths, just unifies ai.count with U32!
        let u32_ty = Basic(ast::BasicType::U32(ast::U32Type {
            span: ai.span.clone(),
        }));
        self.unify_expression(u32_ty, &mut *ai.count)?;

        let arr_ty = if at.dimensions.len() > 1 {
            at.dimensions.remove(0); // perf?
            Array(at)
        } else {
            bos_to_type(at.ty)
        };
        self.unify_expression(arr_ty, &mut *ai.value)
    }

    fn unify_inline_struct(
        &self,
        ty: ast::Type<'ast>,
        is: &mut ast::InlineStructExpression<'ast>,
    ) -> ZVisitorResult {
        use ast::Type::*;
        let st = if let Struct(st) = ty {
            st
        } else {
            return Err(ZVisitorError(format!(
                "ZStatementWalker: inline struct wanted type {:?}:\n{}",
                &ty,
                span_to_string(&is.span),
            )));
        };

        let (mut sm_types, st_name) = self.get_struct(&st.id.value).and_then(|sdef| {
            if sdef.id.value.starts_with(&is.ty.value) {
                let st_name = if &is.ty.value != &sdef.id.value {
                    // rewrite AST to monomorphized version
                    std::mem::replace(&mut is.ty.value, sdef.id.value.clone())
                } else {
                    sdef.id.value.clone()
                };
                let sm_types = sdef
                    .fields
                    .iter()
                    .map(|sf| (sf.id.value.clone(), sf.ty.clone()))
                    .collect::<HashMap<String, ast::Type<'ast>>>();
                Ok((sm_types, st_name))
            } else {
                Err(ZVisitorError(format!(
                    "ZStatementWalker: inline struct wanted struct type {} but declared {}:\n{}",
                    &st.id.value,
                    &is.ty.value,
                    span_to_string(&is.span),
                )))
            }
        })?;

        is.members.iter_mut().try_for_each(|ism| {
            sm_types
                .remove(ism.id.value.as_str())
                .ok_or_else(|| {
                    ZVisitorError(format!(
                        "ZStatementWalker: struct {} has no member {}, or duplicate member in expression",
                        &st_name, &ism.id.value,
                    ))
                })
                .and_then(|sm_ty| self.unify_expression(sm_ty, &mut ism.expression))
        })?;

        // make sure InlineStructExpression declared all members
        if !sm_types.is_empty() {
            Err(ZVisitorError(format!(
                "ZStatementWalker: struct {} inline decl missing members {:?}\n",
                &st_name,
                sm_types.keys().collect::<Vec<_>>()
            )))
        } else {
            Ok(())
        }
    }

    fn unify_inline_array(
        &self,
        ty: ast::Type<'ast>,
        ia: &mut ast::InlineArrayExpression<'ast>,
    ) -> ZVisitorResult {
        use ast::{SpreadOrExpression::*, Type::*};
        let at = if let Array(at) = ty {
            at
        } else {
            return Err(ZVisitorError(format!(
                "ZStatementWalker: inline array wanted type {:?}:\n{}",
                &ty,
                span_to_string(&ia.span),
            )));
        };

        // XXX(unimpl) does not check array lengths, just unifies types!
        ia.expressions.iter_mut().try_for_each(|soe| match soe {
            Spread(s) => self.unify_expression(Array(at.clone()), &mut s.expression),
            Expression(e) => self.unify_expression(bos_to_type(at.ty.clone()), e),
        })
    }

    fn unify_identifier(
        &self,
        ty: ast::Type<'ast>,
        ie: &mut ast::IdentifierExpression<'ast>,
    ) -> ZVisitorResult {
        self.lookup_type(ie).and_then(|ity| eq_type(&ty, &ity))
    }

    fn unify_ternary(
        &self,
        ty: ast::Type<'ast>,
        te: &mut ast::TernaryExpression<'ast>,
    ) -> ZVisitorResult {
        // first expr must have type Bool, others the expected output type
        let bool_ty = ast::Type::Basic(ast::BasicType::Boolean(ast::BooleanType {
            span: te.span.clone(),
        }));
        self.unify_expression(bool_ty, &mut te.first)?;
        self.unify_expression(ty.clone(), &mut te.second)?;
        self.unify_expression(ty, &mut te.third)
    }

    fn unify_binary(
        &self,
        ty: ast::Type<'ast>,
        be: &mut ast::BinaryExpression<'ast>,
    ) -> ZVisitorResult {
        use ast::{BasicType::*, BinaryOperator::*, Type::*};
        let bt = if let Basic(bt) = ty {
            bt
        } else {
            return Err(ZVisitorError(format!(
                "ZStatementWalker: binary operators require Basic operands:\n{}",
                span_to_string(&be.span),
            )));
        };

        let (lt, rt) = match &be.op {
            BitXor | BitAnd | BitOr | Rem => match &bt {
                U8(_) | U16(_) | U32(_) | U64(_) => Ok((Basic(bt.clone()), Basic(bt))),
                _ => Err(ZVisitorError(
                    "ZStatementWalker: Bit/Rem operators require U* operands".to_owned(),
                )),
            },
            RightShift | LeftShift => match &bt {
                U8(_) | U16(_) | U32(_) | U64(_) => Ok((
                    Basic(bt),
                    Basic(U32(ast::U32Type {
                        span: be.span.clone(),
                    })),
                )),
                _ => Err(ZVisitorError(
                    "ZStatementWalker: << and >> operators require U* left operand".to_owned(),
                )),
            },
            Or | And => match &bt {
                Boolean(_) => Ok((Basic(bt.clone()), Basic(bt))),
                _ => Err(ZVisitorError(
                    "ZStatementWalker: Logical-And/Or operators require Bool operands".to_owned(),
                )),
            },
            Add | Sub | Mul | Div => match &bt {
                Boolean(_) => Err(ZVisitorError(
                    "ZStatementWalker: +,-,*,/ operators require Field or U* operands".to_owned(),
                )),
                _ => Ok((Basic(bt.clone()), Basic(bt))),
            },
            Eq | NotEq | Lt | Gt | Lte | Gte => match &bt {
                Boolean(_) => {
                    let mut expr_walker = ZExpressionTyper::new(self);
                    expr_walker.visit_expression(&mut be.left)?;
                    let lty = expr_walker.take();
                    expr_walker.visit_expression(&mut be.right)?;
                    let rty = expr_walker.take();
                    match (&lty, &rty) {
                            (Some(Basic(_)), None) => Ok((lty.clone().unwrap(), lty.unwrap())),
                            (None, Some(Basic(_))) => Ok((rty.clone().unwrap(), rty.unwrap())),
                            (Some(Basic(_)), Some(Basic(_))) => {
                                let lty = lty.unwrap();
                                let rty = rty.unwrap();
                                eq_type(&lty, &rty)
                                    .map_err(|e|
                                    ZVisitorError(format!(
                                        "ZStatementWalker: got differing types {:?}, {:?} for lhs, rhs of expr:\n{}\n{}",
                                        &lty,
                                        &rty,
                                        e.0,
                                        span_to_string(&be.span),
                                    )))
                                    .map(|_| (lty, rty))
                            }
                            (None, None) => Err(ZVisitorError(format!(
                                "ZStatementWalker: could not infer type of binop:\n{}",
                                span_to_string(&be.span),
                            ))),
                            _ => Err(ZVisitorError(format!(
                                "ZStatementWalker: unknown error in binop typing:\n{}",
                                span_to_string(&be.span),
                            ))),
                        }
                        .and_then(|(lty, rty)| if matches!(&be.op, Lt | Gt | Lte | Gte) && matches!(lty, Basic(Boolean(_))) {
                            Err(ZVisitorError(format!(
                                "ZStatementWalker: >,>=,<,<= operators cannot be applied to Bool:\n{}",
                                span_to_string(&be.span),
                            )))
                        } else {
                            Ok((lty, rty))
                        })
                }
                _ => Err(ZVisitorError(
                    "ZStatementWalker: comparison and equality operators output Bool".to_owned(),
                )),
            },
            Pow => match &bt {
                // XXX does POW operator really require U32 RHS?
                Field(_) => Ok((
                    Basic(bt),
                    Basic(U32(ast::U32Type {
                        span: be.span.clone(),
                    })),
                )),
                _ => Err(ZVisitorError(
                    "ZStatementWalker: pow operator must take Field LHS and U32 RHS".to_owned(),
                )),
            },
        }?;
        self.unify_expression(lt, &mut be.left)?;
        self.unify_expression(rt, &mut be.right)
    }

    fn unify_unary(
        &self,
        ty: ast::Type<'ast>,
        ue: &mut ast::UnaryExpression<'ast>,
    ) -> ZVisitorResult {
        use ast::{BasicType::*, Type::*, UnaryOperator::*};
        let bt = if let Basic(bt) = ty {
            bt
        } else {
            return Err(ZVisitorError(format!(
                "ZStatementWalker: unary operators require Basic operands:\n{}",
                span_to_string(&ue.span),
            )));
        };

        let ety = match &ue.op {
            Pos(_) | Neg(_) => match &bt {
                Boolean(_) => Err(ZVisitorError(
                    "ZStatementWalker: +,- unary operators require Field or U* operands"
                        .to_string(),
                )),
                _ => Ok(Basic(bt)),
            },
            Not(_) => match &bt {
                Boolean(_) => Ok(Basic(bt)),
                _ => Err(ZVisitorError(
                    "ZStatementWalker: ! unary operator requires Bool operand".to_string(),
                )),
            },
        }?;

        self.unify_expression(ety, &mut ue.expression)
    }

    fn unify_literal(
        &self,
        ty: ast::Type<'ast>,
        le: &mut ast::LiteralExpression<'ast>,
    ) -> ZVisitorResult {
        use ast::{BasicType::*, LiteralExpression::*, Type::*};
        let bt = if let Basic(bt) = ty {
            bt
        } else {
            return Err(ZVisitorError(format!(
                "ZStatementWalker: literal expressions must yield basic types:\n{}",
                span_to_string(le.span()),
            )));
        };

        match le {
            BooleanLiteral(_) => {
                if let Boolean(_) = &bt {
                    Ok(())
                } else {
                    Err(ZVisitorError(format!(
                        "ZStatementWalker: expected {:?}, found BooleanLiteral:\n{}",
                        &bt,
                        span_to_string(le.span()),
                    )))
                }
            }
            HexLiteral(hle) => {
                use ast::HexNumberExpression as HNE;
                match &hle.value {
                    HNE::U8(_) if matches!(&bt, U8(_)) => Ok(()),
                    HNE::U16(_) if matches!(&bt, U16(_)) => Ok(()),
                    HNE::U32(_) if matches!(&bt, U32(_)) => Ok(()),
                    HNE::U64(_) if matches!(&bt, U64(_)) => Ok(()),
                    _ => Err(ZVisitorError(format!(
                        "ZStatementWalker: HexLiteral seemed to want type {:?}:\n{}",
                        &bt,
                        span_to_string(&hle.span),
                    ))),
                }
            }
            DecimalLiteral(dle) => {
                use ast::DecimalSuffix as DS;
                match &dle.suffix {
                    Some(ds) => match (ds, &bt) {
                        (DS::Field(_), Field(_)) => Ok(()),
                        (DS::U8(_), U8(_)) => Ok(()),
                        (DS::U16(_), U16(_)) => Ok(()),
                        (DS::U32(_), U32(_)) => Ok(()),
                        (DS::U64(_), U64(_)) => Ok(()),
                        _ => Err(ZVisitorError(format!(
                            "ZStatementWalker: DecimalLiteral wanted {:?} found {:?}:\n{}",
                            &bt,
                            ds,
                            span_to_string(&dle.span),
                        ))),
                    },
                    None => match &bt {
                        Boolean(_) => Err(ZVisitorError(format!(
                            "ZStatementWalker: DecimalLiteral wanted Bool:\n{}",
                            span_to_string(&dle.span),
                        ))),
                        Field(_) => Ok(DS::Field(ast::FieldSuffix {
                            span: dle.span.clone(),
                        })),
                        U8(_) => Ok(DS::U8(ast::U8Suffix {
                            span: dle.span.clone(),
                        })),
                        U16(_) => Ok(DS::U16(ast::U16Suffix {
                            span: dle.span.clone(),
                        })),
                        U32(_) => Ok(DS::U32(ast::U32Suffix {
                            span: dle.span.clone(),
                        })),
                        U64(_) => Ok(DS::U64(ast::U64Suffix {
                            span: dle.span.clone(),
                        })),
                    }
                    .map(|ds| {
                        dle.suffix.replace(ds);
                    }),
                }
            }
        }
    }

    // XXX(q) unify access expressions?
    fn walk_accesses<F, T>(
        &self,
        mut ty: ast::Type<'ast>,
        accs: &[T],
        f: F,
    ) -> ZResult<ast::Type<'ast>>
    where
        F: Fn(&T) -> ZResult<MSAccRef<'_, 'ast>>,
    {
        use ast::Type;
        use MSAccRef::*;
        let mut acc_dim_offset = 0;
        for acc in accs {
            if matches!(ty, Type::Basic(_)) {
                return Err(ZVisitorError(
                    "ZStatementWalker: tried to walk accesses into a Basic type".to_string(),
                ));
            }
            ty = match f(acc)? {
                Select(aacc) => {
                    if let Type::Array(aty) = ty {
                        use ast::RangeOrExpression::*;
                        match &aacc.expression {
                            Range(r) => {
                                // XXX(q): range checks here?
                                // XXX(q): can we simplify exprs here to binops on generics?
                                let from = Box::new(if let Some(f) = &r.from {
                                    f.0.clone()
                                } else {
                                    let f_expr = ast::U32NumberExpression {
                                        value: "0x0000".to_string(),
                                        span: r.span.clone(),
                                    };
                                    let f_hlit = ast::HexLiteralExpression {
                                        value: ast::HexNumberExpression::U32(f_expr),
                                        span: r.span.clone(),
                                    };
                                    let f_lexp = ast::LiteralExpression::HexLiteral(f_hlit);
                                    ast::Expression::Literal(f_lexp)
                                });
                                let to = Box::new(if let Some(t) = &r.to {
                                    t.0.clone()
                                } else {
                                    aty.dimensions[acc_dim_offset].clone()
                                });
                                let r_bexp = ast::BinaryExpression {
                                    op: ast::BinaryOperator::Sub,
                                    left: to,
                                    right: from,
                                    span: r.span.clone(),
                                };
                                let mut aty = aty;
                                aty.dimensions[acc_dim_offset] = ast::Expression::Binary(r_bexp);
                                Type::Array(aty)
                            }
                            Expression(_) => {
                                if aty.dimensions.len() - acc_dim_offset > 1 {
                                    acc_dim_offset += 1;
                                    Type::Array(aty)
                                } else {
                                    acc_dim_offset = 0;
                                    bos_to_type(aty.ty)
                                }
                            }
                        }
                    } else {
                        return Err(ZVisitorError(
                            "ZStatementWalker: tried to access an Array as a Struct".to_string(),
                        ));
                    }
                }
                Member(macc) => {
                    // XXX(unimpl) LHS of definitions must make generics explicit
                    if let Type::Struct(sty) = ty {
                        let sdef = self.get_struct(&sty.id.value)
                            .and_then(|sdef| {
                                if !sdef.generics.is_empty() {
                                    Err(ZVisitorError(format!("ZStatementWalker: encountered non-monomorphized struct type {}", &sty.id.value)))
                                } else {
                                    Ok(sdef)
                                }
                            })?;

                        // asdf
                        sdef.fields
                            .iter()
                            .find(|f| &f.id.value == &macc.id.value)
                            .ok_or_else(|| {
                                ZVisitorError(format!(
                                    "ZStatementWalker: struct {} has no member {}",
                                    &sty.id.value, &macc.id.value,
                                ))
                            })
                            .map(|f| f.ty.clone())?
                    } else {
                        return Err(ZVisitorError(
                            "ZStatementWalker: tried to access a Struct as an Array".to_string(),
                        ));
                    }
                }
            }
        }

        // handle any dimensional readjustments we've delayed
        if acc_dim_offset > 0 {
            ty = if let Type::Array(mut aty) = ty {
                Type::Array(ast::ArrayType {
                    ty: aty.ty,
                    dimensions: aty.dimensions.drain(acc_dim_offset..).collect(),
                    span: aty.span,
                })
            } else {
                unreachable!("acc_dim_offset != 0 when ty not Array");
            }
        }

        Ok(ty)
    }

    fn get_function(&self, id: &str) -> ZResult<&ast::FunctionDefinition<'ast>> {
        self.zgen
            .get_function(id)
            .ok_or_else(|| ZVisitorError(format!("ZStatementWalker: undeclared function {}", id)))
    }

    fn get_struct(&self, id: &str) -> ZResult<&ast::StructDefinition<'ast>> {
        self.zgen.get_struct(id).ok_or_else(|| {
            ZVisitorError(format!("ZStatementWalker: undeclared struct type {}", id))
        })
    }

    fn const_defined(&self, id: &str) -> bool {
        self.zgen.const_defined(id)
    }

    fn generic_defined(&self, id: &str) -> bool {
        // XXX(perf) if self.gens is long this could be improved with a HashSet.
        // Realistically, a function will have a small number of generic params.
        self.gens.iter().any(|g| &g.value == id)
    }

    fn var_defined(&self, id: &str) -> bool {
        self.vars.iter().rev().any(|v| v.contains_key(id))
    }

    fn lookup_var(&self, nm: &str) -> Option<ast::Type<'ast>> {
        self.vars.iter().rev().find_map(|v| v.get(nm).cloned())
    }

    fn lookup_type(&self, id: &ast::IdentifierExpression<'ast>) -> ZResult<ast::Type<'ast>> {
        if self.generic_defined(&id.value) {
            // generics are always U32
            Ok(ast::Type::Basic(ast::BasicType::U32(ast::U32Type {
                span: id.span.clone(),
            })))
        } else if let Some(t) = self.zgen.const_ty_lookup_(&id.value) {
            Ok(t.clone())
        } else {
            self.lookup_var(&id.value).ok_or_else(|| {
                ZVisitorError(format!(
                    "ZStatementWalker: identifier {} undefined",
                    &id.value
                ))
            })
        }
    }

    fn apply_varonly<F, R>(&mut self, nm: &str, f: F) -> ZResult<R>
    where
        F: FnOnce(&mut Self, &str) -> R,
    {
        if self.generic_defined(nm) {
            Err(ZVisitorError(format!(
                "ZStatementWalker: attempted to shadow generic {}",
                nm,
            )))
        } else if self.zgen.const_lookup_(nm).is_some() {
            Err(ZVisitorError(format!(
                "ZStatementWalker: attempted to shadow const {}",
                nm,
            )))
        } else {
            Ok(f(self, nm))
        }
    }

    fn lookup_type_varonly(&mut self, nm: &str) -> ZResult<Option<ast::Type<'ast>>> {
        self.apply_varonly(nm, |s, nm| s.lookup_var(nm))
    }

    fn insert_var(&mut self, nm: &str, ty: ast::Type<'ast>) -> ZResult<Option<ast::Type<'ast>>> {
        self.apply_varonly(nm, |s, nm| {
            s.vars.last_mut().unwrap().insert(nm.to_string(), ty)
        })
    }

    fn push_scope(&mut self) {
        self.vars.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.vars.pop();
    }

    // XXX(q) instead monomorphize by making sure that ExplicitGenerics are attached to Type???
    fn monomorphize_struct(&mut self, sty: &mut ast::StructType<'ast>) -> ZResult<ast::Type<'ast>> {
        use ast::ConstantGenericValue::*;

        // get the struct definition and return early if we don't have to handle generics
        let sdef = self.get_struct(&sty.id.value)?;
        if sdef.generics.is_empty() {
            if sty.explicit_generics.is_some() {
                return Err(ZVisitorError(format!(
                    "ZStatementWalker: got explicit generics for non-generic struct type {}:\n{}",
                    &sty.id.value,
                    span_to_string(&sty.span),
                )));
            } else {
                return Ok(ast::Type::Struct(sty.clone()));
            }
        }

        // set up mapping from generic names to values
        let mut gen_values = sty
            .explicit_generics
            .take()
            .ok_or_else(|| {
                ZVisitorError(format!(
                    "ZStatementWalker: must declare explicit generics for type {} in LHS:\n{}",
                    &sty.id.value,
                    span_to_string(&sty.span),
                ))
            })
            .and_then(|eg| {
                if eg.values.len() != sdef.generics.len() {
                    Err(ZVisitorError(format!(
                        "ZStatementWalker: wrong number of explicit generics for struct {}:\n{}",
                        &sty.id.value,
                        span_to_string(&sty.span),
                    )))
                } else if eg.values.iter().any(|v| matches!(v, Underscore(_))) {
                    Err(ZVisitorError(format!(
                        "ZStatementWalker: must specify all generic arguments for LHS struct {}:\n{}",
                        &sty.id.value,
                        span_to_string(&sty.span),
                    )))
                } else {
                    // make sure identifiers are actually defined!
                    eg.values.iter().try_for_each(|v|
                        if let Identifier(ie) = v {
                            if self.const_defined(&ie.value) || self.generic_defined(&ie.value) {
                                Ok(())
                            } else {
                                Err(ZVisitorError(format!(
                                    "ZStatementWalker: {} undef or non-const in {} generics:\n{}",
                                    &ie.value,
                                    &sty.id.value,
                                    span_to_string(&sty.span),
                                )))
                            }
                        } else {
                            Ok(())
                        }
                    ).map(|_| eg)
                }
            })?.values;
        let mut rewriter = ZConstLiteralRewriter::new(None);
        gen_values
            .iter_mut()
            .try_for_each(|cgv| rewriter.visit_constant_generic_value(cgv))?;
        sty.id.value = format!("{}*{:?}", &sdef.id.value, &gen_values);

        // rewrite struct definition if necessary
        if !self.zgen.struct_defined(&sty.id.value) {
            let generics = sdef.generics.clone();
            let mut sdef = ast::StructDefinition {
                id: ast::IdentifierExpression {
                    value: sty.id.value.clone(),
                    span: sdef.id.span.clone(),
                },
                generics: Vec::new(),
                fields: sdef.fields.clone(),
                span: sdef.span.clone(),
            };
            let gvmap = generics
                .into_iter()
                .map(|ie| ie.value)
                .zip(gen_values.into_iter().map(|cgv| match cgv {
                    Underscore(_) => unreachable!(),
                    Value(l) => ast::Expression::Literal(l),
                    Identifier(i) => ast::Expression::Identifier(i),
                }))
                .collect::<HashMap<String, ast::Expression<'ast>>>();

            // rewrite struct definition
            let mut sf_rewriter = ZExpressionRewriter { gvmap };
            sdef.fields
                .iter_mut()
                .try_for_each(|f| sf_rewriter.visit_struct_field(f))?;

            // save it
            self.zgen.insert_struct(&sty.id.value, sdef);
        }

        Ok(ast::Type::Struct(sty.clone()))
    }
}

impl<'ast, 'ret> ZVisitorMut<'ast> for ZStatementWalker<'ast, 'ret> {
    fn visit_return_statement(&mut self, ret: &mut ast::ReturnStatement<'ast>) -> ZVisitorResult {
        if self.rets.len() != ret.expressions.len() {
            return Err(ZVisitorError(
                "ZStatementWalker: mismatched return expression/type".to_owned(),
            ));
        }

        // XXX(unimpl) multi-return statements not supported
        if self.rets.len() > 1 {
            return Err(ZVisitorError(
                "ZStatementWalker: multi-returns not supported".to_owned(),
            ));
        }

        if let Some(expr) = ret.expressions.first_mut() {
            self.unify(self.rets.first().cloned(), expr)?;
        }
        walk_return_statement(self, ret)
    }

    fn visit_assertion_statement(
        &mut self,
        asrt: &mut ast::AssertionStatement<'ast>,
    ) -> ZVisitorResult {
        let bool_ty = ast::Type::Basic(ast::BasicType::Boolean(ast::BooleanType {
            span: asrt.span.clone(),
        }));
        self.unify(Some(bool_ty), &mut asrt.expression)?;
        walk_assertion_statement(self, asrt)
    }

    fn visit_iteration_statement(
        &mut self,
        iter: &mut ast::IterationStatement<'ast>,
    ) -> ZVisitorResult {
        self.visit_type(&mut iter.ty)?;

        self.push_scope(); // {
        self.insert_var(&iter.index.value, iter.ty.clone())?;
        self.visit_identifier_expression(&mut iter.index)?;

        // type propagation for index expressions
        self.unify(Some(iter.ty.clone()), &mut iter.from)?;
        self.visit_expression(&mut iter.from)?;
        self.unify(Some(iter.ty.clone()), &mut iter.to)?;
        self.visit_expression(&mut iter.to)?;

        iter.statements
            .iter_mut()
            .try_for_each(|s| self.visit_statement(s))?;

        self.pop_scope(); // }
        self.visit_span(&mut iter.span)
    }

    fn visit_definition_statement(
        &mut self,
        def: &mut ast::DefinitionStatement<'ast>,
    ) -> ZVisitorResult {
        def.lhs
            .iter_mut()
            .try_for_each(|l| self.visit_typed_identifier_or_assignee(l))?;

        // unify lhs and rhs
        // XXX(unimpl) multi-LHS statements not supported
        if def.lhs.len() > 1 {
            return Err(ZVisitorError(
                "ZStatementWalker: multi-LHS assignments not supported".to_owned(),
            ));
        }
        let ty_accs = def
            .lhs
            .first()
            .map(|tioa| {
                use ast::TypedIdentifierOrAssignee::*;
                let (na, acc) = match tioa {
                    Assignee(a) => (&a.id.value, a.accesses.as_ref()),
                    TypedIdentifier(ti) => (&ti.identifier.value, &[][..]),
                };
                self.lookup_type_varonly(na).map(|t| t.map(|t| (t, acc)))
            })
            .transpose()?
            .flatten();
        if let Some((ty, accs)) = ty_accs {
            let ty = self.walk_accesses(ty, accs, aacc_to_msacc)?;
            self.unify(Some(ty), &mut def.expression)?;
        } else {
            return Err(ZVisitorError(format!(
                "ZStatementWalker: found expression with no LHS:\n{}",
                span_to_string(&def.span),
            )));
        }
        self.visit_expression(&mut def.expression)?;
        self.visit_span(&mut def.span)
    }

    fn visit_typed_identifier_or_assignee(
        &mut self,
        tioa: &mut ast::TypedIdentifierOrAssignee<'ast>,
    ) -> ZVisitorResult {
        use ast::TypedIdentifierOrAssignee::*;
        match tioa {
            Assignee(a) => {
                if !self.var_defined(&a.id.value) {
                    Err(ZVisitorError(format!(
                        "ZStatementWalker: assignment to undeclared variable {}",
                        &a.id.value
                    )))
                } else {
                    self.visit_assignee(a)
                }
            }
            TypedIdentifier(ti) => {
                ZConstLiteralRewriter::new(None).visit_type(&mut ti.ty)?;
                let ty = if let ast::Type::Struct(sty) = &mut ti.ty {
                    self.monomorphize_struct(sty)?
                } else {
                    ti.ty.clone()
                };
                self.insert_var(&ti.identifier.value, ty)?;
                self.visit_typed_identifier(ti)
            }
        }
    }
}

// XXX(TODO) use ast::Type rather than Ty here
pub(super) struct ZConstLiteralRewriter {
    to_ty: Option<Ty>,
    found: bool,
}

impl ZConstLiteralRewriter {
    pub fn new(to_ty: Option<Ty>) -> Self {
        Self {
            to_ty,
            found: false,
        }
    }

    #[allow(dead_code)]
    pub fn found(&self) -> bool {
        self.found
    }

    pub fn replace(&mut self, to_ty: Option<Ty>) -> Option<Ty> {
        std::mem::replace(&mut self.to_ty, to_ty)
    }
}

impl<'ast> ZVisitorMut<'ast> for ZConstLiteralRewriter {
    /*
    Expressions can be any of:

    Binary(BinaryExpression<'ast>),
        -> depends on operator. e.g., == outputs Bool but takes in arbitrary l and r

    Ternary(TernaryExpression<'ast>)
        -> first expr is Bool, other two are expected type

    Unary(UnaryExpression<'ast>),
        -> no change to expected type: each sub-expr should have the expected type

    Postfix(PostfixExpression<'ast>),
        -> cannot type Access results, but descend into sub-exprs to type array indices

    Identifier(IdentifierExpression<'ast>),
        -> nothing to do (terminal)

    Literal(LiteralExpression<'ast>),
        -> literal should have same type as expression

    InlineArray(InlineArrayExpression<'ast>),
        -> descend into SpreadOrExpression, looking for either array or element type

    InlineStruct(InlineStructExpression<'ast>),
        -> XXX(unimpl) we do not know expected type (const structs not supported)

    ArrayInitializer(ArrayInitializerExpression<'ast>),
        -> value should have type of value inside Array
        -> count should have type Field
    */

    fn visit_ternary_expression(
        &mut self,
        te: &mut ast::TernaryExpression<'ast>,
    ) -> ZVisitorResult {
        // first expression in a ternary should have type bool
        let to_ty = self.replace(Some(Ty::Bool));
        self.visit_expression(&mut te.first)?;
        self.replace(to_ty);
        self.visit_expression(&mut te.second)?;
        self.visit_expression(&mut te.third)?;
        self.visit_span(&mut te.span)
    }

    fn visit_binary_expression(&mut self, be: &mut ast::BinaryExpression<'ast>) -> ZVisitorResult {
        let (ty_l, ty_r) = {
            use ast::BinaryOperator::*;
            match be.op {
                Pow | RightShift | LeftShift => (self.to_ty.clone(), Some(Ty::Uint(32))),
                Eq | NotEq | Lt | Gt | Lte | Gte => (None, None),
                _ => (self.to_ty.clone(), self.to_ty.clone()),
            }
        };
        self.visit_binary_operator(&mut be.op)?;
        let to_ty = self.replace(ty_l);
        self.visit_expression(&mut be.left)?;
        self.replace(ty_r);
        self.visit_expression(&mut be.right)?;
        self.replace(to_ty);
        self.visit_span(&mut be.span)
    }

    fn visit_decimal_literal_expression(
        &mut self,
        dle: &mut ast::DecimalLiteralExpression<'ast>,
    ) -> ZVisitorResult {
        if dle.suffix.is_none() && self.to_ty.is_some() {
            self.found = true;
            dle.suffix.replace(match self.to_ty.as_ref().unwrap() {
                Ty::Uint(8) => Ok(ast::DecimalSuffix::U8(ast::U8Suffix {
                    span: dle.span.clone(),
                })),
                Ty::Uint(16) => Ok(ast::DecimalSuffix::U16(ast::U16Suffix {
                    span: dle.span.clone(),
                })),
                Ty::Uint(32) => Ok(ast::DecimalSuffix::U32(ast::U32Suffix {
                    span: dle.span.clone(),
                })),
                Ty::Uint(64) => Ok(ast::DecimalSuffix::U64(ast::U64Suffix {
                    span: dle.span.clone(),
                })),
                Ty::Uint(_) => Err(ZVisitorError(
                    "ZConstLiteralRewriter: Uint size must be divisible by 8".to_owned(),
                )),
                Ty::Field => Ok(ast::DecimalSuffix::Field(ast::FieldSuffix {
                    span: dle.span.clone(),
                })),
                _ => Err(ZVisitorError(
                    "ZConstLiteralRewriter: rewriting DecimalLiteralExpression to incompatible type"
                        .to_owned(),
                )),
            }?);
        }
        walk_decimal_literal_expression(self, dle)
    }

    fn visit_array_initializer_expression(
        &mut self,
        aie: &mut ast::ArrayInitializerExpression<'ast>,
    ) -> ZVisitorResult {
        if self.to_ty.is_some() {
            if let Ty::Array(_, arr_ty) = self.to_ty.clone().unwrap() {
                // ArrayInitializerExpression::value should match arr_ty
                let to_ty = self.replace(Some(*arr_ty));
                self.visit_expression(&mut aie.value)?;
                self.to_ty = to_ty;
            } else {
                return Err(ZVisitorError(
                    "ZConstLiteralRewriter: rewriting ArrayInitializerExpression to non-Array type"
                        .to_string(),
                ));
            }
        }

        // always rewrite ArrayInitializerExpression::count literals to type U32
        let to_ty = self.replace(Some(Ty::Uint(32)));
        self.visit_expression(&mut aie.count)?;
        self.to_ty = to_ty;

        self.visit_span(&mut aie.span)
    }

    fn visit_array_access(&mut self, acc: &mut ast::ArrayAccess<'ast>) -> ZVisitorResult {
        // always rewrite ArrayAccess literals to type U32
        let to_ty = self.replace(Some(Ty::Uint(32)));
        walk_array_access(self, acc)?;
        self.to_ty = to_ty;
        Ok(())
    }

    fn visit_inline_struct_expression(
        &mut self,
        ise: &mut ast::InlineStructExpression<'ast>,
    ) -> ZVisitorResult {
        self.visit_identifier_expression(&mut ise.ty)?;

        let to_ty = self.replace(None);
        ise.members
            .iter_mut()
            .try_for_each(|m| self.visit_inline_struct_member(m))?;
        self.to_ty = to_ty;

        self.visit_span(&mut ise.span)
    }

    fn visit_inline_array_expression(
        &mut self,
        iae: &mut ast::InlineArrayExpression<'ast>,
    ) -> ZVisitorResult {
        let mut inner_ty = if let Some(t) = self.to_ty.as_ref() {
            if let Ty::Array(_, arr_ty) = t.clone() {
                Ok(Some(*arr_ty))
            } else {
                Err(ZVisitorError(
                    "ZConstLiteralRewriter: rewriting InlineArrayExpression to non-Array type"
                        .to_string(),
                ))
            }
        } else {
            Ok(None)
        }?;

        for e in iae.expressions.iter_mut() {
            use ast::SpreadOrExpression::*;
            match e {
                Spread(s) => {
                    // a spread expression is an array; array type should match (we ignore number)
                    self.visit_spread(s)?;
                }
                Expression(e) => {
                    // an expression here is an individual array element, inner type should match
                    inner_ty = self.replace(inner_ty);
                    self.visit_expression(e)?;
                    inner_ty = self.replace(inner_ty);
                }
            }
        }

        self.visit_span(&mut iae.span)
    }

    fn visit_postfix_expression(
        &mut self,
        pe: &mut ast::PostfixExpression<'ast>,
    ) -> ZVisitorResult {
        self.visit_identifier_expression(&mut pe.id)?;

        // descend into accesses. we do not know expected type for these expressions
        // (but we may end up descending into an ArrayAccess, which would get typed)
        let to_ty = self.replace(None);
        pe.accesses
            .iter_mut()
            .try_for_each(|a| self.visit_access(a))?;
        self.to_ty = to_ty;

        self.visit_span(&mut pe.span)
    }

    fn visit_array_type(
        &mut self,
        aty: &mut ast::ArrayType<'ast>,
    ) -> ZVisitorResult {
        if self.to_ty.is_some() {
            if let Ty::Array(_, arr_ty) = self.to_ty.clone().unwrap() {
                // ArrayType::value should match arr_ty
                let to_ty = self.replace(Some(*arr_ty));
                self.visit_basic_or_struct_type(&mut aty.ty)?;
                self.to_ty = to_ty;
            } else {
                return Err(ZVisitorError(
                    "ZConstLiteralRewriter: rewriting ArrayType to non-Array type"
                        .to_string(),
                ));
            }
        }

        // always rewrite ArrayType::dimensions literals to type U32
        let to_ty = self.replace(Some(Ty::Uint(32)));
        aty.dimensions.iter_mut().try_for_each(|d| self.visit_expression(d))?;
        self.to_ty = to_ty;

        self.visit_span(&mut aty.span)
    }

    fn visit_field_type(
        &mut self,
        fty: &mut ast::FieldType<'ast>,
    ) -> ZVisitorResult {
        if self.to_ty.is_some() && !matches!(self.to_ty, Some(Ty::Field)) {
            return Err(ZVisitorError(
                "ZConstLiteralRewriter: Field type mismatch".to_string(),
            ));
        }
        walk_field_type(self, fty)
    }

    fn visit_boolean_type(
        &mut self,
        bty: &mut ast::BooleanType<'ast>,
    ) -> ZVisitorResult {
        if self.to_ty.is_some() && !matches!(self.to_ty, Some(Ty::Bool)) {
            return Err(ZVisitorError(
                "ZConstLiteralRewriter: Bool type mismatch".to_string(),
            ));
        }
        walk_boolean_type(self, bty)
    }

    fn visit_u8_type(
        &mut self,
        u8ty: &mut ast::U8Type<'ast>,
    ) -> ZVisitorResult {
        if self.to_ty.is_some() && !matches!(self.to_ty, Some(Ty::Uint(8))) {
            return Err(ZVisitorError(
                "ZConstLiteralRewriter: u8 type mismatch".to_string(),
            ));
        }
        walk_u8_type(self, u8ty)
    }

    fn visit_u16_type(
        &mut self,
        u16ty: &mut ast::U16Type<'ast>,
    ) -> ZVisitorResult {
        if self.to_ty.is_some() && !matches!(self.to_ty, Some(Ty::Uint(16))) {
            return Err(ZVisitorError(
                "ZConstLiteralRewriter: u16 type mismatch".to_string(),
            ));
        }
        walk_u16_type(self, u16ty)
    }

    fn visit_u32_type(
        &mut self,
        u32ty: &mut ast::U32Type<'ast>,
    ) -> ZVisitorResult {
        if self.to_ty.is_some() && !matches!(self.to_ty, Some(Ty::Uint(32))) {
            return Err(ZVisitorError(
                "ZConstLiteralRewriter: u32 type mismatch".to_string(),
            ));
        }
        walk_u32_type(self, u32ty)
    }

    fn visit_u64_type(
        &mut self,
        u64ty: &mut ast::U64Type<'ast>,
    ) -> ZVisitorResult {
        if self.to_ty.is_some() && !matches!(self.to_ty, Some(Ty::Uint(64))) {
            return Err(ZVisitorError(
                "ZConstLiteralRewriter: u64 type mismatch".to_string(),
            ));
        }
        walk_u64_type(self, u64ty)
    }
}

fn span_to_string(span: &ast::Span) -> String {
    span.lines().collect::<String>()
}

fn bos_to_type<'ast>(bos: ast::BasicOrStructType<'ast>) -> ast::Type<'ast> {
    use ast::{BasicOrStructType::*, Type};
    match bos {
        Struct(st) => Type::Struct(st),
        Basic(bt) => Type::Basic(bt),
    }
}

enum MSAccRef<'a, 'ast> {
    Select(&'a ast::ArrayAccess<'ast>),
    Member(&'a ast::MemberAccess<'ast>),
}

fn aacc_to_msacc<'a, 'ast>(i: &'a ast::AssigneeAccess<'ast>) -> ZResult<MSAccRef<'a, 'ast>> {
    use ast::AssigneeAccess::*;
    Ok(match i {
        Select(t) => MSAccRef::Select(t),
        Member(t) => MSAccRef::Member(t),
    })
}

fn acc_to_msacc<'a, 'ast>(i: &'a ast::Access<'ast>) -> ZResult<MSAccRef<'a, 'ast>> {
    use ast::Access::*;
    match i {
        Select(t) => Ok(MSAccRef::Select(t)),
        Member(t) => Ok(MSAccRef::Member(t)),
        Call(t) => Err(ZVisitorError(format!(
            "Illegal fn call:\n{}",
            span_to_string(&t.span),
        ))),
    }
}
