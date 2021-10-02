//! AST Walker for zokrates_pest_ast
#![allow(missing_docs)]

use zokrates_pest_ast as ast;

use super::term::Ty;

pub struct ZVisitorError(pub String);

pub type ZVisitorResult = Result<(), ZVisitorError>;

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

/*
Expressions can be any of:

Ternary(TernaryExpression<'ast>),
Binary(BinaryExpression<'ast>),
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

impl<'ast> ZVisitorMut<'ast> for ZConstLiteralRewriter {
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
                    "ZConstLiteralRewriter: tried to rewrite DecimalLiteralExpression to complex type"
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
                return Err(ZVisitorError("ZConstLiteralRewriter: tried to rewrite ArrayInitializerExpression to non-Array type".to_string()));
            }
        }

        // always rewrite ArrayInitializerExpression::count literals to type Field
        let to_ty = self.replace(Some(Ty::Field));
        self.visit_expression(&mut aie.count)?;
        self.to_ty = to_ty;

        self.visit_span(&mut aie.span)
    }

    fn visit_array_access(&mut self, acc: &mut ast::ArrayAccess<'ast>) -> ZVisitorResult {
        // always rewrite ArrayAccess literals to type Field
        let to_ty = self.replace(Some(Ty::Field));
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
                    "ZConstLiteralWriter: tried to rewrite InlineArrayExpression to non-Array type"
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
}
