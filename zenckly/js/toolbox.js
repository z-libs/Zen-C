/**
 * Zenckly Toolbox Definition
 * Blocks organized into 13 categories.
 */
'use strict';

var ZENCKLY_TOOLBOX = {
  kind: 'categoryToolbox',
  contents: [
    // --- Variables ---
    {
      kind: 'category',
      name: 'Variables',
      colour: '30',
      contents: [
        { kind: 'block', type: 'zen_let' },
        { kind: 'block', type: 'zen_let_infer' },
        { kind: 'block', type: 'zen_const' },
        { kind: 'block', type: 'zen_static' },
        { kind: 'block', type: 'zen_number' },
        { kind: 'block', type: 'zen_boolean' },
        { kind: 'block', type: 'zen_char' },
        { kind: 'block', type: 'zen_string' },
        { kind: 'block', type: 'zen_variable_get' },
        { kind: 'block', type: 'zen_assign' },
        { kind: 'block', type: 'zen_assign_op' },
        { kind: 'block', type: 'zen_array_decl' },
        { kind: 'block', type: 'zen_array_literal' },
        { kind: 'block', type: 'zen_array_access' },
        { kind: 'block', type: 'zen_array_set' },
        { kind: 'block', type: 'zen_cast' },
        { kind: 'block', type: 'zen_sizeof' },
        { kind: 'block', type: 'zen_typeof' }
      ]
    },
    // --- Math ---
    {
      kind: 'category',
      name: 'Math',
      colour: '230',
      contents: [
        { kind: 'block', type: 'zen_math_arithmetic' },
        { kind: 'block', type: 'zen_math_compare' },
        { kind: 'block', type: 'zen_math_negate' },
        { kind: 'block', type: 'zen_math_constant' },
        { kind: 'block', type: 'zen_math_single' },
        { kind: 'block', type: 'zen_math_minmax' },
        { kind: 'block', type: 'zen_math_trig' },
        { kind: 'block', type: 'zen_math_pow' },
        { kind: 'block', type: 'zen_math_log' },
        { kind: 'block', type: 'zen_math_bitwise' },
        { kind: 'block', type: 'zen_math_bitnot' },
        { kind: 'block', type: 'zen_math_random_int' },
        { kind: 'block', type: 'zen_math_random_float' },
        { kind: 'block', type: 'zen_math_modulo' },
        { kind: 'block', type: 'zen_math_incdec' },
        { kind: 'block', type: 'zen_math_clamp' }
      ]
    },
    // --- Logic ---
    {
      kind: 'category',
      name: 'Logic',
      colour: '210',
      contents: [
        { kind: 'block', type: 'zen_if' },
        { kind: 'block', type: 'zen_if_else' },
        { kind: 'block', type: 'zen_else_if' },
        { kind: 'block', type: 'zen_logic_op' },
        { kind: 'block', type: 'zen_logic_not' },
        { kind: 'block', type: 'zen_match' },
        { kind: 'block', type: 'zen_match_case' }
      ]
    },
    // --- Loops ---
    {
      kind: 'category',
      name: 'Loops',
      colour: '120',
      contents: [
        { kind: 'block', type: 'zen_while' },
        { kind: 'block', type: 'zen_for' },
        { kind: 'block', type: 'zen_for_in' },
        { kind: 'block', type: 'zen_for_each' },
        { kind: 'block', type: 'zen_loop_forever' },
        { kind: 'block', type: 'zen_break' },
        { kind: 'block', type: 'zen_continue' }
      ]
    },
    // --- Functions ---
    {
      kind: 'category',
      name: 'Functions',
      colour: '290',
      contents: [
        { kind: 'block', type: 'zen_main' },
        { kind: 'block', type: 'zen_fn' },
        { kind: 'block', type: 'zen_fn_pub' },
        { kind: 'block', type: 'zen_return' },
        { kind: 'block', type: 'zen_return_void' },
        { kind: 'block', type: 'zen_call' },
        { kind: 'block', type: 'zen_call_expr' },
        { kind: 'block', type: 'zen_lambda' },
        { kind: 'block', type: 'zen_defer' },
        { kind: 'block', type: 'zen_extern' },
        { kind: 'block', type: 'zen_comptime' },
        { kind: 'block', type: 'zen_param' }
      ]
    },
    // --- Structs & Types ---
    {
      kind: 'category',
      name: 'Structs',
      colour: '330',
      contents: [
        { kind: 'block', type: 'zen_struct' },
        { kind: 'block', type: 'zen_struct_field' },
        { kind: 'block', type: 'zen_struct_init' },
        { kind: 'block', type: 'zen_field_access' },
        { kind: 'block', type: 'zen_field_set' },
        { kind: 'block', type: 'zen_impl' },
        { kind: 'block', type: 'zen_method' },
        { kind: 'block', type: 'zen_self' },
        { kind: 'block', type: 'zen_trait' },
        { kind: 'block', type: 'zen_enum' },
        { kind: 'block', type: 'zen_enum_variant' },
        { kind: 'block', type: 'zen_enum_variant_val' },
        { kind: 'block', type: 'zen_type_alias' },
        { kind: 'block', type: 'zen_generic' }
      ]
    },
    // --- Memory ---
    {
      kind: 'category',
      name: 'Memory',
      colour: '0',
      contents: [
        { kind: 'block', type: 'zen_alloc' },
        { kind: 'block', type: 'zen_free' },
        { kind: 'block', type: 'zen_addr_of' },
        { kind: 'block', type: 'zen_deref' },
        { kind: 'block', type: 'zen_ptr_decl' },
        { kind: 'block', type: 'zen_null' },
        { kind: 'block', type: 'zen_undefined' },
        { kind: 'block', type: 'zen_memcpy' },
        { kind: 'block', type: 'zen_memset' },
        { kind: 'block', type: 'zen_ptr_slice' },
        { kind: 'block', type: 'zen_autofree' }
      ]
    },
    // --- Text & I/O ---
    {
      kind: 'category',
      name: 'Text',
      colour: '160',
      contents: [
        { kind: 'block', type: 'zen_println' },
        { kind: 'block', type: 'zen_print' },
        { kind: 'block', type: 'zen_println_fmt' },
        { kind: 'block', type: 'zen_strlen' },
        { kind: 'block', type: 'zen_strcat' },
        { kind: 'block', type: 'zen_strcmp' },
        { kind: 'block', type: 'zen_strslice' },
        { kind: 'block', type: 'zen_char_at' },
        { kind: 'block', type: 'zen_to_string' },
        { kind: 'block', type: 'zen_parse_int' },
        { kind: 'block', type: 'zen_raw_code' },
        { kind: 'block', type: 'zen_comment' }
      ]
    },
    // --- Errors ---
    {
      kind: 'category',
      name: 'Errors',
      colour: '45',
      contents: [
        { kind: 'block', type: 'zen_ok' },
        { kind: 'block', type: 'zen_err' },
        { kind: 'block', type: 'zen_some' },
        { kind: 'block', type: 'zen_none' },
        { kind: 'block', type: 'zen_unwrap' },
        { kind: 'block', type: 'zen_unwrap_or' },
        { kind: 'block', type: 'zen_try' },
        { kind: 'block', type: 'zen_if_let' }
      ]
    },
    // --- Files ---
    {
      kind: 'category',
      name: 'Files',
      colour: '180',
      contents: [
        { kind: 'block', type: 'zen_fopen' },
        { kind: 'block', type: 'zen_fclose' },
        { kind: 'block', type: 'zen_fread' },
        { kind: 'block', type: 'zen_fwrite' },
        { kind: 'block', type: 'zen_freadline' },
        { kind: 'block', type: 'zen_file_exists' },
        { kind: 'block', type: 'zen_read_file' },
        { kind: 'block', type: 'zen_write_file' },
        { kind: 'block', type: 'zen_system' },
        { kind: 'block', type: 'zen_stdin_read' }
      ]
    },
    // --- Advanced ---
    {
      kind: 'category',
      name: 'Advanced',
      colour: '260',
      contents: [
        { kind: 'block', type: 'zen_method_call' },
        { kind: 'block', type: 'zen_method_call_stmt' },
        { kind: 'block', type: 'zen_static_call' },
        { kind: 'block', type: 'zen_static_call_stmt' },
        { kind: 'block', type: 'zen_guard' },
        { kind: 'block', type: 'zen_def' },
        { kind: 'block', type: 'zen_hex_number' },
        { kind: 'block', type: 'zen_packed_struct' }
      ]
    },
    // --- Collections ---
    {
      kind: 'category',
      name: 'Collections',
      colour: '190',
      contents: [
        { kind: 'block', type: 'zen_vec_new' },
        { kind: 'block', type: 'zen_vec_push' },
        { kind: 'block', type: 'zen_vec_pop' },
        { kind: 'block', type: 'zen_vec_len' },
        { kind: 'block', type: 'zen_vec_data_access' }
      ]
    },
    // --- Build ---
    {
      kind: 'category',
      name: 'Build',
      colour: '150',
      contents: [
        { kind: 'block', type: 'zen_import' },
        { kind: 'block', type: 'zen_cflags' },
        { kind: 'block', type: 'zen_target' },
        { kind: 'block', type: 'zen_freestanding' },
        { kind: 'block', type: 'zen_asm' },
        { kind: 'block', type: 'zen_linker' }
      ]
    }
  ]
};
