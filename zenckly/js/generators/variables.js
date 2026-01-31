/**
 * Zen-C Variable Blocks (1-18)
 * let, const, static, types, arrays, cast, assignment, etc.
 */
'use strict';

var ZEN_TYPE_DROPDOWN = [
  ['int', 'int'], ['float', 'float'], ['double', 'double'],
  ['bool', 'bool'], ['char', 'char'], ['u8', 'u8'], ['u16', 'u16'],
  ['u32', 'u32'], ['u64', 'u64'], ['i8', 'i8'], ['i16', 'i16'],
  ['i32', 'i32'], ['i64', 'i64'], ['f32', 'f32'], ['f64', 'f64'],
  ['usize', 'usize'], ['isize', 'isize'], ['void', 'void']
];

// Block 1: let variable
Blockly.Blocks['zen_let'] = {
  init: function() {
    this.appendValueInput('VALUE')
        .appendField('let')
        .appendField(new Blockly.FieldTextInput('x'), 'VAR')
        .appendField(':')
        .appendField(new Blockly.FieldDropdown(ZEN_TYPE_DROPDOWN), 'TYPE')
        .appendField('=');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(30);
    this.setTooltip('Declare a variable with let');
  }
};

ZenC.forBlock['zen_let'] = function(block) {
  var name = block.getFieldValue('VAR');
  var type = block.getFieldValue('TYPE');
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '0';
  return 'let ' + name + ': ' + type + ' = ' + value + ';\n';
};

// Block 2: let (inferred type)
Blockly.Blocks['zen_let_infer'] = {
  init: function() {
    this.appendValueInput('VALUE')
        .appendField('let')
        .appendField(new Blockly.FieldTextInput('x'), 'VAR')
        .appendField('=');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(30);
    this.setTooltip('Declare a variable with inferred type');
  }
};

ZenC.forBlock['zen_let_infer'] = function(block) {
  var name = block.getFieldValue('VAR');
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '0';
  return 'let ' + name + ' = ' + value + ';\n';
};

// Block 3: const
Blockly.Blocks['zen_const'] = {
  init: function() {
    this.appendValueInput('VALUE')
        .appendField('const')
        .appendField(new Blockly.FieldTextInput('MAX'), 'VAR')
        .appendField(':')
        .appendField(new Blockly.FieldDropdown(ZEN_TYPE_DROPDOWN), 'TYPE')
        .appendField('=');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(30);
    this.setTooltip('Declare a compile-time constant');
  }
};

ZenC.forBlock['zen_const'] = function(block) {
  var name = block.getFieldValue('VAR');
  var type = block.getFieldValue('TYPE');
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '0';
  return 'const ' + name + ': ' + type + ' = ' + value + ';\n';
};

// Block 4: static variable
Blockly.Blocks['zen_static'] = {
  init: function() {
    this.appendValueInput('VALUE')
        .appendField('static')
        .appendField(new Blockly.FieldTextInput('counter'), 'VAR')
        .appendField(':')
        .appendField(new Blockly.FieldDropdown(ZEN_TYPE_DROPDOWN), 'TYPE')
        .appendField('=');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(30);
    this.setTooltip('Declare a static variable');
  }
};

ZenC.forBlock['zen_static'] = function(block) {
  var name = block.getFieldValue('VAR');
  var type = block.getFieldValue('TYPE');
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '0';
  return 'static ' + name + ': ' + type + ' = ' + value + ';\n';
};

// Block 5: number literal
Blockly.Blocks['zen_number'] = {
  init: function() {
    this.appendDummyInput()
        .appendField(new Blockly.FieldNumber(0), 'NUM');
    this.setOutput(true, 'Number');
    this.setColour(30);
    this.setTooltip('A number value');
  }
};

ZenC.forBlock['zen_number'] = function(block) {
  var num = block.getFieldValue('NUM');
  var code = String(num);
  if (code.indexOf('.') !== -1 && code.indexOf('.0') === -1) {
    // Already has decimal
  }
  return [code, ZenC.ORDER_ATOMIC];
};

// Block 6: boolean literal
Blockly.Blocks['zen_boolean'] = {
  init: function() {
    this.appendDummyInput()
        .appendField(new Blockly.FieldDropdown([
          ['true', 'true'], ['false', 'false']
        ]), 'BOOL');
    this.setOutput(true, 'Boolean');
    this.setColour(30);
    this.setTooltip('A boolean value');
  }
};

ZenC.forBlock['zen_boolean'] = function(block) {
  return [block.getFieldValue('BOOL'), ZenC.ORDER_ATOMIC];
};

// Block 7: char literal
Blockly.Blocks['zen_char'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("'")
        .appendField(new Blockly.FieldTextInput('a'), 'CHAR')
        .appendField("'");
    this.setOutput(true, 'Char');
    this.setColour(30);
    this.setTooltip('A character value');
  }
};

ZenC.forBlock['zen_char'] = function(block) {
  var ch = block.getFieldValue('CHAR');
  return ["'" + ch.charAt(0) + "'", ZenC.ORDER_ATOMIC];
};

// Block 8: string literal
Blockly.Blocks['zen_string'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('"')
        .appendField(new Blockly.FieldTextInput('hello'), 'TEXT')
        .appendField('"');
    this.setOutput(true, 'String');
    this.setColour(30);
    this.setTooltip('A string value');
  }
};

ZenC.forBlock['zen_string'] = function(block) {
  var text = block.getFieldValue('TEXT');
  return ['"' + text + '"', ZenC.ORDER_ATOMIC];
};

// Block 9: variable reference (get)
Blockly.Blocks['zen_variable_get'] = {
  init: function() {
    this.appendDummyInput()
        .appendField(new Blockly.FieldTextInput('x'), 'VAR');
    this.setOutput(true);
    this.setColour(30);
    this.setTooltip('Get the value of a variable');
  }
};

ZenC.forBlock['zen_variable_get'] = function(block) {
  return [block.getFieldValue('VAR'), ZenC.ORDER_ATOMIC];
};

// Block 10: assign variable
Blockly.Blocks['zen_assign'] = {
  init: function() {
    this.appendValueInput('VALUE')
        .appendField(new Blockly.FieldTextInput('x'), 'VAR')
        .appendField('=');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(30);
    this.setTooltip('Assign a value to a variable');
  }
};

ZenC.forBlock['zen_assign'] = function(block) {
  var name = block.getFieldValue('VAR');
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_ASSIGNMENT) || '0';
  return name + ' = ' + value + ';\n';
};

// Block 11: compound assignment
Blockly.Blocks['zen_assign_op'] = {
  init: function() {
    this.appendValueInput('VALUE')
        .appendField(new Blockly.FieldTextInput('x'), 'VAR')
        .appendField(new Blockly.FieldDropdown([
          ['+=', '+='], ['-=', '-='], ['*=', '*='],
          ['/=', '/='], ['%=', '%='], ['&=', '&='],
          ['|=', '|='], ['^=', '^=']
        ]), 'OP');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(30);
    this.setTooltip('Compound assignment operator');
  }
};

ZenC.forBlock['zen_assign_op'] = function(block) {
  var name = block.getFieldValue('VAR');
  var op = block.getFieldValue('OP');
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_ASSIGNMENT) || '1';
  return name + ' ' + op + ' ' + value + ';\n';
};

// Block 12: array declaration
Blockly.Blocks['zen_array_decl'] = {
  init: function() {
    this.appendValueInput('SIZE')
        .appendField('let')
        .appendField(new Blockly.FieldTextInput('arr'), 'VAR')
        .appendField(': [')
        .appendField(new Blockly.FieldDropdown(ZEN_TYPE_DROPDOWN), 'TYPE')
        .appendField(';');
    this.appendDummyInput()
        .appendField(']');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(30);
    this.setTooltip('Declare a fixed-size array');
  }
};

ZenC.forBlock['zen_array_decl'] = function(block) {
  var name = block.getFieldValue('VAR');
  var type = block.getFieldValue('TYPE');
  var size = ZenC.valueToCode(block, 'SIZE', ZenC.ORDER_NONE) || '10';
  return 'let ' + name + ': ' + type + '[' + size + '];\n';
};

// Block 13: array literal
Blockly.Blocks['zen_array_literal'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('[')
        .appendField(new Blockly.FieldTextInput('1, 2, 3'), 'VALUES')
        .appendField(']');
    this.setOutput(true, 'Array');
    this.setColour(30);
    this.setTooltip('An array literal');
  }
};

ZenC.forBlock['zen_array_literal'] = function(block) {
  var values = block.getFieldValue('VALUES');
  return ['[' + values + ']', ZenC.ORDER_ATOMIC];
};

// Block 14: array access
Blockly.Blocks['zen_array_access'] = {
  init: function() {
    this.appendValueInput('INDEX')
        .appendField(new Blockly.FieldTextInput('arr'), 'VAR')
        .appendField('[');
    this.appendDummyInput()
        .appendField(']');
    this.setOutput(true);
    this.setColour(30);
    this.setTooltip('Access an array element by index');
  }
};

ZenC.forBlock['zen_array_access'] = function(block) {
  var name = block.getFieldValue('VAR');
  var index = ZenC.valueToCode(block, 'INDEX', ZenC.ORDER_NONE) || '0';
  return [name + '[' + index + ']', ZenC.ORDER_MEMBER];
};

// Block 15: array set
Blockly.Blocks['zen_array_set'] = {
  init: function() {
    this.appendValueInput('INDEX')
        .appendField(new Blockly.FieldTextInput('arr'), 'VAR')
        .appendField('[');
    this.appendValueInput('VALUE')
        .appendField('] =');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(30);
    this.setTooltip('Set an array element');
  }
};

ZenC.forBlock['zen_array_set'] = function(block) {
  var name = block.getFieldValue('VAR');
  var index = ZenC.valueToCode(block, 'INDEX', ZenC.ORDER_NONE) || '0';
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_ASSIGNMENT) || '0';
  return name + '[' + index + '] = ' + value + ';\n';
};

// Block 16: type cast
Blockly.Blocks['zen_cast'] = {
  init: function() {
    this.appendValueInput('VALUE')
        .appendField('cast');
    this.appendDummyInput()
        .appendField('as')
        .appendField(new Blockly.FieldDropdown(ZEN_TYPE_DROPDOWN), 'TYPE');
    this.setOutput(true);
    this.setColour(30);
    this.setTooltip('Cast a value to a different type');
  }
};

ZenC.forBlock['zen_cast'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_UNARY_PREFIX) || '0';
  var type = block.getFieldValue('TYPE');
  return ['(' + type + ')' + value, ZenC.ORDER_UNARY_PREFIX];
};

// Block 17: sizeof
Blockly.Blocks['zen_sizeof'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('sizeof(')
        .appendField(new Blockly.FieldDropdown(ZEN_TYPE_DROPDOWN), 'TYPE')
        .appendField(')');
    this.setOutput(true, 'Number');
    this.setColour(30);
    this.setTooltip('Get the size of a type in bytes');
  }
};

ZenC.forBlock['zen_sizeof'] = function(block) {
  var type = block.getFieldValue('TYPE');
  return ['sizeof(' + type + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 18: typeof
Blockly.Blocks['zen_typeof'] = {
  init: function() {
    this.appendValueInput('VALUE')
        .appendField('typeof');
    this.setOutput(true, 'String');
    this.setColour(30);
    this.setTooltip('Get the type of a value');
  }
};

ZenC.forBlock['zen_typeof'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '0';
  return ['typeof(' + value + ')', ZenC.ORDER_FUNCTION_CALL];
};
