/**
 * Zenckly — Advanced blocks
 * Method calls, Vec, guard, def, hex, autofree, packed struct.
 */
'use strict';

// ── Method Call (value) ──────────────────────────────────────────────────
Blockly.Blocks['zen_method_call'] = {
  init: function() {
    this.appendValueInput('OBJ').setCheck(null);
    this.appendDummyInput()
        .appendField('.')
        .appendField(new Blockly.FieldTextInput('method'), 'METHOD')
        .appendField('(')
        .appendField(new Blockly.FieldTextInput(''), 'ARGS')
        .appendField(')');
    this.setInputsInline(true);
    this.setOutput(true, null);
    this.setColour(260);
    this.setTooltip('Call a method on an object (returns value)');
  }
};

ZenC.forBlock['zen_method_call'] = function(block) {
  var obj = ZenC.valueToCode(block, 'OBJ', ZenC.ORDER_MEMBER) || 'obj';
  var method = block.getFieldValue('METHOD');
  var args = block.getFieldValue('ARGS');
  return [obj + '.' + method + '(' + args + ')', ZenC.ORDER_FUNCTION_CALL];
};

// ── Method Call (statement) ──────────────────────────────────────────────
Blockly.Blocks['zen_method_call_stmt'] = {
  init: function() {
    this.appendValueInput('OBJ').setCheck(null);
    this.appendDummyInput()
        .appendField('.')
        .appendField(new Blockly.FieldTextInput('method'), 'METHOD')
        .appendField('(')
        .appendField(new Blockly.FieldTextInput(''), 'ARGS')
        .appendField(')');
    this.setInputsInline(true);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(260);
    this.setTooltip('Call a method on an object (statement)');
  }
};

ZenC.forBlock['zen_method_call_stmt'] = function(block) {
  var obj = ZenC.valueToCode(block, 'OBJ', ZenC.ORDER_MEMBER) || 'obj';
  var method = block.getFieldValue('METHOD');
  var args = block.getFieldValue('ARGS');
  return obj + '.' + method + '(' + args + ');\n';
};

// ── Static Call (value) ──────────────────────────────────────────────────
Blockly.Blocks['zen_static_call'] = {
  init: function() {
    this.appendDummyInput()
        .appendField(new Blockly.FieldTextInput('Type'), 'TYPE')
        .appendField('::')
        .appendField(new Blockly.FieldTextInput('method'), 'METHOD')
        .appendField('(')
        .appendField(new Blockly.FieldTextInput(''), 'ARGS')
        .appendField(')');
    this.setOutput(true, null);
    this.setColour(260);
    this.setTooltip('Call a static/associated method (returns value)');
  }
};

ZenC.forBlock['zen_static_call'] = function(block) {
  var type = block.getFieldValue('TYPE');
  var method = block.getFieldValue('METHOD');
  var args = block.getFieldValue('ARGS');
  return [type + '::' + method + '(' + args + ')', ZenC.ORDER_FUNCTION_CALL];
};

// ── Static Call (statement) ──────────────────────────────────────────────
Blockly.Blocks['zen_static_call_stmt'] = {
  init: function() {
    this.appendDummyInput()
        .appendField(new Blockly.FieldTextInput('Type'), 'TYPE')
        .appendField('::')
        .appendField(new Blockly.FieldTextInput('method'), 'METHOD')
        .appendField('(')
        .appendField(new Blockly.FieldTextInput(''), 'ARGS')
        .appendField(')');
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(260);
    this.setTooltip('Call a static/associated method (statement)');
  }
};

ZenC.forBlock['zen_static_call_stmt'] = function(block) {
  var type = block.getFieldValue('TYPE');
  var method = block.getFieldValue('METHOD');
  var args = block.getFieldValue('ARGS');
  return type + '::' + method + '(' + args + ');\n';
};

// ── Guard ────────────────────────────────────────────────────────────────
Blockly.Blocks['zen_guard'] = {
  init: function() {
    this.appendValueInput('COND').setCheck('Boolean').appendField('guard');
    this.appendStatementInput('ELSE').appendField('else');
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(210);
    this.setTooltip('Guard clause — if condition is false, run else body');
  }
};

ZenC.forBlock['zen_guard'] = function(block) {
  var cond = ZenC.valueToCode(block, 'COND', ZenC.ORDER_NONE) || 'true';
  var body = ZenC.statementToCode(block, 'ELSE');
  return 'guard ' + cond + ' else {\n' + body + '}\n';
};

// ── Def ──────────────────────────────────────────────────────────────────
Blockly.Blocks['zen_def'] = {
  init: function() {
    this.appendValueInput('VALUE').appendField('def')
        .appendField(new Blockly.FieldTextInput('NAME'), 'NAME')
        .appendField('=');
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(30);
    this.setTooltip('Compile-time constant definition');
  }
};

ZenC.forBlock['zen_def'] = function(block) {
  var name = block.getFieldValue('NAME');
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '0';
  return 'def ' + name + ' = ' + value + ';\n';
};

// ── Hex Number ───────────────────────────────────────────────────────────
Blockly.Blocks['zen_hex_number'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('0x')
        .appendField(new Blockly.FieldTextInput('FF'), 'HEX');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setTooltip('Hexadecimal number literal');
  }
};

ZenC.forBlock['zen_hex_number'] = function(block) {
  var hex = block.getFieldValue('HEX');
  return ['0x' + hex, ZenC.ORDER_ATOMIC];
};

// ── Vec::new() ───────────────────────────────────────────────────────────
Blockly.Blocks['zen_vec_new'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('Vec<')
        .appendField(new Blockly.FieldTextInput('int'), 'TYPE')
        .appendField('>::new()');
    this.setOutput(true, null);
    this.setColour(260);
    this.setTooltip('Create a new empty Vec');
  }
};

ZenC.forBlock['zen_vec_new'] = function(block) {
  var type = block.getFieldValue('TYPE');
  return ['Vec<' + type + '>::new()', ZenC.ORDER_FUNCTION_CALL];
};

// ── Vec push ─────────────────────────────────────────────────────────────
Blockly.Blocks['zen_vec_push'] = {
  init: function() {
    this.appendValueInput('VEC').appendField('vec');
    this.appendValueInput('VALUE').appendField('.push(');
    this.appendDummyInput().appendField(')');
    this.setInputsInline(true);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(260);
    this.setTooltip('Push a value onto a Vec');
  }
};

ZenC.forBlock['zen_vec_push'] = function(block) {
  var vec = ZenC.valueToCode(block, 'VEC', ZenC.ORDER_MEMBER) || 'vec';
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '0';
  return vec + '.push(' + value + ');\n';
};

// ── Vec pop ──────────────────────────────────────────────────────────────
Blockly.Blocks['zen_vec_pop'] = {
  init: function() {
    this.appendValueInput('VEC').appendField('vec');
    this.appendDummyInput().appendField('.pop()');
    this.setInputsInline(true);
    this.setOutput(true, null);
    this.setColour(260);
    this.setTooltip('Pop the last value from a Vec');
  }
};

ZenC.forBlock['zen_vec_pop'] = function(block) {
  var vec = ZenC.valueToCode(block, 'VEC', ZenC.ORDER_MEMBER) || 'vec';
  return [vec + '.pop()', ZenC.ORDER_FUNCTION_CALL];
};

// ── Vec len ──────────────────────────────────────────────────────────────
Blockly.Blocks['zen_vec_len'] = {
  init: function() {
    this.appendValueInput('VEC').appendField('vec');
    this.appendDummyInput().appendField('.len');
    this.setInputsInline(true);
    this.setOutput(true, 'Number');
    this.setColour(260);
    this.setTooltip('Get the length of a Vec');
  }
};

ZenC.forBlock['zen_vec_len'] = function(block) {
  var vec = ZenC.valueToCode(block, 'VEC', ZenC.ORDER_MEMBER) || 'vec';
  return [vec + '.len', ZenC.ORDER_MEMBER];
};

// ── Vec data access ──────────────────────────────────────────────────────
Blockly.Blocks['zen_vec_data_access'] = {
  init: function() {
    this.appendValueInput('VEC').appendField('vec');
    this.appendValueInput('INDEX').appendField('.data[');
    this.appendDummyInput().appendField(']');
    this.setInputsInline(true);
    this.setOutput(true, null);
    this.setColour(260);
    this.setTooltip('Access vec.data[index]');
  }
};

ZenC.forBlock['zen_vec_data_access'] = function(block) {
  var vec = ZenC.valueToCode(block, 'VEC', ZenC.ORDER_MEMBER) || 'vec';
  var index = ZenC.valueToCode(block, 'INDEX', ZenC.ORDER_NONE) || '0';
  return [vec + '.data[' + index + ']', ZenC.ORDER_MEMBER];
};

// ── Autofree ─────────────────────────────────────────────────────────────
Blockly.Blocks['zen_autofree'] = {
  init: function() {
    this.appendValueInput('VALUE')
        .appendField('autofree let')
        .appendField(new Blockly.FieldTextInput('x'), 'VAR')
        .appendField('=');
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(0);
    this.setTooltip('Declare a variable with automatic memory freeing');
  }
};

ZenC.forBlock['zen_autofree'] = function(block) {
  var name = block.getFieldValue('VAR');
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || 'null';
  return 'autofree let ' + name + ' = ' + value + ';\n';
};

// ── Packed Struct ────────────────────────────────────────────────────────
Blockly.Blocks['zen_packed_struct'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('@packed struct')
        .appendField(new Blockly.FieldTextInput('Name'), 'NAME');
    this.appendStatementInput('FIELDS');
    this.setColour(330);
    this.setTooltip('Define a packed struct (no padding between fields)');
  }
};

ZenC.forBlock['zen_packed_struct'] = function(block) {
  var name = block.getFieldValue('NAME');
  var fields = ZenC.statementToCode(block, 'FIELDS');
  var code = '@packed struct ' + name + ' {\n' + fields + '}\n\n';
  ZenC.definitions_['packed_struct_' + name] = code;
  return '';
};
