/**
 * Zen-C Function Blocks (49-60)
 * fn, call, lambda, defer, extern, main, return, params
 */
'use strict';

// Block 49: main function
Blockly.Blocks['zen_main'] = {
  init: function() {
    this.appendStatementInput('BODY').appendField('fn main()');
    this.setColour(290);
    this.setTooltip('Main entry point function');
    this.setDeletable(false);
  }
};

ZenC.forBlock['zen_main'] = function(block) {
  var body = ZenC.statementToCode(block, 'BODY');
  return 'fn main() {\n' + body + '}\n';
};

// Block 50: function definition
Blockly.Blocks['zen_fn'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('fn')
        .appendField(new Blockly.FieldTextInput('myFunc'), 'NAME')
        .appendField('(')
        .appendField(new Blockly.FieldTextInput(''), 'PARAMS')
        .appendField(')');
    this.appendDummyInput()
        .appendField('->')
        .appendField(new Blockly.FieldTextInput('void'), 'RETTYPE');
    this.appendStatementInput('BODY');
    this.setColour(290);
    this.setTooltip('Define a function');
  }
};

ZenC.forBlock['zen_fn'] = function(block) {
  var name = block.getFieldValue('NAME');
  var params = block.getFieldValue('PARAMS');
  var retType = block.getFieldValue('RETTYPE');
  var body = ZenC.statementToCode(block, 'BODY');
  var sig = 'fn ' + name + '(' + params + ')';
  if (retType !== 'void') {
    sig += ' -> ' + retType;
  }
  return sig + ' {\n' + body + '}\n';
};

// Block 51: function with pub
Blockly.Blocks['zen_fn_pub'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('pub fn')
        .appendField(new Blockly.FieldTextInput('myFunc'), 'NAME')
        .appendField('(')
        .appendField(new Blockly.FieldTextInput(''), 'PARAMS')
        .appendField(') ->')
        .appendField(new Blockly.FieldTextInput('void'), 'RETTYPE');
    this.appendStatementInput('BODY');
    this.setColour(290);
    this.setTooltip('Define a public function');
  }
};

ZenC.forBlock['zen_fn_pub'] = function(block) {
  var name = block.getFieldValue('NAME');
  var params = block.getFieldValue('PARAMS');
  var retType = block.getFieldValue('RETTYPE');
  var body = ZenC.statementToCode(block, 'BODY');
  var sig = 'pub fn ' + name + '(' + params + ')';
  if (retType !== 'void') {
    sig += ' -> ' + retType;
  }
  return sig + ' {\n' + body + '}\n';
};

// Block 52: return
Blockly.Blocks['zen_return'] = {
  init: function() {
    this.appendValueInput('VALUE').appendField('return');
    this.setPreviousStatement(true);
    this.setColour(290);
    this.setTooltip('Return a value from a function');
  }
};

ZenC.forBlock['zen_return'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE);
  if (value) {
    return 'return ' + value + ';\n';
  }
  return 'return;\n';
};

// Block 53: return void
Blockly.Blocks['zen_return_void'] = {
  init: function() {
    this.appendDummyInput().appendField('return');
    this.setPreviousStatement(true);
    this.setColour(290);
    this.setTooltip('Return from a void function');
  }
};

ZenC.forBlock['zen_return_void'] = function(block) {
  return 'return;\n';
};

// Block 54: function call
Blockly.Blocks['zen_call'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('call')
        .appendField(new Blockly.FieldTextInput('myFunc'), 'NAME')
        .appendField('(')
        .appendField(new Blockly.FieldTextInput(''), 'ARGS')
        .appendField(')');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(290);
    this.setTooltip('Call a function (statement)');
  }
};

ZenC.forBlock['zen_call'] = function(block) {
  var name = block.getFieldValue('NAME');
  var args = block.getFieldValue('ARGS');
  return name + '(' + args + ');\n';
};

// Block 55: function call (expression)
Blockly.Blocks['zen_call_expr'] = {
  init: function() {
    this.appendDummyInput()
        .appendField(new Blockly.FieldTextInput('myFunc'), 'NAME')
        .appendField('(')
        .appendField(new Blockly.FieldTextInput(''), 'ARGS')
        .appendField(')');
    this.setOutput(true);
    this.setColour(290);
    this.setTooltip('Call a function (expression that returns a value)');
  }
};

ZenC.forBlock['zen_call_expr'] = function(block) {
  var name = block.getFieldValue('NAME');
  var args = block.getFieldValue('ARGS');
  return [name + '(' + args + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 56: lambda / anonymous function
Blockly.Blocks['zen_lambda'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('fn(')
        .appendField(new Blockly.FieldTextInput('x: int'), 'PARAMS')
        .appendField(') ->');
    this.appendValueInput('BODY');
    this.setOutput(true);
    this.setColour(290);
    this.setInputsInline(true);
    this.setTooltip('Anonymous function / lambda');
  }
};

ZenC.forBlock['zen_lambda'] = function(block) {
  var params = block.getFieldValue('PARAMS');
  var body = ZenC.valueToCode(block, 'BODY', ZenC.ORDER_NONE) || '0';
  return ['fn(' + params + ') -> ' + body, ZenC.ORDER_ATOMIC];
};

// Block 57: defer
Blockly.Blocks['zen_defer'] = {
  init: function() {
    this.appendStatementInput('BODY').appendField('defer');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(290);
    this.setTooltip('Defer execution until scope exit');
  }
};

ZenC.forBlock['zen_defer'] = function(block) {
  var body = ZenC.statementToCode(block, 'BODY');
  return 'defer {\n' + body + '}\n';
};

// Block 58: extern function declaration
Blockly.Blocks['zen_extern'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('extern fn')
        .appendField(new Blockly.FieldTextInput('printf'), 'NAME')
        .appendField('(')
        .appendField(new Blockly.FieldTextInput('fmt: *const u8, ...'), 'PARAMS')
        .appendField(') ->')
        .appendField(new Blockly.FieldDropdown([
          ['void', 'void'], ['int', 'int'], ['u32', 'u32']
        ]), 'RETTYPE');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(290);
    this.setTooltip('Declare an external function');
  }
};

ZenC.forBlock['zen_extern'] = function(block) {
  var name = block.getFieldValue('NAME');
  var params = block.getFieldValue('PARAMS');
  var retType = block.getFieldValue('RETTYPE');
  var sig = 'extern fn ' + name + '(' + params + ')';
  if (retType !== 'void') {
    sig += ' -> ' + retType;
  }
  return sig + ';\n';
};

// Block 59: comptime block
Blockly.Blocks['zen_comptime'] = {
  init: function() {
    this.appendStatementInput('BODY').appendField('comptime');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(290);
    this.setTooltip('Execute at compile time');
  }
};

ZenC.forBlock['zen_comptime'] = function(block) {
  var body = ZenC.statementToCode(block, 'BODY');
  return 'comptime {\n' + body + '}\n';
};

// Block 60: function parameter block (value)
Blockly.Blocks['zen_param'] = {
  init: function() {
    this.appendDummyInput()
        .appendField(new Blockly.FieldTextInput('x'), 'NAME')
        .appendField(':')
        .appendField(new Blockly.FieldDropdown(ZEN_TYPE_DROPDOWN), 'TYPE');
    this.setOutput(true);
    this.setColour(290);
    this.setTooltip('Function parameter');
  }
};

ZenC.forBlock['zen_param'] = function(block) {
  var name = block.getFieldValue('NAME');
  var type = block.getFieldValue('TYPE');
  return [name + ': ' + type, ZenC.ORDER_ATOMIC];
};
