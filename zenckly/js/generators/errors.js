/**
 * Zen-C Error Handling Blocks (97-104)
 * Result, Option, unwrap, try, Ok, Err, Some, None
 */
'use strict';

// Block 97: Ok value
Blockly.Blocks['zen_ok'] = {
  init: function() {
    this.appendValueInput('VALUE').appendField('Ok(');
    this.appendDummyInput().appendField(')');
    this.setOutput(true);
    this.setColour(45);
    this.setInputsInline(true);
    this.setTooltip('Wrap a value in Ok');
  }
};

ZenC.forBlock['zen_ok'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '0';
  return ['Ok(' + value + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 98: Err value
Blockly.Blocks['zen_err'] = {
  init: function() {
    this.appendValueInput('VALUE').appendField('Err(');
    this.appendDummyInput().appendField(')');
    this.setOutput(true);
    this.setColour(45);
    this.setInputsInline(true);
    this.setTooltip('Wrap a value in Err');
  }
};

ZenC.forBlock['zen_err'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '"error"';
  return ['Err(' + value + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 99: Some value
Blockly.Blocks['zen_some'] = {
  init: function() {
    this.appendValueInput('VALUE').appendField('Some(');
    this.appendDummyInput().appendField(')');
    this.setOutput(true);
    this.setColour(45);
    this.setInputsInline(true);
    this.setTooltip('Wrap a value in Some');
  }
};

ZenC.forBlock['zen_some'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '0';
  return ['Some(' + value + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 100: None
Blockly.Blocks['zen_none'] = {
  init: function() {
    this.appendDummyInput().appendField('None');
    this.setOutput(true);
    this.setColour(45);
    this.setTooltip('No value (None)');
  }
};

ZenC.forBlock['zen_none'] = function(block) {
  return ['None', ZenC.ORDER_ATOMIC];
};

// Block 101: unwrap
Blockly.Blocks['zen_unwrap'] = {
  init: function() {
    this.appendValueInput('VALUE').appendField('unwrap(');
    this.appendDummyInput().appendField(')');
    this.setOutput(true);
    this.setColour(45);
    this.setInputsInline(true);
    this.setTooltip('Unwrap a Result or Option (panics on error/None)');
  }
};

ZenC.forBlock['zen_unwrap'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_MEMBER) || 'result';
  return [value + '.unwrap()', ZenC.ORDER_FUNCTION_CALL];
};

// Block 102: unwrap_or
Blockly.Blocks['zen_unwrap_or'] = {
  init: function() {
    this.appendValueInput('VALUE');
    this.appendValueInput('DEFAULT').appendField('.unwrap_or(');
    this.appendDummyInput().appendField(')');
    this.setOutput(true);
    this.setColour(45);
    this.setInputsInline(true);
    this.setTooltip('Unwrap with a default value');
  }
};

ZenC.forBlock['zen_unwrap_or'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_MEMBER) || 'result';
  var def = ZenC.valueToCode(block, 'DEFAULT', ZenC.ORDER_NONE) || '0';
  return [value + '.unwrap_or(' + def + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 103: try (? operator)
Blockly.Blocks['zen_try'] = {
  init: function() {
    this.appendValueInput('VALUE').appendField('try');
    this.setOutput(true);
    this.setColour(45);
    this.setTooltip('Try operator — propagate errors');
  }
};

ZenC.forBlock['zen_try'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || 'result';
  return [value + '?', ZenC.ORDER_UNARY_PREFIX];
};

// Block 104: if-let unwrap pattern
Blockly.Blocks['zen_if_let'] = {
  init: function() {
    this.appendValueInput('EXPR')
        .appendField('if let')
        .appendField(new Blockly.FieldTextInput('value'), 'VAR')
        .appendField('=');
    this.appendStatementInput('DO').appendField('then');
    this.appendStatementInput('ELSE').appendField('else');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(45);
    this.setTooltip('Unwrap with if-let pattern');
  }
};

ZenC.forBlock['zen_if_let'] = function(block) {
  var varName = block.getFieldValue('VAR');
  var expr = ZenC.valueToCode(block, 'EXPR', ZenC.ORDER_NONE) || 'opt';
  var doBody = ZenC.statementToCode(block, 'DO');
  var elseBody = ZenC.statementToCode(block, 'ELSE');
  var code = 'if (' + expr + ') |' + varName + '| {\n' + doBody + '}';
  if (elseBody) {
    code += ' else {\n' + elseBody + '}';
  }
  return code + '\n';
};
