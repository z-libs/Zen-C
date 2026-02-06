/**
 * Zen-C Logic & Control Flow Blocks (35-48)
 * if/else, while, for, for-in, match, break, continue, logical ops
 */
'use strict';

// Block 35: if
Blockly.Blocks['zen_if'] = {
  init: function() {
    this.appendValueInput('COND').appendField('if');
    this.appendStatementInput('DO').appendField('then');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(210);
    this.setTooltip('If condition is true, run the enclosed blocks');
  }
};

ZenC.forBlock['zen_if'] = function(block) {
  var cond = ZenC.valueToCode(block, 'COND', ZenC.ORDER_NONE) || 'true';
  var body = ZenC.statementToCode(block, 'DO');
  return 'if ' + cond + ' {\n' + body + '}\n';
};

// Block 36: if/else
Blockly.Blocks['zen_if_else'] = {
  init: function() {
    this.appendValueInput('COND').appendField('if');
    this.appendStatementInput('DO').appendField('then');
    this.appendStatementInput('ELSE').appendField('else');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(210);
    this.setTooltip('If/else conditional');
  }
};

ZenC.forBlock['zen_if_else'] = function(block) {
  var cond = ZenC.valueToCode(block, 'COND', ZenC.ORDER_NONE) || 'true';
  var doBody = ZenC.statementToCode(block, 'DO');
  var elseBody = ZenC.statementToCode(block, 'ELSE');
  return 'if ' + cond + ' {\n' + doBody + '} else {\n' + elseBody + '}\n';
};

// Block 37: else if (chained)
Blockly.Blocks['zen_else_if'] = {
  init: function() {
    this.appendValueInput('COND').appendField('else if');
    this.appendStatementInput('DO').appendField('then');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(210);
    this.setTooltip('Else-if branch (attach after an if block)');
  }
};

ZenC.forBlock['zen_else_if'] = function(block) {
  var cond = ZenC.valueToCode(block, 'COND', ZenC.ORDER_NONE) || 'true';
  var body = ZenC.statementToCode(block, 'DO');
  return 'else if ' + cond + ' {\n' + body + '}\n';
};

// Block 38: while loop
Blockly.Blocks['zen_while'] = {
  init: function() {
    this.appendValueInput('COND').appendField('while');
    this.appendStatementInput('DO').appendField('do');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(120);
    this.setTooltip('While loop');
  }
};

ZenC.forBlock['zen_while'] = function(block) {
  var cond = ZenC.valueToCode(block, 'COND', ZenC.ORDER_NONE) || 'true';
  var body = ZenC.statementToCode(block, 'DO');
  return 'while ' + cond + ' {\n' + body + '}\n';
};

// Block 39: for loop (C-style)
Blockly.Blocks['zen_for'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('for')
        .appendField(new Blockly.FieldTextInput('i'), 'VAR')
        .appendField('from');
    this.appendValueInput('FROM');
    this.appendValueInput('TO').appendField('to');
    this.appendStatementInput('DO').appendField('do');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(120);
    this.setInputsInline(true);
    this.setTooltip('For loop with range');
  }
};

ZenC.forBlock['zen_for'] = function(block) {
  var varName = block.getFieldValue('VAR');
  var from = ZenC.valueToCode(block, 'FROM', ZenC.ORDER_NONE) || '0';
  var to = ZenC.valueToCode(block, 'TO', ZenC.ORDER_NONE) || '10';
  var body = ZenC.statementToCode(block, 'DO');
  return 'for (let ' + varName + ' = ' + from + '; ' + varName + ' < ' + to + '; ' + varName + ' += 1) {\n' + body + '}\n';
};

// Block 40: for-in (range)
Blockly.Blocks['zen_for_in'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('for')
        .appendField(new Blockly.FieldTextInput('i'), 'VAR')
        .appendField('in');
    this.appendValueInput('START');
    this.appendValueInput('END').appendField('..');
    this.appendStatementInput('DO').appendField('do');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(120);
    this.setInputsInline(true);
    this.setTooltip('Iterate over a range');
  }
};

ZenC.forBlock['zen_for_in'] = function(block) {
  var varName = block.getFieldValue('VAR');
  var start = ZenC.valueToCode(block, 'START', ZenC.ORDER_NONE) || '0';
  var end = ZenC.valueToCode(block, 'END', ZenC.ORDER_NONE) || '10';
  var body = ZenC.statementToCode(block, 'DO');
  return 'for ' + varName + ' in ' + start + '..' + end + ' {\n' + body + '}\n';
};

// Block 41: for-each (iterate collection)
Blockly.Blocks['zen_for_each'] = {
  init: function() {
    this.appendValueInput('COLLECTION')
        .appendField('for each')
        .appendField(new Blockly.FieldTextInput('item'), 'VAR')
        .appendField('in');
    this.appendStatementInput('DO').appendField('do');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(120);
    this.setTooltip('Iterate over a collection');
  }
};

ZenC.forBlock['zen_for_each'] = function(block) {
  var varName = block.getFieldValue('VAR');
  var collection = ZenC.valueToCode(block, 'COLLECTION', ZenC.ORDER_NONE) || 'arr';
  var body = ZenC.statementToCode(block, 'DO');
  return 'for ' + varName + ' in ' + collection + ' {\n' + body + '}\n';
};

// Block 42: infinite loop
Blockly.Blocks['zen_loop_forever'] = {
  init: function() {
    this.appendStatementInput('DO').appendField('loop forever');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(120);
    this.setTooltip('Infinite loop (use break to exit)');
  }
};

ZenC.forBlock['zen_loop_forever'] = function(block) {
  var body = ZenC.statementToCode(block, 'DO');
  return 'loop {\n' + body + '}\n';
};

// Block 43: break
Blockly.Blocks['zen_break'] = {
  init: function() {
    this.appendDummyInput().appendField('break');
    this.setPreviousStatement(true);
    this.setColour(120);
    this.setTooltip('Break out of a loop');
  }
};

ZenC.forBlock['zen_break'] = function(block) {
  return 'break;\n';
};

// Block 44: continue
Blockly.Blocks['zen_continue'] = {
  init: function() {
    this.appendDummyInput().appendField('continue');
    this.setPreviousStatement(true);
    this.setColour(120);
    this.setTooltip('Skip to the next iteration of a loop');
  }
};

ZenC.forBlock['zen_continue'] = function(block) {
  return 'continue;\n';
};

// Block 45: match (switch)
Blockly.Blocks['zen_match'] = {
  init: function() {
    this.appendValueInput('EXPR').appendField('match');
    this.appendStatementInput('CASES').appendField('cases');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(210);
    this.setTooltip('Match expression (like switch)');
  }
};

ZenC.forBlock['zen_match'] = function(block) {
  var expr = ZenC.valueToCode(block, 'EXPR', ZenC.ORDER_NONE) || 'x';
  var cases = ZenC.statementToCode(block, 'CASES');
  return 'match ' + expr + ' {\n' + cases + '}\n';
};

// Block 46: match case
Blockly.Blocks['zen_match_case'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('case')
        .appendField(new Blockly.FieldTextInput('0'), 'VALUE')
        .appendField(':');
    this.appendStatementInput('DO');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(210);
    this.setTooltip('A match case');
  }
};

ZenC.forBlock['zen_match_case'] = function(block) {
  var value = block.getFieldValue('VALUE');
  var body = ZenC.statementToCode(block, 'DO');
  return value + ' => {\n' + body + '},\n';
};

// Block 47: logical AND / OR
Blockly.Blocks['zen_logic_op'] = {
  init: function() {
    this.appendValueInput('A');
    this.appendValueInput('B')
        .appendField(new Blockly.FieldDropdown([
          ['and', 'and'], ['or', 'or']
        ]), 'OP');
    this.setOutput(true, 'Boolean');
    this.setColour(210);
    this.setInputsInline(true);
    this.setTooltip('Logical AND / OR');
  }
};

ZenC.forBlock['zen_logic_op'] = function(block) {
  var op = block.getFieldValue('OP');
  var order = (op === 'and') ? ZenC.ORDER_LOGICAL_AND : ZenC.ORDER_LOGICAL_OR;
  var a = ZenC.valueToCode(block, 'A', order) || 'true';
  var b = ZenC.valueToCode(block, 'B', order) || 'true';
  return [a + ' ' + op + ' ' + b, order];
};

// Block 48: logical NOT
Blockly.Blocks['zen_logic_not'] = {
  init: function() {
    this.appendValueInput('VALUE').appendField('not');
    this.setOutput(true, 'Boolean');
    this.setColour(210);
    this.setTooltip('Logical NOT');
  }
};

ZenC.forBlock['zen_logic_not'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_UNARY_PREFIX) || 'true';
  return ['!' + value, ZenC.ORDER_UNARY_PREFIX];
};
