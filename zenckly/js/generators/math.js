/**
 * Zen-C Math Blocks (19-34)
 * Arithmetic, comparison, trig, bitwise, random, etc.
 */
'use strict';

// Block 19: arithmetic operator
Blockly.Blocks['zen_math_arithmetic'] = {
  init: function() {
    this.appendValueInput('A');
    this.appendValueInput('B')
        .appendField(new Blockly.FieldDropdown([
          ['+', '+'], ['-', '-'], ['*', '*'],
          ['/', '/'], ['%', '%']
        ]), 'OP');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setInputsInline(true);
    this.setTooltip('Arithmetic operation');
  }
};

ZenC.forBlock['zen_math_arithmetic'] = function(block) {
  var op = block.getFieldValue('OP');
  var order = (op === '*' || op === '/' || op === '%') ?
    ZenC.ORDER_MULTIPLICATION : ZenC.ORDER_ADDITION;
  var a = ZenC.valueToCode(block, 'A', order) || '0';
  var b = ZenC.valueToCode(block, 'B', order) || '0';
  return [a + ' ' + op + ' ' + b, order];
};

// Block 20: comparison
Blockly.Blocks['zen_math_compare'] = {
  init: function() {
    this.appendValueInput('A');
    this.appendValueInput('B')
        .appendField(new Blockly.FieldDropdown([
          ['==', '=='], ['!=', '!='], ['<', '<'],
          ['>', '>'], ['<=', '<='], ['>=', '>=']
        ]), 'OP');
    this.setOutput(true, 'Boolean');
    this.setColour(230);
    this.setInputsInline(true);
    this.setTooltip('Compare two values');
  }
};

ZenC.forBlock['zen_math_compare'] = function(block) {
  var op = block.getFieldValue('OP');
  var order = (op === '==' || op === '!=') ?
    ZenC.ORDER_EQUALITY : ZenC.ORDER_RELATIONAL;
  var a = ZenC.valueToCode(block, 'A', order) || '0';
  var b = ZenC.valueToCode(block, 'B', order) || '0';
  return [a + ' ' + op + ' ' + b, order];
};

// Block 21: unary negation
Blockly.Blocks['zen_math_negate'] = {
  init: function() {
    this.appendValueInput('VALUE')
        .appendField('-');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setTooltip('Negate a number');
  }
};

ZenC.forBlock['zen_math_negate'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_UNARY_PREFIX) || '0';
  return ['-' + value, ZenC.ORDER_UNARY_PREFIX];
};

// Block 22: math constant
Blockly.Blocks['zen_math_constant'] = {
  init: function() {
    this.appendDummyInput()
        .appendField(new Blockly.FieldDropdown([
          ['PI', '3.14159265358979'],
          ['E', '2.71828182845905'],
          ['TAU', '6.28318530717959'],
          ['INF', '1.0/0.0']
        ]), 'CONST');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setTooltip('Mathematical constant');
  }
};

ZenC.forBlock['zen_math_constant'] = function(block) {
  return [block.getFieldValue('CONST'), ZenC.ORDER_ATOMIC];
};

// Block 23: abs / min / max / clamp
Blockly.Blocks['zen_math_single'] = {
  init: function() {
    this.appendValueInput('VALUE')
        .appendField(new Blockly.FieldDropdown([
          ['abs', 'abs'], ['sqrt', 'sqrt'],
          ['floor', 'floor'], ['ceil', 'ceil'],
          ['round', 'round']
        ]), 'OP')
        .appendField('(');
    this.appendDummyInput().appendField(')');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setInputsInline(true);
    this.setTooltip('Single-argument math function');
  }
};

ZenC.forBlock['zen_math_single'] = function(block) {
  var op = block.getFieldValue('OP');
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '0';
  return ['@' + op + '(' + value + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 24: min/max (two args)
Blockly.Blocks['zen_math_minmax'] = {
  init: function() {
    this.appendValueInput('A')
        .appendField(new Blockly.FieldDropdown([
          ['min', 'min'], ['max', 'max']
        ]), 'OP')
        .appendField('(');
    this.appendValueInput('B').appendField(',');
    this.appendDummyInput().appendField(')');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setInputsInline(true);
    this.setTooltip('Min or max of two values');
  }
};

ZenC.forBlock['zen_math_minmax'] = function(block) {
  var op = block.getFieldValue('OP');
  var a = ZenC.valueToCode(block, 'A', ZenC.ORDER_NONE) || '0';
  var b = ZenC.valueToCode(block, 'B', ZenC.ORDER_NONE) || '0';
  return ['@' + op + '(' + a + ', ' + b + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 25: trig functions
Blockly.Blocks['zen_math_trig'] = {
  init: function() {
    this.appendValueInput('VALUE')
        .appendField(new Blockly.FieldDropdown([
          ['sin', 'sin'], ['cos', 'cos'], ['tan', 'tan'],
          ['asin', 'asin'], ['acos', 'acos'], ['atan', 'atan']
        ]), 'OP')
        .appendField('(');
    this.appendDummyInput().appendField(')');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setInputsInline(true);
    this.setTooltip('Trigonometric function');
  }
};

ZenC.forBlock['zen_math_trig'] = function(block) {
  var op = block.getFieldValue('OP');
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '0';
  return ['@' + op + '(' + value + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 26: power
Blockly.Blocks['zen_math_pow'] = {
  init: function() {
    this.appendValueInput('BASE').appendField('pow(');
    this.appendValueInput('EXP').appendField(',');
    this.appendDummyInput().appendField(')');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setInputsInline(true);
    this.setTooltip('Raise base to the power of exponent');
  }
};

ZenC.forBlock['zen_math_pow'] = function(block) {
  var base = ZenC.valueToCode(block, 'BASE', ZenC.ORDER_NONE) || '0';
  var exp = ZenC.valueToCode(block, 'EXP', ZenC.ORDER_NONE) || '1';
  return ['@pow(' + base + ', ' + exp + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 27: log
Blockly.Blocks['zen_math_log'] = {
  init: function() {
    this.appendValueInput('VALUE')
        .appendField(new Blockly.FieldDropdown([
          ['ln', 'log'], ['log2', 'log2'], ['log10', 'log10']
        ]), 'OP')
        .appendField('(');
    this.appendDummyInput().appendField(')');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setInputsInline(true);
    this.setTooltip('Logarithmic function');
  }
};

ZenC.forBlock['zen_math_log'] = function(block) {
  var op = block.getFieldValue('OP');
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '1';
  return ['@' + op + '(' + value + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 28: bitwise operators
Blockly.Blocks['zen_math_bitwise'] = {
  init: function() {
    this.appendValueInput('A');
    this.appendValueInput('B')
        .appendField(new Blockly.FieldDropdown([
          ['&', '&'], ['|', '|'], ['^', '^'],
          ['<<', '<<'], ['>>', '>>']
        ]), 'OP');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setInputsInline(true);
    this.setTooltip('Bitwise operation');
  }
};

ZenC.forBlock['zen_math_bitwise'] = function(block) {
  var op = block.getFieldValue('OP');
  var orderMap = {
    '&': ZenC.ORDER_BITWISE_AND,
    '|': ZenC.ORDER_BITWISE_OR,
    '^': ZenC.ORDER_BITWISE_XOR,
    '<<': ZenC.ORDER_SHIFT,
    '>>': ZenC.ORDER_SHIFT
  };
  var order = orderMap[op];
  var a = ZenC.valueToCode(block, 'A', order) || '0';
  var b = ZenC.valueToCode(block, 'B', order) || '0';
  return [a + ' ' + op + ' ' + b, order];
};

// Block 29: bitwise NOT
Blockly.Blocks['zen_math_bitnot'] = {
  init: function() {
    this.appendValueInput('VALUE').appendField('~');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setTooltip('Bitwise NOT');
  }
};

ZenC.forBlock['zen_math_bitnot'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_UNARY_PREFIX) || '0';
  return ['~' + value, ZenC.ORDER_UNARY_PREFIX];
};

// Block 30: random integer
Blockly.Blocks['zen_math_random_int'] = {
  init: function() {
    this.appendValueInput('MIN').appendField('random int from');
    this.appendValueInput('MAX').appendField('to');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setInputsInline(true);
    this.setTooltip('Generate a random integer in range');
  }
};

ZenC.forBlock['zen_math_random_int'] = function(block) {
  var min = ZenC.valueToCode(block, 'MIN', ZenC.ORDER_NONE) || '0';
  var max = ZenC.valueToCode(block, 'MAX', ZenC.ORDER_NONE) || '100';
  return ['random_int(' + min + ', ' + max + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 31: random float
Blockly.Blocks['zen_math_random_float'] = {
  init: function() {
    this.appendDummyInput().appendField('random float 0.0..1.0');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setTooltip('Generate a random float between 0.0 and 1.0');
  }
};

ZenC.forBlock['zen_math_random_float'] = function(block) {
  return ['random_float()', ZenC.ORDER_FUNCTION_CALL];
};

// Block 32: modulo
Blockly.Blocks['zen_math_modulo'] = {
  init: function() {
    this.appendValueInput('A');
    this.appendValueInput('B').appendField('mod');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setInputsInline(true);
    this.setTooltip('Modulo operation');
  }
};

ZenC.forBlock['zen_math_modulo'] = function(block) {
  var a = ZenC.valueToCode(block, 'A', ZenC.ORDER_MULTIPLICATION) || '0';
  var b = ZenC.valueToCode(block, 'B', ZenC.ORDER_MULTIPLICATION) || '1';
  return [a + ' % ' + b, ZenC.ORDER_MULTIPLICATION];
};

// Block 33: integer increment/decrement
Blockly.Blocks['zen_math_incdec'] = {
  init: function() {
    this.appendDummyInput()
        .appendField(new Blockly.FieldTextInput('x'), 'VAR')
        .appendField(new Blockly.FieldDropdown([
          ['++', '+='], ['--', '-=']
        ]), 'OP');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(230);
    this.setTooltip('Increment or decrement a variable');
  }
};

ZenC.forBlock['zen_math_incdec'] = function(block) {
  var name = block.getFieldValue('VAR');
  var op = block.getFieldValue('OP');
  return name + ' ' + op + ' 1;\n';
};

// Block 34: clamp
Blockly.Blocks['zen_math_clamp'] = {
  init: function() {
    this.appendValueInput('VALUE').appendField('clamp(');
    this.appendValueInput('MIN').appendField(',');
    this.appendValueInput('MAX').appendField(',');
    this.appendDummyInput().appendField(')');
    this.setOutput(true, 'Number');
    this.setColour(230);
    this.setInputsInline(true);
    this.setTooltip('Clamp a value between min and max');
  }
};

ZenC.forBlock['zen_math_clamp'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '0';
  var min = ZenC.valueToCode(block, 'MIN', ZenC.ORDER_NONE) || '0';
  var max = ZenC.valueToCode(block, 'MAX', ZenC.ORDER_NONE) || '100';
  return ['@clamp(' + value + ', ' + min + ', ' + max + ')', ZenC.ORDER_FUNCTION_CALL];
};
