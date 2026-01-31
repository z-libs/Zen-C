/**
 * Zen-C Text & I/O Blocks (85-96)
 * print, println, string ops, format, raw code block
 */
'use strict';

// Block 85: println
Blockly.Blocks['zen_println'] = {
  init: function() {
    this.appendValueInput('TEXT').appendField('println');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(160);
    this.setTooltip('Print a line to stdout');
  }
};

ZenC.forBlock['zen_println'] = function(block) {
  var text = ZenC.valueToCode(block, 'TEXT', ZenC.ORDER_NONE) || '""';
  // println requires a string literal — wrap non-strings in interpolation
  if (text.charAt(0) !== '"') {
    text = '"{' + text + '}"';
  }
  return 'println ' + text + ';\n';
};

// Block 86: print (no newline)
Blockly.Blocks['zen_print'] = {
  init: function() {
    this.appendValueInput('TEXT').appendField('print');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(160);
    this.setTooltip('Print text without a newline');
  }
};

ZenC.forBlock['zen_print'] = function(block) {
  var text = ZenC.valueToCode(block, 'TEXT', ZenC.ORDER_NONE) || '""';
  // print requires a string literal — wrap non-strings in interpolation
  if (text.charAt(0) !== '"') {
    text = '"{' + text + '}"';
  }
  return 'print ' + text + ';\n';
};

// Block 87: println with format
Blockly.Blocks['zen_println_fmt'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('println')
        .appendField(new Blockly.FieldTextInput('x = {}'), 'FMT');
    this.appendValueInput('ARG1').appendField('arg1:');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(160);
    this.setInputsInline(false);
    this.setTooltip('Print formatted output');
  }
};

ZenC.forBlock['zen_println_fmt'] = function(block) {
  var fmt = block.getFieldValue('FMT');
  return 'println "' + fmt + '";\n';
};

// Block 88: string length
Blockly.Blocks['zen_strlen'] = {
  init: function() {
    this.appendValueInput('STR').appendField('length of');
    this.setOutput(true, 'Number');
    this.setColour(160);
    this.setTooltip('Get the length of a string');
  }
};

ZenC.forBlock['zen_strlen'] = function(block) {
  var str = ZenC.valueToCode(block, 'STR', ZenC.ORDER_NONE) || '""';
  return [str + '.len', ZenC.ORDER_MEMBER];
};

// Block 89: string concatenation
Blockly.Blocks['zen_strcat'] = {
  init: function() {
    this.appendValueInput('A').appendField('concat');
    this.appendValueInput('B').appendField('+');
    this.setOutput(true, 'String');
    this.setColour(160);
    this.setInputsInline(true);
    this.setTooltip('Concatenate two strings');
  }
};

ZenC.forBlock['zen_strcat'] = function(block) {
  var a = ZenC.valueToCode(block, 'A', ZenC.ORDER_ADDITION) || '""';
  var b = ZenC.valueToCode(block, 'B', ZenC.ORDER_ADDITION) || '""';
  return [a + ' ++ ' + b, ZenC.ORDER_ADDITION];
};

// Block 90: string comparison
Blockly.Blocks['zen_strcmp'] = {
  init: function() {
    this.appendValueInput('A');
    this.appendValueInput('B')
        .appendField(new Blockly.FieldDropdown([
          ['equals', '=='], ['not equals', '!=']
        ]), 'OP');
    this.setOutput(true, 'Boolean');
    this.setColour(160);
    this.setInputsInline(true);
    this.setTooltip('Compare two strings');
  }
};

ZenC.forBlock['zen_strcmp'] = function(block) {
  var op = block.getFieldValue('OP');
  var a = ZenC.valueToCode(block, 'A', ZenC.ORDER_EQUALITY) || '""';
  var b = ZenC.valueToCode(block, 'B', ZenC.ORDER_EQUALITY) || '""';
  return [a + ' ' + op + ' ' + b, ZenC.ORDER_EQUALITY];
};

// Block 91: string slice
Blockly.Blocks['zen_strslice'] = {
  init: function() {
    this.appendValueInput('STR').appendField('slice');
    this.appendValueInput('START').appendField('[');
    this.appendValueInput('END').appendField('..');
    this.appendDummyInput().appendField(']');
    this.setOutput(true, 'String');
    this.setColour(160);
    this.setInputsInline(true);
    this.setTooltip('Get a substring (slice)');
  }
};

ZenC.forBlock['zen_strslice'] = function(block) {
  var str = ZenC.valueToCode(block, 'STR', ZenC.ORDER_MEMBER) || '""';
  var start = ZenC.valueToCode(block, 'START', ZenC.ORDER_NONE) || '0';
  var end = ZenC.valueToCode(block, 'END', ZenC.ORDER_NONE) || '';
  return [str + '[' + start + '..' + end + ']', ZenC.ORDER_MEMBER];
};

// Block 92: char at index
Blockly.Blocks['zen_char_at'] = {
  init: function() {
    this.appendValueInput('STR');
    this.appendValueInput('INDEX').appendField('char at');
    this.setOutput(true, 'Char');
    this.setColour(160);
    this.setInputsInline(true);
    this.setTooltip('Get character at index');
  }
};

ZenC.forBlock['zen_char_at'] = function(block) {
  var str = ZenC.valueToCode(block, 'STR', ZenC.ORDER_MEMBER) || '""';
  var index = ZenC.valueToCode(block, 'INDEX', ZenC.ORDER_NONE) || '0';
  return [str + '[' + index + ']', ZenC.ORDER_MEMBER];
};

// Block 93: to_string
Blockly.Blocks['zen_to_string'] = {
  init: function() {
    this.appendValueInput('VALUE').appendField('to_string(');
    this.appendDummyInput().appendField(')');
    this.setOutput(true, 'String');
    this.setColour(160);
    this.setInputsInline(true);
    this.setTooltip('Convert a value to string');
  }
};

ZenC.forBlock['zen_to_string'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || '0';
  return ['to_string(' + value + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 94: parse number from string
Blockly.Blocks['zen_parse_int'] = {
  init: function() {
    this.appendValueInput('STR')
        .appendField(new Blockly.FieldDropdown([
          ['parse_int', 'parse_int'],
          ['parse_float', 'parse_float']
        ]), 'OP')
        .appendField('(');
    this.appendDummyInput().appendField(')');
    this.setOutput(true, 'Number');
    this.setColour(160);
    this.setInputsInline(true);
    this.setTooltip('Parse a number from a string');
  }
};

ZenC.forBlock['zen_parse_int'] = function(block) {
  var op = block.getFieldValue('OP');
  var str = ZenC.valueToCode(block, 'STR', ZenC.ORDER_NONE) || '"0"';
  return [op + '(' + str + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 95: raw code block
Blockly.Blocks['zen_raw_code'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('raw code:')
        .appendField(new Blockly.FieldTextInput('// your code here'), 'CODE');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(160);
    this.setTooltip('Insert raw Zen-C code');
  }
};

ZenC.forBlock['zen_raw_code'] = function(block) {
  return block.getFieldValue('CODE') + '\n';
};

// Block 96: comment
Blockly.Blocks['zen_comment'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('//')
        .appendField(new Blockly.FieldTextInput('comment'), 'TEXT');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(160);
    this.setTooltip('Add a comment');
  }
};

ZenC.forBlock['zen_comment'] = function(block) {
  return '// ' + block.getFieldValue('TEXT') + '\n';
};
