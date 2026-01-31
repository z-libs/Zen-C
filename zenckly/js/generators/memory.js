/**
 * Zen-C Memory Blocks (75-84)
 * alloc, free, pointers, address-of, dereference
 */
'use strict';

// Block 75: alloc
Blockly.Blocks['zen_alloc'] = {
  init: function() {
    this.appendValueInput('SIZE')
        .appendField('alloc(')
        .appendField(new Blockly.FieldDropdown(ZEN_TYPE_DROPDOWN), 'TYPE')
        .appendField(',');
    this.appendDummyInput().appendField(')');
    this.setOutput(true);
    this.setColour(0);
    this.setInputsInline(true);
    this.setTooltip('Allocate heap memory');
  }
};

ZenC.forBlock['zen_alloc'] = function(block) {
  var type = block.getFieldValue('TYPE');
  var size = ZenC.valueToCode(block, 'SIZE', ZenC.ORDER_NONE) || '1';
  if (size === '1') {
    return ['alloc<' + type + '>()', ZenC.ORDER_FUNCTION_CALL];
  }
  return ['alloc_n<' + type + '>(' + size + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 76: free
Blockly.Blocks['zen_free'] = {
  init: function() {
    this.appendValueInput('PTR').appendField('free(');
    this.appendDummyInput().appendField(')');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(0);
    this.setInputsInline(true);
    this.setTooltip('Free heap memory');
  }
};

ZenC.forBlock['zen_free'] = function(block) {
  var ptr = ZenC.valueToCode(block, 'PTR', ZenC.ORDER_NONE) || 'ptr';
  return 'free(' + ptr + ');\n';
};

// Block 77: address-of (&)
Blockly.Blocks['zen_addr_of'] = {
  init: function() {
    this.appendValueInput('VALUE').appendField('&');
    this.setOutput(true);
    this.setColour(0);
    this.setTooltip('Get the address of a value');
  }
};

ZenC.forBlock['zen_addr_of'] = function(block) {
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_UNARY_PREFIX) || 'x';
  return ['&' + value, ZenC.ORDER_UNARY_PREFIX];
};

// Block 78: dereference (*)
Blockly.Blocks['zen_deref'] = {
  init: function() {
    this.appendValueInput('PTR').appendField('*');
    this.setOutput(true);
    this.setColour(0);
    this.setTooltip('Dereference a pointer');
  }
};

ZenC.forBlock['zen_deref'] = function(block) {
  var ptr = ZenC.valueToCode(block, 'PTR', ZenC.ORDER_UNARY_PREFIX) || 'ptr';
  return ['(*' + ptr + ')', ZenC.ORDER_UNARY_PREFIX];
};

// Block 79: pointer type declaration
Blockly.Blocks['zen_ptr_decl'] = {
  init: function() {
    this.appendValueInput('VALUE')
        .appendField('let')
        .appendField(new Blockly.FieldTextInput('ptr'), 'VAR')
        .appendField(': *')
        .appendField(new Blockly.FieldDropdown(ZEN_TYPE_DROPDOWN), 'TYPE')
        .appendField('=');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(0);
    this.setTooltip('Declare a pointer variable');
  }
};

ZenC.forBlock['zen_ptr_decl'] = function(block) {
  var name = block.getFieldValue('VAR');
  var type = block.getFieldValue('TYPE');
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_NONE) || 'NULL';
  return 'let ' + name + ': ' + type + '* = ' + value + ';\n';
};

// Block 80: null pointer
Blockly.Blocks['zen_null'] = {
  init: function() {
    this.appendDummyInput().appendField('null');
    this.setOutput(true);
    this.setColour(0);
    this.setTooltip('Null pointer');
  }
};

ZenC.forBlock['zen_null'] = function(block) {
  return ['NULL', ZenC.ORDER_ATOMIC];
};

// Block 81: undefined
Blockly.Blocks['zen_undefined'] = {
  init: function() {
    this.appendDummyInput().appendField('undefined');
    this.setOutput(true);
    this.setColour(0);
    this.setTooltip('Undefined value');
  }
};

ZenC.forBlock['zen_undefined'] = function(block) {
  return ['undefined', ZenC.ORDER_ATOMIC];
};

// Block 82: memcpy
Blockly.Blocks['zen_memcpy'] = {
  init: function() {
    this.appendValueInput('DEST').appendField('memcpy(');
    this.appendValueInput('SRC').appendField(',');
    this.appendValueInput('SIZE').appendField(',');
    this.appendDummyInput().appendField(')');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(0);
    this.setInputsInline(true);
    this.setTooltip('Copy memory');
  }
};

ZenC.forBlock['zen_memcpy'] = function(block) {
  var dest = ZenC.valueToCode(block, 'DEST', ZenC.ORDER_NONE) || 'dest';
  var src = ZenC.valueToCode(block, 'SRC', ZenC.ORDER_NONE) || 'src';
  var size = ZenC.valueToCode(block, 'SIZE', ZenC.ORDER_NONE) || '0';
  return 'memcpy(' + dest + ', ' + src + ', ' + size + ');\n';
};

// Block 83: memset
Blockly.Blocks['zen_memset'] = {
  init: function() {
    this.appendValueInput('DEST').appendField('memset(');
    this.appendValueInput('VAL').appendField(',');
    this.appendValueInput('SIZE').appendField(',');
    this.appendDummyInput().appendField(')');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(0);
    this.setInputsInline(true);
    this.setTooltip('Set memory to a value');
  }
};

ZenC.forBlock['zen_memset'] = function(block) {
  var dest = ZenC.valueToCode(block, 'DEST', ZenC.ORDER_NONE) || 'dest';
  var val = ZenC.valueToCode(block, 'VAL', ZenC.ORDER_NONE) || '0';
  var size = ZenC.valueToCode(block, 'SIZE', ZenC.ORDER_NONE) || '0';
  return 'memset(' + dest + ', ' + val + ', ' + size + ');\n';
};

// Block 84: slice from pointer
Blockly.Blocks['zen_ptr_slice'] = {
  init: function() {
    this.appendValueInput('PTR');
    this.appendValueInput('START').appendField('[');
    this.appendValueInput('END').appendField('..');
    this.appendDummyInput().appendField(']');
    this.setOutput(true);
    this.setColour(0);
    this.setInputsInline(true);
    this.setTooltip('Create a slice from a pointer');
  }
};

ZenC.forBlock['zen_ptr_slice'] = function(block) {
  var ptr = ZenC.valueToCode(block, 'PTR', ZenC.ORDER_MEMBER) || 'ptr';
  var start = ZenC.valueToCode(block, 'START', ZenC.ORDER_NONE) || '0';
  var end = ZenC.valueToCode(block, 'END', ZenC.ORDER_NONE) || '';
  return [ptr + '[' + start + '..' + end + ']', ZenC.ORDER_MEMBER];
};
