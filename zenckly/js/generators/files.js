/**
 * Zen-C File I/O & System Blocks (105-114)
 * fopen, fclose, fread, fwrite, system, etc.
 */
'use strict';

// Block 105: file open
Blockly.Blocks['zen_fopen'] = {
  init: function() {
    this.appendValueInput('PATH')
        .appendField('fopen(');
    this.appendDummyInput()
        .appendField(',')
        .appendField(new Blockly.FieldDropdown([
          ['read', '"r"'], ['write', '"w"'], ['append', '"a"'],
          ['read+write', '"r+"'], ['binary read', '"rb"'],
          ['binary write', '"wb"']
        ]), 'MODE')
        .appendField(')');
    this.setOutput(true);
    this.setColour(180);
    this.setInputsInline(true);
    this.setTooltip('Open a file');
  }
};

ZenC.forBlock['zen_fopen'] = function(block) {
  var path = ZenC.valueToCode(block, 'PATH', ZenC.ORDER_NONE) || '"file.txt"';
  var mode = block.getFieldValue('MODE');
  return ['fopen(' + path + ', ' + mode + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 106: file close
Blockly.Blocks['zen_fclose'] = {
  init: function() {
    this.appendValueInput('FILE').appendField('fclose(');
    this.appendDummyInput().appendField(')');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(180);
    this.setInputsInline(true);
    this.setTooltip('Close a file');
  }
};

ZenC.forBlock['zen_fclose'] = function(block) {
  var file = ZenC.valueToCode(block, 'FILE', ZenC.ORDER_NONE) || 'file';
  return 'fclose(' + file + ');\n';
};

// Block 107: file read
Blockly.Blocks['zen_fread'] = {
  init: function() {
    this.appendValueInput('FILE').appendField('fread(');
    this.appendValueInput('SIZE').appendField(',');
    this.appendDummyInput().appendField(')');
    this.setOutput(true);
    this.setColour(180);
    this.setInputsInline(true);
    this.setTooltip('Read from a file');
  }
};

ZenC.forBlock['zen_fread'] = function(block) {
  var file = ZenC.valueToCode(block, 'FILE', ZenC.ORDER_NONE) || 'file';
  var size = ZenC.valueToCode(block, 'SIZE', ZenC.ORDER_NONE) || '1024';
  return ['fread(' + file + ', ' + size + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 108: file write
Blockly.Blocks['zen_fwrite'] = {
  init: function() {
    this.appendValueInput('FILE').appendField('fwrite(');
    this.appendValueInput('DATA').appendField(',');
    this.appendDummyInput().appendField(')');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(180);
    this.setInputsInline(true);
    this.setTooltip('Write to a file');
  }
};

ZenC.forBlock['zen_fwrite'] = function(block) {
  var file = ZenC.valueToCode(block, 'FILE', ZenC.ORDER_NONE) || 'file';
  var data = ZenC.valueToCode(block, 'DATA', ZenC.ORDER_NONE) || '""';
  return 'fwrite(' + file + ', ' + data + ');\n';
};

// Block 109: file read line
Blockly.Blocks['zen_freadline'] = {
  init: function() {
    this.appendValueInput('FILE').appendField('readline(');
    this.appendDummyInput().appendField(')');
    this.setOutput(true);
    this.setColour(180);
    this.setInputsInline(true);
    this.setTooltip('Read a line from a file');
  }
};

ZenC.forBlock['zen_freadline'] = function(block) {
  var file = ZenC.valueToCode(block, 'FILE', ZenC.ORDER_NONE) || 'file';
  return ['readline(' + file + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 110: file exists
Blockly.Blocks['zen_file_exists'] = {
  init: function() {
    this.appendValueInput('PATH').appendField('file_exists(');
    this.appendDummyInput().appendField(')');
    this.setOutput(true, 'Boolean');
    this.setColour(180);
    this.setInputsInline(true);
    this.setTooltip('Check if a file exists');
  }
};

ZenC.forBlock['zen_file_exists'] = function(block) {
  var path = ZenC.valueToCode(block, 'PATH', ZenC.ORDER_NONE) || '"file.txt"';
  return ['file_exists(' + path + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 111: read entire file
Blockly.Blocks['zen_read_file'] = {
  init: function() {
    this.appendValueInput('PATH').appendField('read_file(');
    this.appendDummyInput().appendField(')');
    this.setOutput(true, 'String');
    this.setColour(180);
    this.setInputsInline(true);
    this.setTooltip('Read an entire file into a string');
  }
};

ZenC.forBlock['zen_read_file'] = function(block) {
  var path = ZenC.valueToCode(block, 'PATH', ZenC.ORDER_NONE) || '"file.txt"';
  return ['read_file(' + path + ')', ZenC.ORDER_FUNCTION_CALL];
};

// Block 112: write entire file
Blockly.Blocks['zen_write_file'] = {
  init: function() {
    this.appendValueInput('PATH').appendField('write_file(');
    this.appendValueInput('DATA').appendField(',');
    this.appendDummyInput().appendField(')');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(180);
    this.setInputsInline(true);
    this.setTooltip('Write a string to a file');
  }
};

ZenC.forBlock['zen_write_file'] = function(block) {
  var path = ZenC.valueToCode(block, 'PATH', ZenC.ORDER_NONE) || '"file.txt"';
  var data = ZenC.valueToCode(block, 'DATA', ZenC.ORDER_NONE) || '""';
  return 'write_file(' + path + ', ' + data + ');\n';
};

// Block 113: system command
Blockly.Blocks['zen_system'] = {
  init: function() {
    this.appendValueInput('CMD').appendField('system(');
    this.appendDummyInput().appendField(')');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(180);
    this.setInputsInline(true);
    this.setTooltip('Execute a system command');
  }
};

ZenC.forBlock['zen_system'] = function(block) {
  var cmd = ZenC.valueToCode(block, 'CMD', ZenC.ORDER_NONE) || '""';
  return 'system(' + cmd + ');\n';
};

// Block 114: stdin read
Blockly.Blocks['zen_stdin_read'] = {
  init: function() {
    this.appendDummyInput().appendField('read_stdin()');
    this.setOutput(true, 'String');
    this.setColour(180);
    this.setTooltip('Read a line from standard input');
  }
};

ZenC.forBlock['zen_stdin_read'] = function(block) {
  return ['read_stdin()', ZenC.ORDER_FUNCTION_CALL];
};
