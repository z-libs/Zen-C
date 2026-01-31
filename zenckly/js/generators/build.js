/**
 * Zen-C Build/Import Blocks (169-174)
 * import, cflags, target, freestanding, inline asm
 */
'use strict';

// Block 169: import
Blockly.Blocks['zen_import'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('import')
        .appendField(new Blockly.FieldTextInput('std/io.zc'), 'PATH');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(150);
    this.setTooltip('Import a module');
  }
};

ZenC.forBlock['zen_import'] = function(block) {
  return 'import "' + block.getFieldValue('PATH') + '"\n';
};

// Block 170: cflags directive
Blockly.Blocks['zen_cflags'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('//> cflags')
        .appendField(new Blockly.FieldTextInput('-lm'), 'FLAGS');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(150);
    this.setTooltip('Compiler flags directive');
  }
};

ZenC.forBlock['zen_cflags'] = function(block) {
  return '//> cflags ' + block.getFieldValue('FLAGS') + '\n';
};

// Block 171: target directive
Blockly.Blocks['zen_target'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('//> target')
        .appendField(new Blockly.FieldDropdown([
          ['native', 'native'],
          ['x86_64-linux', 'x86_64-linux'],
          ['aarch64-linux', 'aarch64-linux'],
          ['avr-freestanding', 'avr-freestanding'],
          ['thumb-freestanding', 'thumb-freestanding'],
          ['riscv32-freestanding', 'riscv32-freestanding']
        ]), 'TARGET');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(150);
    this.setTooltip('Set compilation target');
  }
};

ZenC.forBlock['zen_target'] = function(block) {
  return '//> target ' + block.getFieldValue('TARGET') + '\n';
};

// Block 172: freestanding directive
Blockly.Blocks['zen_freestanding'] = {
  init: function() {
    this.appendDummyInput().appendField('//> freestanding');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(150);
    this.setTooltip('Enable freestanding mode (no OS)');
  }
};

ZenC.forBlock['zen_freestanding'] = function(block) {
  return '//> freestanding\n';
};

// Block 173: inline assembly
Blockly.Blocks['zen_asm'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('asm')
        .appendField(new Blockly.FieldTextInput('nop'), 'CODE');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(150);
    this.setTooltip('Inline assembly instruction');
  }
};

ZenC.forBlock['zen_asm'] = function(block) {
  return 'asm("' + block.getFieldValue('CODE') + '");\n';
};

// Block 174: linker script
Blockly.Blocks['zen_linker'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('//> linker')
        .appendField(new Blockly.FieldTextInput('link.ld'), 'SCRIPT');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(150);
    this.setTooltip('Specify a linker script');
  }
};

ZenC.forBlock['zen_linker'] = function(block) {
  return '//> linker ' + block.getFieldValue('SCRIPT') + '\n';
};
