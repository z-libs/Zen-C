/**
 * Zen-C Struct, Impl, Trait, Enum, Generics Blocks (61-74)
 */
'use strict';

// Block 61: struct definition
Blockly.Blocks['zen_struct'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('struct')
        .appendField(new Blockly.FieldTextInput('Point'), 'NAME');
    this.appendStatementInput('FIELDS').appendField('fields');
    this.setColour(330);
    this.setTooltip('Define a struct');
  }
};

ZenC.forBlock['zen_struct'] = function(block) {
  var name = block.getFieldValue('NAME');
  var fields = ZenC.statementToCode(block, 'FIELDS');
  return 'struct ' + name + ' {\n' + fields + '}\n';
};

// Block 62: struct field
Blockly.Blocks['zen_struct_field'] = {
  init: function() {
    this.appendDummyInput()
        .appendField(new Blockly.FieldTextInput('x'), 'NAME')
        .appendField(':')
        .appendField(new Blockly.FieldTextInput('int'), 'TYPE');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(330);
    this.setTooltip('Struct field');
  }
};

ZenC.forBlock['zen_struct_field'] = function(block) {
  var name = block.getFieldValue('NAME');
  var type = block.getFieldValue('TYPE');
  return name + ': ' + type + ';\n';
};

// Block 63: struct instantiation
Blockly.Blocks['zen_struct_init'] = {
  init: function() {
    this.appendDummyInput()
        .appendField(new Blockly.FieldTextInput('Point'), 'NAME')
        .appendField('{');
    this.appendDummyInput()
        .appendField(new Blockly.FieldTextInput('x: 0, y: 0'), 'FIELDS')
        .appendField('}');
    this.setOutput(true);
    this.setColour(330);
    this.setTooltip('Create a struct instance');
  }
};

ZenC.forBlock['zen_struct_init'] = function(block) {
  var name = block.getFieldValue('NAME');
  var fields = block.getFieldValue('FIELDS');
  return [name + '{ ' + fields + ' }', ZenC.ORDER_ATOMIC];
};

// Block 64: field access
Blockly.Blocks['zen_field_access'] = {
  init: function() {
    this.appendValueInput('OBJ');
    this.appendDummyInput()
        .appendField('.')
        .appendField(new Blockly.FieldTextInput('x'), 'FIELD');
    this.setOutput(true);
    this.setColour(330);
    this.setInputsInline(true);
    this.setTooltip('Access a struct field');
  }
};

ZenC.forBlock['zen_field_access'] = function(block) {
  var obj = ZenC.valueToCode(block, 'OBJ', ZenC.ORDER_MEMBER) || 'self';
  var field = block.getFieldValue('FIELD');
  return [obj + '.' + field, ZenC.ORDER_MEMBER];
};

// Block 65: field set
Blockly.Blocks['zen_field_set'] = {
  init: function() {
    this.appendValueInput('OBJ');
    this.appendValueInput('VALUE')
        .appendField('.')
        .appendField(new Blockly.FieldTextInput('x'), 'FIELD')
        .appendField('=');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(330);
    this.setInputsInline(true);
    this.setTooltip('Set a struct field');
  }
};

ZenC.forBlock['zen_field_set'] = function(block) {
  var obj = ZenC.valueToCode(block, 'OBJ', ZenC.ORDER_MEMBER) || 'self';
  var field = block.getFieldValue('FIELD');
  var value = ZenC.valueToCode(block, 'VALUE', ZenC.ORDER_ASSIGNMENT) || '0';
  return obj + '.' + field + ' = ' + value + ';\n';
};

// Block 66: impl block
Blockly.Blocks['zen_impl'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('impl')
        .appendField(new Blockly.FieldTextInput('Point'), 'NAME');
    this.appendStatementInput('METHODS').appendField('methods');
    this.setColour(330);
    this.setTooltip('Implement methods for a struct');
  }
};

ZenC.forBlock['zen_impl'] = function(block) {
  var name = block.getFieldValue('NAME');
  var methods = ZenC.statementToCode(block, 'METHODS');
  return 'impl ' + name + ' {\n' + methods + '}\n';
};

// Block 67: method definition (inside impl)
Blockly.Blocks['zen_method'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('fn')
        .appendField(new Blockly.FieldTextInput('init'), 'NAME')
        .appendField('(self')
        .appendField(new Blockly.FieldTextInput(''), 'EXTRA_PARAMS')
        .appendField(') ->')
        .appendField(new Blockly.FieldTextInput('void'), 'RETTYPE');
    this.appendStatementInput('BODY');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(330);
    this.setTooltip('Define a method');
  }
};

ZenC.forBlock['zen_method'] = function(block) {
  var name = block.getFieldValue('NAME');
  var extra = block.getFieldValue('EXTRA_PARAMS');
  var retType = block.getFieldValue('RETTYPE');
  var body = ZenC.statementToCode(block, 'BODY');
  var params = 'self' + (extra ? ', ' + extra : '');
  var sig = 'fn ' + name + '(' + params + ')';
  if (retType !== 'void') {
    sig += ' -> ' + retType;
  }
  return sig + ' {\n' + body + '}\n';
};

// Block 68: self reference
Blockly.Blocks['zen_self'] = {
  init: function() {
    this.appendDummyInput().appendField('self');
    this.setOutput(true);
    this.setColour(330);
    this.setTooltip('Reference to self');
  }
};

ZenC.forBlock['zen_self'] = function(block) {
  return ['self', ZenC.ORDER_ATOMIC];
};

// Block 69: trait definition
Blockly.Blocks['zen_trait'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('trait')
        .appendField(new Blockly.FieldTextInput('Drawable'), 'NAME');
    this.appendStatementInput('METHODS').appendField('methods');
    this.setColour(330);
    this.setTooltip('Define a trait');
  }
};

ZenC.forBlock['zen_trait'] = function(block) {
  var name = block.getFieldValue('NAME');
  var methods = ZenC.statementToCode(block, 'METHODS');
  return 'trait ' + name + ' {\n' + methods + '}\n';
};

// Block 70: enum definition
Blockly.Blocks['zen_enum'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('enum')
        .appendField(new Blockly.FieldTextInput('Color'), 'NAME');
    this.appendStatementInput('VARIANTS').appendField('variants');
    this.setColour(330);
    this.setTooltip('Define an enum');
  }
};

ZenC.forBlock['zen_enum'] = function(block) {
  var name = block.getFieldValue('NAME');
  var variants = ZenC.statementToCode(block, 'VARIANTS');
  return 'enum ' + name + ' {\n' + variants + '}\n';
};

// Block 71: enum variant
Blockly.Blocks['zen_enum_variant'] = {
  init: function() {
    this.appendDummyInput()
        .appendField(new Blockly.FieldTextInput('Red'), 'NAME');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(330);
    this.setTooltip('Enum variant');
  }
};

ZenC.forBlock['zen_enum_variant'] = function(block) {
  return block.getFieldValue('NAME') + ',\n';
};

// Block 72: enum variant with value
Blockly.Blocks['zen_enum_variant_val'] = {
  init: function() {
    this.appendDummyInput()
        .appendField(new Blockly.FieldTextInput('Code'), 'NAME')
        .appendField('=')
        .appendField(new Blockly.FieldTextInput('0'), 'VALUE');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(330);
    this.setTooltip('Enum variant with an explicit value');
  }
};

ZenC.forBlock['zen_enum_variant_val'] = function(block) {
  return block.getFieldValue('NAME') + ' = ' + block.getFieldValue('VALUE') + ',\n';
};

// Block 73: type alias
Blockly.Blocks['zen_type_alias'] = {
  init: function() {
    this.appendDummyInput()
        .appendField('type')
        .appendField(new Blockly.FieldTextInput('MyInt'), 'NAME')
        .appendField('=')
        .appendField(new Blockly.FieldTextInput('int'), 'TYPE');
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setColour(330);
    this.setTooltip('Define a type alias');
  }
};

ZenC.forBlock['zen_type_alias'] = function(block) {
  return 'type ' + block.getFieldValue('NAME') + ' = ' + block.getFieldValue('TYPE') + ';\n';
};

// Block 74: generic type usage
Blockly.Blocks['zen_generic'] = {
  init: function() {
    this.appendDummyInput()
        .appendField(new Blockly.FieldTextInput('Vec'), 'NAME')
        .appendField('<')
        .appendField(new Blockly.FieldDropdown(ZEN_TYPE_DROPDOWN), 'TYPE')
        .appendField('>');
    this.setOutput(true);
    this.setColour(330);
    this.setTooltip('Generic type');
  }
};

ZenC.forBlock['zen_generic'] = function(block) {
  var name = block.getFieldValue('NAME');
  var type = block.getFieldValue('TYPE');
  return [name + '<' + type + '>', ZenC.ORDER_ATOMIC];
};
