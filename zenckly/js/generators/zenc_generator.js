/**
 * Zen-C code generator for Blockly (v12 compatible).
 * Base generator class with operator precedence and scaffolding.
 */
'use strict';

var ZenC = new Blockly.CodeGenerator('ZenC');

// Operator precedence (higher number = binds looser)
ZenC.ORDER_ATOMIC = 0;           // Literals, identifiers
ZenC.ORDER_MEMBER = 1;           // . ->
ZenC.ORDER_FUNCTION_CALL = 2;    // ()
ZenC.ORDER_UNARY_PREFIX = 3;     // ! - ~ & *
ZenC.ORDER_MULTIPLICATION = 4;   // * / %
ZenC.ORDER_ADDITION = 5;         // + -
ZenC.ORDER_SHIFT = 6;            // << >>
ZenC.ORDER_RELATIONAL = 7;       // < > <= >=
ZenC.ORDER_EQUALITY = 8;         // == !=
ZenC.ORDER_BITWISE_AND = 9;
ZenC.ORDER_BITWISE_XOR = 10;
ZenC.ORDER_BITWISE_OR = 11;
ZenC.ORDER_LOGICAL_AND = 12;
ZenC.ORDER_LOGICAL_OR = 13;
ZenC.ORDER_ASSIGNMENT = 14;      // =
ZenC.ORDER_NONE = 99;

/**
 * Initialise the generator. Called before code generation starts.
 */
ZenC.init = function(workspace) {
  if (Blockly.Names) {
    ZenC.nameDB_ = new Blockly.Names(ZenC.RESERVED_WORDS_);
    if (workspace.getVariableMap) {
      ZenC.nameDB_.setVariableMap(workspace.getVariableMap());
    }
  }
  ZenC.definitions_ = Object.create(null);
  ZenC.functionDefinitions_ = Object.create(null);
};

/**
 * Finalise the generated code.
 */
ZenC.finish = function(code) {
  var definitions = [];
  for (var name in ZenC.definitions_) {
    definitions.push(ZenC.definitions_[name]);
  }
  var funcDefs = [];
  for (var name in ZenC.functionDefinitions_) {
    funcDefs.push(ZenC.functionDefinitions_[name]);
  }
  var prefix = '';
  if (definitions.length) {
    prefix += definitions.join('\n') + '\n\n';
  }
  if (funcDefs.length) {
    prefix += funcDefs.join('\n\n') + '\n\n';
  }

  // Clean up temporary data.
  delete ZenC.definitions_;
  delete ZenC.functionDefinitions_;
  if (ZenC.nameDB_) {
    ZenC.nameDB_.reset();
  }
  return prefix + code;
};

/**
 * Naked values are top-level blocks with outputs that aren't plugged into
 * anything. We turn them into expression statements.
 */
ZenC.scrubNakedValue = function(line) {
  return line + ';\n';
};

/**
 * Common tasks for generating code from blocks. Handles comments and
 * next-block recursion.
 */
ZenC.scrub_ = function(block, code, opt_thisOnly) {
  var commentCode = '';
  if (!block.outputConnection) {
    var comment = block.getCommentText();
    if (comment) {
      commentCode += ZenC.prefixLines(comment, '// ') + '\n';
    }
  }
  var nextBlock = block.nextConnection && block.nextConnection.targetBlock();
  var nextCode = opt_thisOnly ? '' : ZenC.blockToCode(nextBlock);
  return commentCode + code + nextCode;
};

/**
 * Reserved words list for Zen-C.
 */
ZenC.RESERVED_WORDS_ =
  'let,const,var,static,mut,fn,return,if,else,while,for,in,match,break,' +
  'continue,struct,impl,trait,enum,type,pub,import,defer,extern,comptime,' +
  'true,false,null,undefined,int,float,double,bool,char,void,u8,u16,u32,' +
  'u64,i8,i16,i32,i64,f32,f64,usize,isize,alloc,free,sizeof,typeof,' +
  'as,try,catch,Result,Option,Some,None,Ok,Err,self,Self,cflags,target';

ZenC.COMMENT_WRAP = 80;
