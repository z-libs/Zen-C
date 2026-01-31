/**
 * Zenckly — Main application logic
 * Workspace init, code generation, target selector, save/load, examples.
 * Compatible with Blockly v12+.
 */
'use strict';

(function() {

  // --- Globals ---
  var workspace = null;
  var currentTarget = 'linux';

  // --- Blockly Theme ---
  var zenTheme = new Blockly.Theme('zenckly_dark', {}, {}, {
    workspaceBackgroundColour: '#1e1e2e',
    toolboxBackgroundColour: '#181825',
    toolboxForegroundColour: '#cdd6f4',
    flyoutBackgroundColour: '#1a1a2e',
    flyoutForegroundColour: '#cdd6f4',
    flyoutOpacity: 0.95,
    scrollbarColour: '#313244',
    scrollbarOpacity: 0.6,
    insertionMarkerColour: '#89b4fa',
    insertionMarkerOpacity: 0.5
  });
  zenTheme.setFontStyle({
    family: '"Segoe UI", system-ui, sans-serif',
    size: 12
  });

  // --- Init Workspace ---
  function initWorkspace() {
    workspace = Blockly.inject('blockly-div', {
      toolbox: ZENCKLY_TOOLBOX,
      theme: zenTheme,
      grid: {
        spacing: 25,
        length: 3,
        colour: '#2a2a3e',
        snap: true
      },
      zoom: {
        controls: true,
        wheel: true,
        startScale: 0.9,
        maxScale: 2,
        minScale: 0.3,
        scaleSpeed: 1.1
      },
      trashcan: true,
      move: {
        scrollbars: true,
        drag: true,
        wheel: true
      },
      renderer: 'zelos',
      sounds: false
    });

    // Listen for workspace changes to regenerate code
    workspace.addChangeListener(onWorkspaceChange);

    // Add a default main block
    addDefaultMainBlock();
  }

  // --- Add default main block ---
  function addDefaultMainBlock() {
    var mainBlock = workspace.newBlock('zen_main');
    mainBlock.initSvg();
    mainBlock.render();
    mainBlock.moveBy(50, 50);
  }

  // --- Target Directives ---
  var TARGET_DIRECTIVES = {
    linux: ''
  };

  // --- Generate Code ---
  var suppressCodeGen = false; // Temporarily suppress code generation during example loading

  function onWorkspaceChange(event) {
    if (event.isUiEvent) return;
    if (suppressCodeGen) {
      // Only resume code generation when user actually creates or moves a block
      if (event.type === Blockly.Events.BLOCK_CREATE ||
          event.type === Blockly.Events.BLOCK_MOVE ||
          event.type === Blockly.Events.BLOCK_CHANGE) {
        suppressCodeGen = false;
      } else {
        return;
      }
    }
    generateCode();
  }

  function generateCode() {
    if (!workspace) return '';
    try {
      var code = ZenC.workspaceToCode(workspace);
      var directives = TARGET_DIRECTIVES[currentTarget] || '';
      var fullCode = directives + code;
      document.getElementById('code-text').textContent = fullCode || '// Drag blocks to generate Zen-C code';
      return fullCode;
    } catch (e) {
      console.error('Code generation error:', e);
      document.getElementById('code-text').textContent = '// Error generating code: ' + e.message;
      return '';
    }
  }

  // --- Copy Code ---
  function copyCode() {
    var code = document.getElementById('code-text').textContent;
    navigator.clipboard.writeText(code).then(function() {
      setStatus('Code copied to clipboard');
      setTimeout(function() { setStatus('Ready'); }, 2000);
    }).catch(function() {
      // Fallback: select + copy
      var el = document.getElementById('code-text');
      var range = document.createRange();
      range.selectNodeContents(el);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      document.execCommand('copy');
      sel.removeAllRanges();
      setStatus('Code copied to clipboard');
      setTimeout(function() { setStatus('Ready'); }, 2000);
    });
  }

  // --- Save Workspace ---
  function saveWorkspace() {
    try {
      var state = Blockly.serialization.workspaces.save(workspace);
      var data = {
        target: currentTarget,
        workspace: state
      };
      localStorage.setItem('zenckly_workspace', JSON.stringify(data));
      setStatus('Workspace saved');
      setTimeout(function() { setStatus('Ready'); }, 2000);
    } catch (e) {
      setStatus('Error saving: ' + e.message);
    }
  }

  // --- Load Workspace ---
  function loadWorkspace() {
    var dataStr = localStorage.getItem('zenckly_workspace');
    if (!dataStr) {
      setStatus('No saved workspace found');
      setTimeout(function() { setStatus('Ready'); }, 2000);
      return;
    }
    try {
      var data = JSON.parse(dataStr);
      workspace.clear();
      if (data.workspace && typeof data.workspace === 'object') {
        // New JSON serialization format
        Blockly.serialization.workspaces.load(data.workspace, workspace);
      } else if (data.workspace && typeof data.workspace === 'string') {
        // Legacy XML format fallback
        var xml = Blockly.utils.xml.textToDom(data.workspace);
        Blockly.Xml.domToWorkspace(xml, workspace);
      }
      if (data.target) {
        currentTarget = data.target;
        document.getElementById('target-select').value = currentTarget;
      }
      generateCode();
      setStatus('Workspace loaded');
      setTimeout(function() { setStatus('Ready'); }, 2000);
    } catch (e) {
      setStatus('Error loading workspace: ' + e.message);
    }
  }

  // --- Output Panel ---
  function showOutput(text) {
    var panel = document.getElementById('output-panel');
    var outputText = document.getElementById('output-text');
    outputText.textContent = text;
    panel.classList.remove('hidden');
    Blockly.svgResize(workspace);
  }

  function appendOutput(text) {
    var outputText = document.getElementById('output-text');
    outputText.textContent += text;
    outputText.scrollTop = outputText.scrollHeight;
  }

  function hideOutput() {
    document.getElementById('output-panel').classList.add('hidden');
    Blockly.svgResize(workspace);
  }

  // --- Get current code from preview (always fresh) ---
  function getCurrentCode() {
    // Regenerate to ensure code-text is up to date, then read it
    generateCode();
    return document.getElementById('code-text').textContent;
  }

  // --- Compile ---
  function compile() {
    var code = getCurrentCode();
    if (!code.trim() || code.indexOf('// Drag blocks') === 0) {
      setStatus('Nothing to compile');
      return;
    }

    // Electron native compile
    if (window.zenckly) {
      setStatus('Compiling...');
      showOutput('Compiling...\n');
      window.zenckly.compile(code).then(function(result) {
        showOutput(result.output);
        setStatus(result.success ? 'Compilation successful' : 'Compilation failed');
      });
      return;
    }

    // Browser fallback — copy code
    setStatus('Compiling for ' + currentTarget + '...');
    setTimeout(function() {
      setStatus('Compile complete (code copied — paste into .zc file and run: zc build)');
      copyCode();
    }, 500);
  }

  // --- Run ---
  function run() {
    var code = getCurrentCode();
    if (!code.trim() || code.indexOf('// Drag blocks') === 0) {
      setStatus('Nothing to run');
      return;
    }

    // Electron native run
    if (window.zenckly) {
      setStatus('Compiling & running...');
      showOutput('Compiling & running...\n');
      window.zenckly.run(code).then(function(result) {
        showOutput(result.output);
        setStatus(result.success ? 'Program finished successfully' : 'Program exited with error');
      });
      return;
    }

    // Browser fallback
    compile();
  }

  // --- Status Bar ---
  function setStatus(msg) {
    document.getElementById('status-text').textContent = msg;
  }

  // --- Resizable Divider ---
  function initDivider() {
    var divider = document.getElementById('divider');
    var codePanel = document.getElementById('code-panel');
    var dragging = false;

    divider.addEventListener('mousedown', function(e) {
      dragging = true;
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    });

    document.addEventListener('mousemove', function(e) {
      if (!dragging) return;
      var containerRight = document.getElementById('main').getBoundingClientRect().right;
      var newWidth = containerRight - e.clientX;
      if (newWidth < 200) newWidth = 200;
      if (newWidth > 800) newWidth = 800;
      codePanel.style.width = newWidth + 'px';
      Blockly.svgResize(workspace);
    });

    document.addEventListener('mouseup', function() {
      if (dragging) {
        dragging = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        Blockly.svgResize(workspace);
      }
    });
  }

  // --- Example Projects (build blocks programmatically) ---

  // Helper: create a block, init SVG, return it
  function makeBlock(type) {
    var b = workspace.newBlock(type);
    b.initSvg();
    return b;
  }

  // Helper: connect a value block to a parent's value input
  function connectValue(parent, inputName, child) {
    var input = parent.getInput(inputName);
    if (input && input.connection && child.outputConnection) {
      input.connection.connect(child.outputConnection);
    }
  }

  // Helper: connect a statement block to a parent's statement input
  function connectStatement(parent, inputName, child) {
    var input = parent.getInput(inputName);
    if (input && input.connection && child.previousConnection) {
      input.connection.connect(child.previousConnection);
    }
  }

  // Helper: chain two statement blocks (second after first)
  function chainBlocks(first, second) {
    if (first.nextConnection && second.previousConnection) {
      first.nextConnection.connect(second.previousConnection);
    }
  }

  // --- Example builders ---

  function buildHelloWorld() {
    var main = makeBlock('zen_main');
    main.moveBy(50, 50);

    var println = makeBlock('zen_println');
    var str = makeBlock('zen_string');
    str.setFieldValue('Hello, World!', 'TEXT');
    connectValue(println, 'TEXT', str);

    connectStatement(main, 'BODY', println);

    main.render();
  }

  function buildFibonacci() {
    // fn fibonacci(n: int) -> int
    var fn = makeBlock('zen_fn');
    fn.moveBy(50, 50);
    fn.setFieldValue('fibonacci', 'NAME');
    fn.setFieldValue('n: int', 'PARAMS');
    fn.setFieldValue('int', 'RETTYPE');

    // if/else
    var ifElse = makeBlock('zen_if_else');

    // condition: n <= 1
    var cmp = makeBlock('zen_math_compare');
    var nVar1 = makeBlock('zen_variable_get');
    nVar1.setFieldValue('n', 'VAR');
    var one = makeBlock('zen_number');
    one.setFieldValue(1, 'NUM');
    connectValue(cmp, 'A', nVar1);
    connectValue(cmp, 'B', one);
    cmp.setFieldValue('<=', 'OP');
    connectValue(ifElse, 'COND', cmp);

    // then: return n
    var ret1 = makeBlock('zen_return');
    var nVar2 = makeBlock('zen_variable_get');
    nVar2.setFieldValue('n', 'VAR');
    connectValue(ret1, 'VALUE', nVar2);
    connectStatement(ifElse, 'DO', ret1);

    // else: return fibonacci(n-1) + fibonacci(n-2)
    var ret2 = makeBlock('zen_return');
    var add = makeBlock('zen_math_arithmetic');
    add.setFieldValue('+', 'OP');
    var call1 = makeBlock('zen_call_expr');
    call1.setFieldValue('fibonacci', 'NAME');
    call1.setFieldValue('n - 1', 'ARGS');
    var call2 = makeBlock('zen_call_expr');
    call2.setFieldValue('fibonacci', 'NAME');
    call2.setFieldValue('n - 2', 'ARGS');
    connectValue(add, 'A', call1);
    connectValue(add, 'B', call2);
    connectValue(ret2, 'VALUE', add);
    connectStatement(ifElse, 'ELSE', ret2);

    connectStatement(fn, 'BODY', ifElse);
    fn.render();

    // fn main
    var main = makeBlock('zen_main');
    main.moveBy(50, 350);

    var letVar = makeBlock('zen_let');
    letVar.setFieldValue('result', 'VAR');
    letVar.setFieldValue('int', 'TYPE');
    var callFib = makeBlock('zen_call_expr');
    callFib.setFieldValue('fibonacci', 'NAME');
    callFib.setFieldValue('10', 'ARGS');
    connectValue(letVar, 'VALUE', callFib);

    var println = makeBlock('zen_println');
    var resultVar = makeBlock('zen_variable_get');
    resultVar.setFieldValue('result', 'VAR');
    connectValue(println, 'TEXT', resultVar);

    connectStatement(main, 'BODY', letVar);
    chainBlocks(letVar, println);

    main.render();
  }

  function buildStructExample() {
    // struct Point { x: int, y: int }
    var struct = makeBlock('zen_struct');
    struct.moveBy(50, 50);
    struct.setFieldValue('Point', 'NAME');

    var field1 = makeBlock('zen_struct_field');
    field1.setFieldValue('x', 'NAME');
    field1.setFieldValue('int', 'TYPE');
    var field2 = makeBlock('zen_struct_field');
    field2.setFieldValue('y', 'NAME');
    field2.setFieldValue('int', 'TYPE');

    connectStatement(struct, 'FIELDS', field1);
    chainBlocks(field1, field2);
    struct.render();

    // fn main
    var main = makeBlock('zen_main');
    main.moveBy(50, 250);

    var letVar = makeBlock('zen_let_infer');
    letVar.setFieldValue('p', 'VAR');
    var structInit = makeBlock('zen_struct_init');
    structInit.setFieldValue('Point', 'NAME');
    structInit.setFieldValue('x: 10, y: 20', 'FIELDS');
    connectValue(letVar, 'VALUE', structInit);

    var println = makeBlock('zen_println');
    var str = makeBlock('zen_string');
    str.setFieldValue('Point: ({p.x}, {p.y})', 'TEXT');
    connectValue(println, 'TEXT', str);

    connectStatement(main, 'BODY', letVar);
    chainBlocks(letVar, println);

    main.render();
  }

  function buildFileIo() {
    // import must be top-level
    var importBlock = makeBlock('zen_raw_code');
    importBlock.moveBy(50, 20);
    importBlock.setFieldValue('import "std/fs.zc"', 'CODE');
    importBlock.render();

    var main = makeBlock('zen_main');
    main.moveBy(50, 70);

    // let res = File::open("output.txt", "w")
    var letRes = makeBlock('zen_let_infer');
    letRes.setFieldValue('res', 'VAR');
    var fileOpen = makeBlock('zen_static_call');
    fileOpen.setFieldValue('File', 'TYPE');
    fileOpen.setFieldValue('open', 'METHOD');
    fileOpen.setFieldValue('"output.txt", "w"', 'ARGS');
    connectValue(letRes, 'VALUE', fileOpen);

    // if res.is_err() { println "Failed"; return; }
    var ifErr = makeBlock('zen_if');
    var isErr = makeBlock('zen_method_call');
    var resRef1 = makeBlock('zen_variable_get');
    resRef1.setFieldValue('res', 'VAR');
    connectValue(isErr, 'OBJ', resRef1);
    isErr.setFieldValue('is_err', 'METHOD');
    isErr.setFieldValue('', 'ARGS');
    connectValue(ifErr, 'COND', isErr);

    var printFail = makeBlock('zen_println');
    var failStr = makeBlock('zen_string');
    failStr.setFieldValue('Failed to open file', 'TEXT');
    connectValue(printFail, 'TEXT', failStr);
    var retVoid1 = makeBlock('zen_return_void');
    connectStatement(ifErr, 'DO', printFail);
    chainBlocks(printFail, retVoid1);

    // let f = res.unwrap()
    var letF = makeBlock('zen_let_infer');
    letF.setFieldValue('f', 'VAR');
    var unwrap = makeBlock('zen_method_call');
    var resRef2 = makeBlock('zen_variable_get');
    resRef2.setFieldValue('res', 'VAR');
    connectValue(unwrap, 'OBJ', resRef2);
    unwrap.setFieldValue('unwrap', 'METHOD');
    unwrap.setFieldValue('', 'ARGS');
    connectValue(letF, 'VALUE', unwrap);

    // f.write_string("Hello from Zen-C!\n")
    var writeStmt = makeBlock('zen_method_call_stmt');
    var fRef1 = makeBlock('zen_variable_get');
    fRef1.setFieldValue('f', 'VAR');
    connectValue(writeStmt, 'OBJ', fRef1);
    writeStmt.setFieldValue('write_string', 'METHOD');
    writeStmt.setFieldValue('"Hello from Zen-C!\\n"', 'ARGS');

    // f.close()
    var closeStmt = makeBlock('zen_method_call_stmt');
    var fRef2 = makeBlock('zen_variable_get');
    fRef2.setFieldValue('f', 'VAR');
    connectValue(closeStmt, 'OBJ', fRef2);
    closeStmt.setFieldValue('close', 'METHOD');
    closeStmt.setFieldValue('', 'ARGS');

    // println "File written successfully"
    var printOk = makeBlock('zen_println');
    var okStr = makeBlock('zen_string');
    okStr.setFieldValue('File written successfully', 'TEXT');
    connectValue(printOk, 'TEXT', okStr);

    // Chain everything in main
    connectStatement(main, 'BODY', letRes);
    chainBlocks(letRes, ifErr);
    chainBlocks(ifErr, letF);
    chainBlocks(letF, writeStmt);
    chainBlocks(writeStmt, closeStmt);
    chainBlocks(closeStmt, printOk);

    main.render();
  }

  // ===== Block Tutorial Examples =====

  // --- Variables & Math ---
  function buildBlocksVariables() {
    var main = makeBlock('zen_main');
    main.moveBy(50, 50);

    // let x: int = 42
    var letBlock = makeBlock('zen_let');
    letBlock.setFieldValue('x', 'VAR');
    letBlock.setFieldValue('int', 'TYPE');
    var num42 = makeBlock('zen_number');
    num42.setFieldValue(42, 'NUM');
    connectValue(letBlock, 'VALUE', num42);

    // let name = "Zen-C"  (type inferred)
    var letInfer = makeBlock('zen_let_infer');
    letInfer.setFieldValue('name', 'VAR');
    var str = makeBlock('zen_string');
    str.setFieldValue('Zen-C', 'TEXT');
    connectValue(letInfer, 'VALUE', str);

    // let flag: bool = true
    var letBool = makeBlock('zen_let');
    letBool.setFieldValue('flag', 'VAR');
    letBool.setFieldValue('bool', 'TYPE');
    var boolTrue = makeBlock('zen_boolean');
    boolTrue.setFieldValue('true', 'BOOL');
    connectValue(letBool, 'VALUE', boolTrue);

    // let sum = x + 10
    var letSum = makeBlock('zen_let_infer');
    letSum.setFieldValue('sum', 'VAR');
    var add = makeBlock('zen_math_arithmetic');
    add.setFieldValue('+', 'OP');
    var getX = makeBlock('zen_variable_get');
    getX.setFieldValue('x', 'VAR');
    var num10 = makeBlock('zen_number');
    num10.setFieldValue(10, 'NUM');
    connectValue(add, 'A', getX);
    connectValue(add, 'B', num10);
    connectValue(letSum, 'VALUE', add);

    // x = x * 2  (assign)
    var assign = makeBlock('zen_assign');
    assign.setFieldValue('x', 'VAR');
    var mul = makeBlock('zen_math_arithmetic');
    mul.setFieldValue('*', 'OP');
    var getX2 = makeBlock('zen_variable_get');
    getX2.setFieldValue('x', 'VAR');
    var num2 = makeBlock('zen_number');
    num2.setFieldValue(2, 'NUM');
    connectValue(mul, 'A', getX2);
    connectValue(mul, 'B', num2);
    connectValue(assign, 'VALUE', mul);

    // println "x = {x}, sum = {sum}"
    var println = makeBlock('zen_println');
    var msgStr = makeBlock('zen_string');
    msgStr.setFieldValue('x = {x}, sum = {sum}, name = {name}, flag = {flag}', 'TEXT');
    connectValue(println, 'TEXT', msgStr);

    connectStatement(main, 'BODY', letBlock);
    chainBlocks(letBlock, letInfer);
    chainBlocks(letInfer, letBool);
    chainBlocks(letBool, letSum);
    chainBlocks(letSum, assign);
    chainBlocks(assign, println);

    main.render();
  }

  // --- Logic & Loops ---
  function buildBlocksLogic() {
    var main = makeBlock('zen_main');
    main.moveBy(50, 50);

    // let x: int = 15
    var letX = makeBlock('zen_let');
    letX.setFieldValue('x', 'VAR');
    letX.setFieldValue('int', 'TYPE');
    var num15 = makeBlock('zen_number');
    num15.setFieldValue(15, 'NUM');
    connectValue(letX, 'VALUE', num15);

    // if x > 10 { println "big" } else { println "small" }
    var ifElse = makeBlock('zen_if_else');
    var cmp = makeBlock('zen_math_compare');
    cmp.setFieldValue('>', 'OP');
    var getX = makeBlock('zen_variable_get');
    getX.setFieldValue('x', 'VAR');
    var num10 = makeBlock('zen_number');
    num10.setFieldValue(10, 'NUM');
    connectValue(cmp, 'A', getX);
    connectValue(cmp, 'B', num10);
    connectValue(ifElse, 'COND', cmp);

    var printBig = makeBlock('zen_println');
    var strBig = makeBlock('zen_string');
    strBig.setFieldValue('{x} is big', 'TEXT');
    connectValue(printBig, 'TEXT', strBig);
    connectStatement(ifElse, 'DO', printBig);

    var printSmall = makeBlock('zen_println');
    var strSmall = makeBlock('zen_string');
    strSmall.setFieldValue('{x} is small', 'TEXT');
    connectValue(printSmall, 'TEXT', strSmall);
    connectStatement(ifElse, 'ELSE', printSmall);

    // for i in 0..5 { print "{i} " }
    var forIn = makeBlock('zen_for_in');
    forIn.setFieldValue('i', 'VAR');
    var start0 = makeBlock('zen_number');
    start0.setFieldValue(0, 'NUM');
    var end5 = makeBlock('zen_number');
    end5.setFieldValue(5, 'NUM');
    connectValue(forIn, 'START', start0);
    connectValue(forIn, 'END', end5);

    var printI = makeBlock('zen_print');
    var strI = makeBlock('zen_string');
    strI.setFieldValue('{i} ', 'TEXT');
    connectValue(printI, 'TEXT', strI);
    connectStatement(forIn, 'DO', printI);

    var newline = makeBlock('zen_println');
    var emptyStr = makeBlock('zen_string');
    emptyStr.setFieldValue('', 'TEXT');
    connectValue(newline, 'TEXT', emptyStr);

    // while x > 0 { x = x - 3 }
    var whileLoop = makeBlock('zen_while');
    var cmpW = makeBlock('zen_math_compare');
    cmpW.setFieldValue('>', 'OP');
    var getX2 = makeBlock('zen_variable_get');
    getX2.setFieldValue('x', 'VAR');
    var num0 = makeBlock('zen_number');
    num0.setFieldValue(0, 'NUM');
    connectValue(cmpW, 'A', getX2);
    connectValue(cmpW, 'B', num0);
    connectValue(whileLoop, 'COND', cmpW);

    var assignOp = makeBlock('zen_assign_op');
    assignOp.setFieldValue('x', 'VAR');
    assignOp.setFieldValue('-=', 'OP');
    var num3 = makeBlock('zen_number');
    num3.setFieldValue(3, 'NUM');
    connectValue(assignOp, 'VALUE', num3);
    connectStatement(whileLoop, 'DO', assignOp);

    var printFinal = makeBlock('zen_println');
    var strFinal = makeBlock('zen_string');
    strFinal.setFieldValue('x after loop: {x}', 'TEXT');
    connectValue(printFinal, 'TEXT', strFinal);

    connectStatement(main, 'BODY', letX);
    chainBlocks(letX, ifElse);
    chainBlocks(ifElse, forIn);
    chainBlocks(forIn, newline);
    chainBlocks(newline, whileLoop);
    chainBlocks(whileLoop, printFinal);

    main.render();
  }

  // --- Functions & Calls ---
  function buildBlocksFunctions() {
    // fn add(a: int, b: int) -> int
    var addFn = makeBlock('zen_fn');
    addFn.moveBy(50, 50);
    addFn.setFieldValue('add', 'NAME');
    addFn.setFieldValue('a: int, b: int', 'PARAMS');
    addFn.setFieldValue('int', 'RETTYPE');

    var retAdd = makeBlock('zen_return');
    var addExpr = makeBlock('zen_math_arithmetic');
    addExpr.setFieldValue('+', 'OP');
    var aVar = makeBlock('zen_variable_get');
    aVar.setFieldValue('a', 'VAR');
    var bVar = makeBlock('zen_variable_get');
    bVar.setFieldValue('b', 'VAR');
    connectValue(addExpr, 'A', aVar);
    connectValue(addExpr, 'B', bVar);
    connectValue(retAdd, 'VALUE', addExpr);
    connectStatement(addFn, 'BODY', retAdd);
    addFn.render();

    // fn greet(name: char*)
    var greetFn = makeBlock('zen_fn');
    greetFn.moveBy(50, 200);
    greetFn.setFieldValue('greet', 'NAME');
    greetFn.setFieldValue('name: char*', 'PARAMS');
    greetFn.setFieldValue('void', 'RETTYPE');

    var printGreet = makeBlock('zen_println');
    var greetStr = makeBlock('zen_string');
    greetStr.setFieldValue('Hello, {name}!', 'TEXT');
    connectValue(printGreet, 'TEXT', greetStr);
    connectStatement(greetFn, 'BODY', printGreet);
    greetFn.render();

    // fn main
    var main = makeBlock('zen_main');
    main.moveBy(50, 370);

    // let result = add(10, 20)
    var letResult = makeBlock('zen_let_infer');
    letResult.setFieldValue('result', 'VAR');
    var callAdd = makeBlock('zen_call_expr');
    callAdd.setFieldValue('add', 'NAME');
    callAdd.setFieldValue('10, 20', 'ARGS');
    connectValue(letResult, 'VALUE', callAdd);

    var printResult = makeBlock('zen_println');
    var resStr = makeBlock('zen_string');
    resStr.setFieldValue('10 + 20 = {result}', 'TEXT');
    connectValue(printResult, 'TEXT', resStr);

    // greet("World")
    var callGreet = makeBlock('zen_call');
    callGreet.setFieldValue('greet', 'NAME');
    callGreet.setFieldValue('"World"', 'ARGS');

    // defer { println "Done!" }
    var deferBlock = makeBlock('zen_defer');
    var printDone = makeBlock('zen_println');
    var doneStr = makeBlock('zen_string');
    doneStr.setFieldValue('Done!', 'TEXT');
    connectValue(printDone, 'TEXT', doneStr);
    connectStatement(deferBlock, 'BODY', printDone);

    connectStatement(main, 'BODY', deferBlock);
    chainBlocks(deferBlock, letResult);
    chainBlocks(letResult, printResult);
    chainBlocks(printResult, callGreet);

    main.render();
  }

  // --- Structs & Methods ---
  function buildBlocksStructs() {
    // struct Point
    var structBlock = makeBlock('zen_struct');
    structBlock.moveBy(50, 50);
    structBlock.setFieldValue('Point', 'NAME');

    var f1 = makeBlock('zen_struct_field');
    f1.setFieldValue('x', 'NAME');
    f1.setFieldValue('int', 'TYPE');
    var f2 = makeBlock('zen_struct_field');
    f2.setFieldValue('y', 'NAME');
    f2.setFieldValue('int', 'TYPE');
    connectStatement(structBlock, 'FIELDS', f1);
    chainBlocks(f1, f2);
    structBlock.render();

    // impl Point
    var implBlock = makeBlock('zen_impl');
    implBlock.moveBy(50, 200);
    implBlock.setFieldValue('Point', 'NAME');

    var distMethod = makeBlock('zen_method');
    distMethod.setFieldValue('sum', 'NAME');
    distMethod.setFieldValue('', 'EXTRA_PARAMS');
    distMethod.setFieldValue('int', 'RETTYPE');

    var retDist = makeBlock('zen_return');
    var addXY = makeBlock('zen_math_arithmetic');
    addXY.setFieldValue('+', 'OP');
    var selfX = makeBlock('zen_field_access');
    var self1 = makeBlock('zen_self');
    connectValue(selfX, 'OBJ', self1);
    selfX.setFieldValue('x', 'FIELD');
    var selfY = makeBlock('zen_field_access');
    var self2 = makeBlock('zen_self');
    connectValue(selfY, 'OBJ', self2);
    selfY.setFieldValue('y', 'FIELD');
    connectValue(addXY, 'A', selfX);
    connectValue(addXY, 'B', selfY);
    connectValue(retDist, 'VALUE', addXY);
    connectStatement(distMethod, 'BODY', retDist);
    connectStatement(implBlock, 'METHODS', distMethod);
    implBlock.render();

    // fn main
    var main = makeBlock('zen_main');
    main.moveBy(50, 450);

    // let p = Point{ x: 10, y: 20 }
    var letP = makeBlock('zen_let_infer');
    letP.setFieldValue('p', 'VAR');
    var structInit = makeBlock('zen_struct_init');
    structInit.setFieldValue('Point', 'NAME');
    structInit.setFieldValue('x: 10, y: 20', 'FIELDS');
    connectValue(letP, 'VALUE', structInit);

    // println "p.x = {p.x}, p.y = {p.y}"
    var printP = makeBlock('zen_println');
    var pStr = makeBlock('zen_string');
    pStr.setFieldValue('Point({p.x}, {p.y}), sum = {p.sum()}', 'TEXT');
    connectValue(printP, 'TEXT', pStr);

    connectStatement(main, 'BODY', letP);
    chainBlocks(letP, printP);

    main.render();
  }

  // --- Memory & Pointers ---
  function buildBlocksMemory() {
    var importBlock = makeBlock('zen_raw_code');
    importBlock.moveBy(50, 20);
    importBlock.setFieldValue('import "std/mem.zc"', 'CODE');
    importBlock.render();

    var main = makeBlock('zen_main');
    main.moveBy(50, 70);

    // let x: int = 42
    var letX = makeBlock('zen_let');
    letX.setFieldValue('x', 'VAR');
    letX.setFieldValue('int', 'TYPE');
    var num42 = makeBlock('zen_number');
    num42.setFieldValue(42, 'NUM');
    connectValue(letX, 'VALUE', num42);

    // let ptr = &x
    var letPtr = makeBlock('zen_let_infer');
    letPtr.setFieldValue('ptr', 'VAR');
    var addrOf = makeBlock('zen_addr_of');
    var getX = makeBlock('zen_variable_get');
    getX.setFieldValue('x', 'VAR');
    connectValue(addrOf, 'VALUE', getX);
    connectValue(letPtr, 'VALUE', addrOf);

    // println "address: {ptr}, value: {*ptr}"
    var printPtr = makeBlock('zen_println');
    var ptrStr = makeBlock('zen_string');
    ptrStr.setFieldValue('value of x via pointer: {(*ptr)}', 'TEXT');
    connectValue(printPtr, 'TEXT', ptrStr);

    // let heap_val = alloc<int>()
    var letHeap = makeBlock('zen_let_infer');
    letHeap.setFieldValue('heap_val', 'VAR');
    var allocBlock = makeBlock('zen_alloc');
    allocBlock.setFieldValue('int', 'TYPE');
    var one = makeBlock('zen_number');
    one.setFieldValue(1, 'NUM');
    connectValue(allocBlock, 'SIZE', one);
    connectValue(letHeap, 'VALUE', allocBlock);

    // (*heap_val) = 99 — use raw code for clarity
    var assignHeap = makeBlock('zen_raw_code');
    assignHeap.setFieldValue('(*heap_val) = 99;\nprintln "heap value: {(*heap_val)}";', 'CODE');

    // free(heap_val)
    var freeBlock = makeBlock('zen_free');
    var getHeap = makeBlock('zen_variable_get');
    getHeap.setFieldValue('heap_val', 'VAR');
    connectValue(freeBlock, 'PTR', getHeap);

    var printDone = makeBlock('zen_println');
    var doneStr = makeBlock('zen_string');
    doneStr.setFieldValue('memory freed!', 'TEXT');
    connectValue(printDone, 'TEXT', doneStr);

    connectStatement(main, 'BODY', letX);
    chainBlocks(letX, letPtr);
    chainBlocks(letPtr, printPtr);
    chainBlocks(printPtr, letHeap);
    chainBlocks(letHeap, assignHeap);
    chainBlocks(assignHeap, freeBlock);
    chainBlocks(freeBlock, printDone);

    main.render();
  }

  // --- Vec & Collections ---
  function buildBlocksCollections() {
    var importBlock = makeBlock('zen_raw_code');
    importBlock.moveBy(50, 20);
    importBlock.setFieldValue('import "std.zc"', 'CODE');
    importBlock.render();

    var main = makeBlock('zen_main');
    main.moveBy(50, 70);

    // let v = Vec<int>::new()
    var letV = makeBlock('zen_let_infer');
    letV.setFieldValue('v', 'VAR');
    var vecNew = makeBlock('zen_vec_new');
    vecNew.setFieldValue('int', 'TYPE');
    connectValue(letV, 'VALUE', vecNew);

    // defer v.free()
    var deferBlock = makeBlock('zen_raw_code');
    deferBlock.setFieldValue('defer v.free();', 'CODE');

    // v.push(10), v.push(20), v.push(30)
    var push1 = makeBlock('zen_vec_push');
    var vecRef1 = makeBlock('zen_variable_get');
    vecRef1.setFieldValue('v', 'VAR');
    connectValue(push1, 'VEC', vecRef1);
    var val1 = makeBlock('zen_number');
    val1.setFieldValue(10, 'NUM');
    connectValue(push1, 'VALUE', val1);

    var push2 = makeBlock('zen_vec_push');
    var vecRef2 = makeBlock('zen_variable_get');
    vecRef2.setFieldValue('v', 'VAR');
    connectValue(push2, 'VEC', vecRef2);
    var val2 = makeBlock('zen_number');
    val2.setFieldValue(20, 'NUM');
    connectValue(push2, 'VALUE', val2);

    var push3 = makeBlock('zen_vec_push');
    var vecRef3 = makeBlock('zen_variable_get');
    vecRef3.setFieldValue('v', 'VAR');
    connectValue(push3, 'VEC', vecRef3);
    var val3 = makeBlock('zen_number');
    val3.setFieldValue(30, 'NUM');
    connectValue(push3, 'VALUE', val3);

    // println "length: {v.len}"
    var printLen = makeBlock('zen_println');
    var lenStr = makeBlock('zen_string');
    lenStr.setFieldValue('length: {v.len}', 'TEXT');
    connectValue(printLen, 'TEXT', lenStr);

    // for i in 0..v.len { print v.data[i] }
    var forIn = makeBlock('zen_for_in');
    forIn.setFieldValue('i', 'VAR');
    var start0 = makeBlock('zen_number');
    start0.setFieldValue(0, 'NUM');
    connectValue(forIn, 'START', start0);

    // end = cast v.len as int
    var castLen = makeBlock('zen_cast');
    castLen.setFieldValue('int', 'TYPE');
    var vecLenBlock = makeBlock('zen_vec_len');
    var vecRefLen = makeBlock('zen_variable_get');
    vecRefLen.setFieldValue('v', 'VAR');
    connectValue(vecLenBlock, 'VEC', vecRefLen);
    connectValue(castLen, 'VALUE', vecLenBlock);
    connectValue(forIn, 'END', castLen);

    // inside loop: print "{v.data[i]} "
    var printItem = makeBlock('zen_print');
    var itemStr = makeBlock('zen_string');
    itemStr.setFieldValue('{v.data[i]} ', 'TEXT');
    connectValue(printItem, 'TEXT', itemStr);
    connectStatement(forIn, 'DO', printItem);

    // println "" (newline after loop)
    var newline = makeBlock('zen_println');
    var emptyStr = makeBlock('zen_string');
    emptyStr.setFieldValue('', 'TEXT');
    connectValue(newline, 'TEXT', emptyStr);

    // let popped = v.pop()
    var letPop = makeBlock('zen_let_infer');
    letPop.setFieldValue('popped', 'VAR');
    var vecPop = makeBlock('zen_vec_pop');
    var vecRef4 = makeBlock('zen_variable_get');
    vecRef4.setFieldValue('v', 'VAR');
    connectValue(vecPop, 'VEC', vecRef4);
    connectValue(letPop, 'VALUE', vecPop);

    var printPop = makeBlock('zen_println');
    var popStr = makeBlock('zen_string');
    popStr.setFieldValue('popped: {popped}, new length: {v.len}', 'TEXT');
    connectValue(printPop, 'TEXT', popStr);

    connectStatement(main, 'BODY', letV);
    chainBlocks(letV, deferBlock);
    chainBlocks(deferBlock, push1);
    chainBlocks(push1, push2);
    chainBlocks(push2, push3);
    chainBlocks(push3, printLen);
    chainBlocks(printLen, forIn);
    chainBlocks(forIn, newline);
    chainBlocks(newline, letPop);
    chainBlocks(letPop, printPop);

    main.render();
  }

  // --- Match & Enums ---
  function buildBlocksMatch() {
    var main = makeBlock('zen_main');
    main.moveBy(50, 50);

    // let day: int = 3
    var letDay = makeBlock('zen_let');
    letDay.setFieldValue('day', 'VAR');
    letDay.setFieldValue('int', 'TYPE');
    var num3 = makeBlock('zen_number');
    num3.setFieldValue(3, 'NUM');
    connectValue(letDay, 'VALUE', num3);

    // match day { ... }
    var matchBlock = makeBlock('zen_match');
    var getDay = makeBlock('zen_variable_get');
    getDay.setFieldValue('day', 'VAR');
    connectValue(matchBlock, 'EXPR', getDay);

    // case 1: println "Monday"
    var case1 = makeBlock('zen_match_case');
    case1.setFieldValue('1', 'VALUE');
    var print1 = makeBlock('zen_println');
    var str1 = makeBlock('zen_string');
    str1.setFieldValue('Monday', 'TEXT');
    connectValue(print1, 'TEXT', str1);
    connectStatement(case1, 'DO', print1);

    // case 2: println "Tuesday"
    var case2 = makeBlock('zen_match_case');
    case2.setFieldValue('2', 'VALUE');
    var print2 = makeBlock('zen_println');
    var str2 = makeBlock('zen_string');
    str2.setFieldValue('Tuesday', 'TEXT');
    connectValue(print2, 'TEXT', str2);
    connectStatement(case2, 'DO', print2);

    // case 3: println "Wednesday"
    var case3 = makeBlock('zen_match_case');
    case3.setFieldValue('3', 'VALUE');
    var print3 = makeBlock('zen_println');
    var str3 = makeBlock('zen_string');
    str3.setFieldValue('Wednesday', 'TEXT');
    connectValue(print3, 'TEXT', str3);
    connectStatement(case3, 'DO', print3);

    // case 4: println "Thursday"
    var case4 = makeBlock('zen_match_case');
    case4.setFieldValue('4', 'VALUE');
    var print4 = makeBlock('zen_println');
    var str4 = makeBlock('zen_string');
    str4.setFieldValue('Thursday', 'TEXT');
    connectValue(print4, 'TEXT', str4);
    connectStatement(case4, 'DO', print4);

    // case 5: println "Friday"
    var case5 = makeBlock('zen_match_case');
    case5.setFieldValue('5', 'VALUE');
    var print5 = makeBlock('zen_println');
    var str5 = makeBlock('zen_string');
    str5.setFieldValue('Friday', 'TEXT');
    connectValue(print5, 'TEXT', str5);
    connectStatement(case5, 'DO', print5);

    // case _: println "Weekend"
    var caseDefault = makeBlock('zen_match_case');
    caseDefault.setFieldValue('_', 'VALUE');
    var printDef = makeBlock('zen_println');
    var strDef = makeBlock('zen_string');
    strDef.setFieldValue('Weekend!', 'TEXT');
    connectValue(printDef, 'TEXT', strDef);
    connectStatement(caseDefault, 'DO', printDef);

    // chain cases
    connectStatement(matchBlock, 'CASES', case1);
    chainBlocks(case1, case2);
    chainBlocks(case2, case3);
    chainBlocks(case3, case4);
    chainBlocks(case4, case5);
    chainBlocks(case5, caseDefault);

    // println after match
    var printResult = makeBlock('zen_println');
    var resStr = makeBlock('zen_string');
    resStr.setFieldValue('day number was: {day}', 'TEXT');
    connectValue(printResult, 'TEXT', resStr);

    connectStatement(main, 'BODY', letDay);
    chainBlocks(letDay, matchBlock);
    chainBlocks(matchBlock, printResult);

    main.render();
  }

  // --- Advanced Blocks ---
  function buildBlocksAdvanced() {
    var importBlock = makeBlock('zen_raw_code');
    importBlock.moveBy(50, 20);
    importBlock.setFieldValue('import "std.zc"', 'CODE');
    importBlock.render();

    var main = makeBlock('zen_main');
    main.moveBy(50, 70);

    // def MAX = 5
    var defBlock = makeBlock('zen_def');
    defBlock.setFieldValue('MAX', 'NAME');
    var num5 = makeBlock('zen_number');
    num5.setFieldValue(5, 'NUM');
    connectValue(defBlock, 'VALUE', num5);

    // let mask = 0xFF
    var letMask = makeBlock('zen_let_infer');
    letMask.setFieldValue('mask', 'VAR');
    var hexFF = makeBlock('zen_hex_number');
    hexFF.setFieldValue('FF', 'HEX');
    connectValue(letMask, 'VALUE', hexFF);

    // guard MAX > 0 else { println "Invalid!"; return; }
    var guardBlock = makeBlock('zen_guard');
    var cmpMax = makeBlock('zen_math_compare');
    cmpMax.setFieldValue('>', 'OP');
    var getMax = makeBlock('zen_variable_get');
    getMax.setFieldValue('MAX', 'VAR');
    var num0 = makeBlock('zen_number');
    num0.setFieldValue(0, 'NUM');
    connectValue(cmpMax, 'A', getMax);
    connectValue(cmpMax, 'B', num0);
    connectValue(guardBlock, 'COND', cmpMax);

    var guardPrint = makeBlock('zen_println');
    var guardStr = makeBlock('zen_string');
    guardStr.setFieldValue('Invalid MAX!', 'TEXT');
    connectValue(guardPrint, 'TEXT', guardStr);
    var guardRet = makeBlock('zen_return_void');
    connectStatement(guardBlock, 'ELSE', guardPrint);
    chainBlocks(guardPrint, guardRet);

    // let v = Vec<int>::new() — using static_call block
    var letV = makeBlock('zen_let_infer');
    letV.setFieldValue('v', 'VAR');
    var vecNew = makeBlock('zen_static_call');
    vecNew.setFieldValue('Vec<int>', 'TYPE');
    vecNew.setFieldValue('new', 'METHOD');
    vecNew.setFieldValue('', 'ARGS');
    connectValue(letV, 'VALUE', vecNew);

    // v.push(mask) — using method_call_stmt block
    var pushStmt1 = makeBlock('zen_method_call_stmt');
    var vRef1 = makeBlock('zen_variable_get');
    vRef1.setFieldValue('v', 'VAR');
    connectValue(pushStmt1, 'OBJ', vRef1);
    pushStmt1.setFieldValue('push', 'METHOD');
    pushStmt1.setFieldValue('mask', 'ARGS');

    // v.push(42) — using method_call_stmt block
    var pushStmt2 = makeBlock('zen_method_call_stmt');
    var vRef2 = makeBlock('zen_variable_get');
    vRef2.setFieldValue('v', 'VAR');
    connectValue(pushStmt2, 'OBJ', vRef2);
    pushStmt2.setFieldValue('push', 'METHOD');
    pushStmt2.setFieldValue('42', 'ARGS');

    // println results
    var printInfo = makeBlock('zen_println');
    var infoStr = makeBlock('zen_string');
    infoStr.setFieldValue('MAX={MAX}, mask={mask}, vec len={v.len}', 'TEXT');
    connectValue(printInfo, 'TEXT', infoStr);

    // v.free() — using method_call_stmt block
    var freeStmt = makeBlock('zen_method_call_stmt');
    var vRef3 = makeBlock('zen_variable_get');
    vRef3.setFieldValue('v', 'VAR');
    connectValue(freeStmt, 'OBJ', vRef3);
    freeStmt.setFieldValue('free', 'METHOD');
    freeStmt.setFieldValue('', 'ARGS');

    connectStatement(main, 'BODY', defBlock);
    chainBlocks(defBlock, letMask);
    chainBlocks(letMask, guardBlock);
    chainBlocks(guardBlock, letV);
    chainBlocks(letV, pushStmt1);
    chainBlocks(pushStmt1, pushStmt2);
    chainBlocks(pushStmt2, printInfo);
    chainBlocks(printInfo, freeStmt);

    main.render();
  }

  var EXAMPLE_BUILDERS = {
    hello_world: buildHelloWorld,
    fibonacci: buildFibonacci,
    struct_example: buildStructExample,
    file_io: buildFileIo,
    blocks_variables: buildBlocksVariables,
    blocks_logic: buildBlocksLogic,
    blocks_functions: buildBlocksFunctions,
    blocks_structs: buildBlocksStructs,
    blocks_memory: buildBlocksMemory,
    blocks_collections: buildBlocksCollections,
    blocks_match: buildBlocksMatch,
    blocks_advanced: buildBlocksAdvanced
  };

  function loadExample(name) {
    if (!EXAMPLE_BUILDERS[name]) {
      setStatus('Example "' + name + '" not found');
      return;
    }
    workspace.clear();
    EXAMPLE_BUILDERS[name]();
    generateCode();
    setStatus('Loaded: ' + name.replace(/_/g, ' '));
    setTimeout(function() { setStatus('Ready'); }, 3000);
  }

  // --- Window Resize ---
  function onResize() {
    if (workspace) {
      Blockly.svgResize(workspace);
    }
  }

  // --- Bind Events ---
  function bindEvents() {
    document.getElementById('target-select').addEventListener('change', function(e) {
      currentTarget = e.target.value;
      generateCode();
      setStatus('Target: ' + currentTarget);
    });

    document.getElementById('btn-compile').addEventListener('click', compile);
    document.getElementById('btn-run').addEventListener('click', run);
    document.getElementById('btn-save').addEventListener('click', saveWorkspace);
    document.getElementById('btn-load').addEventListener('click', loadWorkspace);
    document.getElementById('btn-copy').addEventListener('click', copyCode);

    // Output panel close
    document.getElementById('output-close').addEventListener('click', hideOutput);

    // Stream output from Electron
    if (window.zenckly) {
      window.zenckly.onOutputData(function(data) {
        appendOutput(data);
      });
    }

    // Examples modal
    document.getElementById('btn-examples').addEventListener('click', function() {
      document.getElementById('examples-modal').classList.remove('hidden');
    });
    document.getElementById('examples-close').addEventListener('click', function() {
      document.getElementById('examples-modal').classList.add('hidden');
    });
    document.getElementById('examples-modal').addEventListener('click', function(e) {
      if (e.target === this) this.classList.add('hidden');
    });

    // Example buttons
    var exampleBtns = document.querySelectorAll('.example-btn');
    for (var i = 0; i < exampleBtns.length; i++) {
      exampleBtns[i].addEventListener('click', function() {
        var name = this.getAttribute('data-example');
        loadExample(name);
        document.getElementById('examples-modal').classList.add('hidden');
      });
    }

    window.addEventListener('resize', onResize);
  }

  // --- Init ---
  function init() {
    initWorkspace();
    initDivider();
    bindEvents();
    generateCode();
    setStatus('Ready — drag blocks to start coding!');
  }

  // Wait for DOM
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
