const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, execFile } = require('child_process');
const os = require('os');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: 'Zenckly — Visual Zen-C',
    backgroundColor: '#1e1e2e',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  mainWindow.loadFile('index.html');
  mainWindow.setMenuBarVisibility(false);
}

// Find the zc compiler
function findZc() {
  const candidates = [
    'zc',                                         // PATH
    path.join(__dirname, '..', 'zc'),             // ../zc (sibling of zenckly dir)
    '/usr/local/bin/zc',
    path.join(os.homedir(), '.local', 'bin', 'zc')
  ];
  for (const c of candidates) {
    try {
      // Check if it exists (absolute) or is in PATH (bare name)
      if (path.isAbsolute(c) || c.includes(path.sep)) {
        if (fs.existsSync(c)) return c;
      } else {
        // For bare 'zc', try which
        const { execSync } = require('child_process');
        const result = execSync('which ' + c, { encoding: 'utf8' }).trim();
        if (result) return result;
      }
    } catch (_) { /* next candidate */ }
  }
  return null;
}

// Create temp file and return its path
let tempCounter = 0;
function writeTempFile(code) {
  const dir = os.tmpdir();
  const name = 'zenckly_' + Date.now() + '_' + (++tempCounter) + '.zc';
  const filePath = path.join(dir, name);
  fs.writeFileSync(filePath, code, 'utf8');
  return filePath;
}

// IPC: compile
ipcMain.handle('compile', async (_event, code) => {
  const zc = findZc();
  if (!zc) {
    return { success: false, output: 'Error: zc compiler not found.\nChecked: PATH, ../zc, /usr/local/bin/zc, ~/.local/bin/zc' };
  }

  const srcPath = writeTempFile(code);
  const outPath = srcPath.replace(/\.zc$/, '');

  return new Promise((resolve) => {
    execFile(zc, ['build', srcPath, '-o', outPath], { timeout: 30000 }, (err, stdout, stderr) => {
      const output = (stdout || '') + (stderr || '');
      if (err) {
        resolve({ success: false, output: output || err.message, binary: null });
      } else {
        resolve({ success: true, output: output || 'Compilation successful.', binary: outPath });
      }
      // Clean up source
      try { fs.unlinkSync(srcPath); } catch (_) {}
    });
  });
});

// IPC: run (compile then execute)
ipcMain.handle('run', async (event, code) => {
  const zc = findZc();
  if (!zc) {
    return { success: false, output: 'Error: zc compiler not found.' };
  }

  const srcPath = writeTempFile(code);
  const outPath = srcPath.replace(/\.zc$/, '');

  // Compile first
  return new Promise((resolve) => {
    execFile(zc, ['build', srcPath, '-o', outPath], { timeout: 30000 }, (compErr, compStdout, compStderr) => {
      // Clean up source
      try { fs.unlinkSync(srcPath); } catch (_) {}

      const compOutput = (compStdout || '') + (compStderr || '');
      if (compErr) {
        resolve({ success: false, output: 'Compilation failed:\n' + (compOutput || compErr.message) });
        return;
      }

      // Make executable
      try { fs.chmodSync(outPath, 0o755); } catch (_) {}

      // Run the binary
      const child = spawn(outPath, [], { timeout: 10000 });
      let output = compOutput ? compOutput + '\n--- Program Output ---\n' : '';

      child.stdout.on('data', (data) => {
        output += data.toString();
        // Stream output to renderer
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('output-data', data.toString());
        }
      });

      child.stderr.on('data', (data) => {
        output += data.toString();
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('output-data', data.toString());
        }
      });

      child.on('close', (exitCode) => {
        output += '\n--- Process exited with code ' + exitCode + ' ---';
        // Clean up binary
        try { fs.unlinkSync(outPath); } catch (_) {}
        resolve({ success: exitCode === 0, output: output });
      });

      child.on('error', (err) => {
        output += '\nExecution error: ' + err.message;
        try { fs.unlinkSync(outPath); } catch (_) {}
        resolve({ success: false, output: output });
      });
    });
  });
});

// IPC: save file dialog
ipcMain.handle('save-file', async (_event, code, defaultName) => {
  const { dialog } = require('electron');
  const result = await dialog.showSaveDialog(mainWindow, {
    defaultPath: defaultName || 'program.zc',
    filters: [{ name: 'Zen-C Source', extensions: ['zc'] }]
  });
  if (!result.canceled && result.filePath) {
    fs.writeFileSync(result.filePath, code, 'utf8');
    return { success: true, path: result.filePath };
  }
  return { success: false };
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
