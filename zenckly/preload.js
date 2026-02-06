const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('zenckly', {
  compile: (code) => ipcRenderer.invoke('compile', code),
  run: (code) => ipcRenderer.invoke('run', code),
  saveFile: (code, name) => ipcRenderer.invoke('save-file', code, name),
  onOutputData: (callback) => ipcRenderer.on('output-data', (_event, data) => callback(data))
});
