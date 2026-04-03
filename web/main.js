let currentFile = null;
  let history     = [];

  const $  = id => document.getElementById(id);
  const url = () => $('api-url').value.replace(/\/$/, '');

  /* ── Health check ── */
  async function checkHealth() {
    const dot = $('dot'), txt = $('status-text');
    dot.className = 'dot';
    txt.textContent = 'Checking...';
    try {
      const r = await fetch(url() + '/health');
      const d = await r.json();
      if (d.status === 'ok' && d.model_loaded) {
        dot.className = 'dot ok';
        txt.textContent = 'API online · model loaded';
      } else {
        dot.className = 'dot err';
        txt.textContent = d.model_loaded ? 'API online · model not loaded' : 'Unhealthy';
      }
    } catch {
      dot.className = 'dot err';
      txt.textContent = 'Cannot reach API at ' + url();
    }
  }

  /* ── File select ── */
  function onFileSelect(file) {
    if (!file) return;
    clearError();
    const warn = $('size-warn');
    if (file.size > 1 * 1024 * 1024) {
      warn.classList.add('show');
      $('predict-btn').disabled = true;
      return;
    }
    warn.classList.remove('show');
    currentFile = file;
    $('preview-img').src = URL.createObjectURL(file);
    $('preview-name').textContent = file.name;
    $('preview-size').textContent = (file.size / 1024).toFixed(1) + ' KB';
    $('preview-wrap').classList.add('show');
    $('drop-zone').style.display = 'none';
    $('predict-btn').disabled = false;
  }

  /* ── Clear ── */
  function clearFile() {
    currentFile = null;
    $('preview-wrap').classList.remove('show');
    $('drop-zone').style.display = '';
    $('file-input').value = '';
    $('predict-btn').disabled = true;
    $('size-warn').classList.remove('show');
    clearError();
  }

  function clearAll() {
    clearFile();
    $('result-block').classList.remove('show');
    $('digit-display').textContent = '—';
  }

  function clearError() {
    const b = $('error-box');
    b.textContent = '';
    b.classList.remove('show');
  }

  function showError(msg) {
    const b = $('error-box');
    b.textContent = msg;
    b.classList.add('show');
  }

  /* ── Predict ── */
  async function predict() {
    if (!currentFile) return;
    clearError();
    const btn  = $('predict-btn');
    const spin = $('spinner');
    btn.disabled = true;
    spin.classList.add('show');

    const form = new FormData();
    form.append('file', currentFile);

    try {
      const r = await fetch(url() + '/predict', { method: 'POST', body: form });
      const d = await r.json();
      if (!r.ok) { showError('Error ' + r.status + ': ' + (d.detail || 'Unknown error')); return; }
      renderResult(d);
      addHistory(d);
    } catch {
      showError('Failed to reach API. Is it running at ' + url() + '?');
    } finally {
      btn.disabled = false;
      spin.classList.remove('show');
    }
  }

  /* ── Render result ── */
  function renderResult(d) {
    $('result-block').classList.add('show');
    $('digit-display').textContent = d.predicted_digit;

    const pct = Math.round(d.confidence * 100);
    $('conf-val').textContent = pct + '%';
    $('bar-fill').style.width  = pct + '%';

    const grid = $('prob-grid');
    grid.innerHTML = '';
    for (let i = 0; i <= 9; i++) {
      const p   = d.probabilities[String(i)] || 0;
      const win = i === d.predicted_digit;
      const cell = document.createElement('div');
      cell.className = 'prob-cell' + (win ? ' winner' : '');
      cell.innerHTML =
        '<div class="prob-fill" style="height:' + Math.round(p * 100) + '%"></div>' +
        '<span class="prob-digit">' + i + '</span>' +
        '<span class="prob-pct">'  + Math.round(p * 100) + '%</span>';
      grid.appendChild(cell);
    }
  }

  /* ── History ── */
  function addHistory(d) {
    history.unshift({ digit: d.predicted_digit, conf: Math.round(d.confidence * 100) });
    if (history.length > 10) history.pop();
    const container = $('history-chips');
    $('history-section').classList.add('show');
    container.innerHTML = '';
    history.forEach(h => {
      const chip = document.createElement('div');
      chip.className = 'chip';
      chip.innerHTML = '<span class="c-digit">' + h.digit + '</span><span class="c-conf">' + h.conf + '%</span>';
      container.appendChild(chip);
    });
  }

  /* ── Drag and drop ── */
  const dz = $('drop-zone');
  dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('drag'); });
  dz.addEventListener('dragleave', ()  => dz.classList.remove('drag'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('drag');
    const f = e.dataTransfer.files[0];
    if (f) onFileSelect(f);
  });

  checkHealth();