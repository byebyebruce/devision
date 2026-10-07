/* UI state stays in the browser; all model decisions use the public HTTP API. */
(() => {
  'use strict';
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const esc = (value) => String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const percent = (value) => `${(value * 100).toFixed(1)}%`;
  const state = { image: null, scene: null, scenes: [], filter: 'all', imageRevision: 0, inputRevision: 0, loading: false, busy: false, result: null, controller: null, questionId: 0 };

  function notice(message = '') {
    $('#input-notice').textContent = message;
    $('#input-notice').hidden = !message;
  }
  function status(message, kind = '') {
    $('#status').textContent = message;
    $('#status').className = `result-status ${kind}`;
  }
  function changed() {
    state.inputRevision++;
    $('#stale').hidden = !state.result;
    notice();
  }
  function syncControls() {
    $('#editor').disabled = state.busy;
    $('#run').disabled = state.busy || state.loading;
    $('#flip').disabled = !state.image || state.loading;
    $('#clear-image').disabled = !state.image || state.loading;
    $('#cancel').hidden = !state.busy;
    $('#export').disabled = !state.result || !state.result.runs.length || state.busy;
    $('#results').setAttribute('aria-busy', String(state.busy));
    $$('.scene').forEach((el) => { el.disabled = state.busy; });
    const count = $('#questions').children.length;
    $('#question-count').textContent = `${count} / 12 道题`;
    $('#add-choice').disabled = $('#add-noul').disabled = count >= 12;
    $('#run-label').textContent = state.busy ? '正在分析…' : state.loading ? '图片加载中…' : `开始分析${count ? ` · ${count} 道题` : ''}`;
    if (!state.image) $('#compare-mode').value = 'none';
    $('#compare-mode').disabled = !state.image || state.loading;
  }

  // Scene changes and uploads use a revision token so late network responses cannot replace newer input.
  function renderScenes() {
    const visible = state.scenes.filter((scene) => state.filter === 'all' || scene.kind === state.filter);
    $('#scene-count').textContent = String(visible.length);
    $('#scenes').innerHTML = visible.map((scene) => `<button class="scene ${state.scene === scene.id ? 'active' : ''}" type="button" data-scene="${esc(scene.id)}" aria-pressed="${state.scene === scene.id}"><div class="scene-image"><img src="${esc(scene.url)}" alt="${esc(scene.title)}"><span class="check" aria-hidden="true">✓</span></div><div class="scene-info"><strong>${esc(scene.title)}</strong><span>${esc(scene.subtitle)}</span></div></button>`).join('') || '<p class="scene-empty">没有可用的照片示例，你可以上传自己的图片。</p>';
    $$('.scene').forEach((el) => el.addEventListener('click', () => selectScene(state.scenes.find((scene) => scene.id === el.dataset.scene))));
    syncControls();
  }
  function readFile(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(new Error('图片读取失败，请重新选择。'));
      reader.readAsDataURL(blob);
    });
  }
  function decodeImage(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error('无法打开这张图片，请使用 JPG、PNG 或 WebP。'));
      image.src = url;
    });
  }
  async function rasterize(url, mirror = false) {
    const image = await decodeImage(url);
    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('浏览器暂时无法处理图片，请重新加载页面。');
    if (mirror) { ctx.translate(canvas.width, 0); ctx.scale(-1, 1); }
    ctx.drawImage(image, 0, 0);
    const result = canvas.toDataURL('image/png');
    if (!result.startsWith('data:image/png;')) throw new Error('图片转换失败，请换一张较小的图片。');
    return result;
  }
  async function setImage(url, metadata, revision) {
    const image = await decodeImage(url);
    if (revision !== state.imageRevision) return false;
    state.image = { url, ...metadata, width: image.naturalWidth, height: image.naturalHeight };
    $('#preview').src = url;
    $('#preview').alt = metadata.title;
    $('#preview').hidden = false;
    $('#upload-empty').hidden = true;
    $('#image-title').textContent = metadata.title;
    $('#image-caption').textContent = metadata.caption;
    $('#image-kind').textContent = metadata.kind;
    $('#image-size').textContent = `${image.naturalWidth} × ${image.naturalHeight}`;
    changed();
    return true;
  }
  async function selectScene(scene) {
    if (!scene || state.busy) return;
    const revision = ++state.imageRevision;
    state.loading = true;
    notice();
    syncControls();
    try {
      let url;
      if (scene.kind === 'diagram') url = await rasterize(scene.url);
      else {
        const response = await fetch(scene.url, { signal: AbortSignal.timeout(15000) });
        if (!response.ok) throw new Error(`示例图片加载失败（${response.status}），请上传图片或选另一个场景。`);
        url = await readFile(await response.blob());
      }
      if (!await setImage(url, { title: scene.title, caption: scene.subtitle, kind: scene.kind === 'diagram' ? '合成示例' : '真实照片', mirrored: false }, revision)) return;
      state.scene = scene.id;
      $('#questions').replaceChildren();
      scene.questions.forEach((question) => addQuestion(question));
      $('#context').value = '';
      $('#scene-description').textContent = scene.description;
      renderScenes();
    } catch (error) {
      if (revision === state.imageRevision) notice(error.message);
    } finally {
      if (revision === state.imageRevision) { state.loading = false; syncControls(); }
    }
  }
  async function upload(file) {
    if (!file || state.busy) return;
    const revision = ++state.imageRevision;
    state.loading = false;
    syncControls();
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) return notice('请选择 JPG、PNG 或 WebP 图片。');
    if (file.size > 10 * 1024 * 1024) return notice('图片超过 10 MB，请选择较小的文件。');
    state.loading = true;
    syncControls();
    try {
      const url = await readFile(file);
      if (!await setImage(url, { title: file.name, caption: '你的图片 · 自由提问', kind: '上传图片', mirrored: false }, revision)) return;
      state.scene = null;
      $('#scene-description').textContent = '已保留当前问题，请根据新图片编辑。模型目前支持英文问题。';
      renderScenes();
    } catch (error) {
      if (revision === state.imageRevision) notice(error.message);
    } finally {
      if (revision === state.imageRevision) { state.loading = false; syncControls(); }
    }
  }
  $('#drop').addEventListener('click', () => $('#file').click());
  $('#upload').addEventListener('click', () => $('#file').click());
  $('#file').addEventListener('change', (event) => { upload(event.target.files[0]); event.target.value = ''; });
  ['dragenter', 'dragover'].forEach((type) => $('#drop').addEventListener(type, (event) => { event.preventDefault(); if (!state.busy) $('#drop').classList.add('over'); }));
  $('#drop').addEventListener('dragleave', () => $('#drop').classList.remove('over'));
  $('#drop').addEventListener('drop', (event) => { event.preventDefault(); $('#drop').classList.remove('over'); upload(event.dataTransfer.files[0]); });
  $('#clear-image').addEventListener('click', () => {
    state.imageRevision++;
    state.image = null;
    state.scene = null;
    $('#preview').hidden = true;
    $('#preview').removeAttribute('src');
    $('#upload-empty').hidden = false;
    $('#image-title').textContent = '纯文本模式';
    $('#image-caption').textContent = '本次只发送问题和背景信息。';
    $('#image-kind').textContent = '未使用图片';
    $('#image-size').textContent = '';
    $('#scene-description').textContent = '可以直接提问，也可以上传图片回到视觉体验。';
    changed();
    renderScenes();
  });
  $('#flip').addEventListener('click', async () => {
    if (!state.image || state.busy || state.loading) return;
    const revision = ++state.imageRevision;
    const current = { ...state.image };
    state.loading = true;
    syncControls();
    try {
      const url = await rasterize(current.url, true);
      if (await setImage(url, { ...current, url, mirrored: !current.mirrored, caption: !current.mirrored ? '已水平镜像 · 问题保持不变' : '已恢复方向 · 问题保持不变' }, revision)) {
        state.scene = null;
        renderScenes();
      }
    } catch (error) { notice(error.message); }
    finally { if (revision === state.imageRevision) { state.loading = false; syncControls(); } }
  });
  $$('[data-filter]').forEach((button) => button.addEventListener('click', () => {
    state.filter = button.dataset.filter;
    $$('[data-filter]').forEach((el) => { el.classList.toggle('active', el === button); el.setAttribute('aria-pressed', String(el === button)); });
    renderScenes();
  }));

  function addQuestion({ type = 'noul', text = '', options = [] } = {}, focus = false) {
    if ($('#questions').children.length >= 12) return notice('一次最多展示 12 道题，请先删除不需要的问题。');
    const id = ++state.questionId;
    const element = document.createElement('div');
    element.className = 'question';
    element.dataset.type = type;
    element.innerHTML = `<div class="question-top"><label for="instruction-${id}"><span class="question-number"></span><span class="question-type">${type === 'noul' ? '是 / 否判断' : '候选项选择'}</span></label><button type="button" class="remove-question" aria-label="删除问题">×</button></div><textarea id="instruction-${id}" class="instruction" rows="1" maxlength="2000" placeholder="${type === 'noul' ? 'Is there a dog in the image?' : 'What color is the object?'}"></textarea>${type === 'choice' ? `<div class="option-input"><label for="options-${id}">候选答案 · 逗号或换行分隔</label><textarea id="options-${id}" class="options" rows="1" maxlength="4000" placeholder="red, green, blue"></textarea></div>` : ''}`;
    element.querySelector('.instruction').value = text;
    if (type === 'choice') element.querySelector('.options').value = options.join(', ');
    element.querySelector('.remove-question').addEventListener('click', () => {
      const neighbor = element.nextElementSibling || element.previousElementSibling;
      element.remove(); renumber(); changed();
      (neighbor?.querySelector('.instruction') || $('#add-noul')).focus();
    });
    $('#questions').appendChild(element);
    renumber();
    changed();
    if (focus) element.querySelector('.instruction').focus();
  }
  function renumber() {
    $$('.question').forEach((element, index) => {
      element.querySelector('.question-number').textContent = `Q${String(index + 1).padStart(2, '0')}`;
      element.querySelector('.remove-question').setAttribute('aria-label', `删除第 ${index + 1} 道问题`);
    });
    syncControls();
  }
  $('#add-noul').addEventListener('click', () => addQuestion({ type: 'noul' }, true));
  $('#add-choice').addEventListener('click', () => addQuestion({ type: 'choice' }, true));
  $('#workspace').addEventListener('input', changed);
  $('#compare-mode').addEventListener('change', changed);

  function buildRequest() {
    const elements = $$('.question');
    if (!elements.length) throw new Error('先添加至少一道问题。');
    const questions = Object.fromEntries(elements.map((element, index) => {
      const instruction = element.querySelector('.instruction');
      const question = { type: element.dataset.type, instructions: instruction.value.trim() };
      if (!question.instructions) { instruction.focus(); throw new Error(`第 ${index + 1} 道题还没有问题内容，请用英文填写。`); }
      if (question.type === 'choice') {
        const input = element.querySelector('.options');
        const options = input.value.split(/[,，\n]/).map((option) => option.trim()).filter(Boolean);
        if (options.length < 2 || options.length > 255) { input.focus(); throw new Error(`第 ${index + 1} 道选择题需要 2–255 个候选答案。`); }
        if (new Set(options).size !== options.length) { input.focus(); throw new Error(`第 ${index + 1} 道题有重复选项，请修改后重试。`); }
        question.criteria = Object.fromEntries(options.map((option) => [option, null]));
      }
      return [`q${index + 1}`, question];
    }));
    const body = { model: 'devision', state: $('#context').value.trim(), questions };
    if (state.image) body.image = state.image.url.split(',')[1];
    return body;
  }
  function shownRequest(body) {
    return body.image ? { ...body, image: `[图片已省略：${body.image.length} 个字符]` } : body;
  }
  function validateAnswers(data, questions) {
    const probability = (value) => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
    for (const [id, question] of Object.entries(questions)) {
      const answer = data?.answers?.[id];
      if (!answer || answer.type !== question.type) throw new Error('接口返回的题目或答案类型不完整，请检查服务后重试。');
      if (answer.type === 'noul') {
        if (!probability(answer.noul)) throw new Error('接口返回了无效的是非概率。');
      } else {
        const entries = Object.entries(answer.probabilities || {});
        if (entries.length !== Object.keys(question.criteria).length || entries.some(([key, value]) => !Object.hasOwn(question.criteria, key) || !probability(value)) || !Object.hasOwn(answer.probabilities, answer.choice) || !probability(answer.confidence) || Math.abs(entries.reduce((sum, [, p]) => sum + p, 0) - 1) > .001) throw new Error('接口返回了无效的选项概率。');
      }
    }
  }
  async function ask(body, controller) {
    const started = performance.now();
    const response = await fetch('/v1/systemone', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: controller.signal });
    const text = await response.text();
    let data;
    try { data = JSON.parse(text); } catch { throw new Error(`服务未返回 JSON（HTTP ${response.status}），请检查模型服务是否已启动。`); }
    if (!response.ok) throw new Error(`请求失败（${response.status}）：${typeof data.error === 'string' ? data.error : typeof data.detail === 'string' ? data.detail : '请检查输入和模型服务。'}`);
    validateAnswers(data, body.questions);
    return { request: shownRequest(body), response: data, elapsed_ms: Math.round(performance.now() - started) };
  }
  function probabilities(answer) {
    return answer.type === 'noul' ? [['yes', answer.noul], ['no', 1 - answer.noul]] : Object.entries(answer.probabilities);
  }
  function winner(answer) {
    return answer.type === 'noul' ? (answer.noul >= .5 ? 'yes' : 'no') : answer.choice;
  }
  function distribution(answer) {
    return `<div class="distribution">${probabilities(answer).sort((a, b) => b[1] - a[1]).map(([label, value]) => `<div class="probability ${label === winner(answer) ? 'top' : ''}"><div class="probability-label"><span>${esc(label)}</span><span>${percent(value)}</span></div><div class="probability-track" aria-hidden="true"><div class="probability-fill" style="width:${value * 100}%"></div></div></div>`).join('')}</div>`;
  }
  function renderResults() {
    const result = state.result;
    $('#raw').textContent = JSON.stringify(result, null, 2);
    $('#answers').replaceChildren();
    const first = result?.runs[0];
    $('#results-empty').hidden = Boolean(first);
    $('#probability-note').hidden = !first;
    if (!first) return;
    const comparison = result.runs[1];
    $('#answers').innerHTML = Object.entries(first.request.questions).map(([id, question], index) => {
      const answer = first.response.answers[id];
      const label = winner(answer);
      const p = probabilities(answer).find(([key]) => key === label)[1];
      const other = comparison?.response.answers[id];
      const otherLabel = other && winner(other);
      const otherP = other && probabilities(other).find(([key]) => key === otherLabel)[1];
      const delta = other && (probabilities(other).find(([key]) => key === label)[1] - p) * 100;
      return `<article class="answer-card"><div class="answer-heading"><span>Q${String(index + 1).padStart(2, '0')}</span><span class="tag">${answer.type === 'noul' ? '是 / 否' : '选择题'}</span></div><h3>${esc(question.instructions)}</h3><div class="answer-winner"><strong>${esc(label)}</strong><span>${percent(p)} 概率</span></div>${distribution(answer)}${answer.type === 'choice' ? `<p class="confidence" title="Jev confidence = (选项数 × 最高概率 − 1) / (选项数 − 1)，与答案概率不同">决策置信度 ${percent(answer.confidence)} · Jev 指标</p>` : ''}${other ? `<div class="comparison"><h4>${result.compare === 'mirror' ? '水平镜像对照' : '无图对照'}</h4><p><b>${esc(otherLabel)}</b><span>${percent(otherP)} 概率</span></p>${distribution(other)}<small>${otherLabel === label ? '首选答案相同' : '首选答案改变'} · 原首选项概率 ${delta >= 0 ? '+' : ''}${delta.toFixed(1)} 个百分点</small></div>` : ''}</article>`;
    }).join('');
  }
  $('#workspace').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (state.busy || state.loading) return;
    let body;
    try { body = buildRequest(); } catch (error) { notice(error.message); return; }
    const mode = $('#compare-mode').value;
    const image = state.image;
    const revision = state.inputRevision;
    const controller = new AbortController();
    state.controller = controller;
    state.busy = true;
    let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 120000);
    state.result = { created_at: new Date().toISOString(), image: image ? { title: image.title, width: image.width, height: image.height, mirrored: image.mirrored } : null, compare: mode, runs: [] };
    $('#stale').hidden = true;
    notice();
    syncControls();
    renderResults();
    status(mode === 'none' ? '模型正在观察并回答…' : '正在分析当前图片（1 / 2）…', 'loading');
    $('#results').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' });
    try {
      const first = await ask(body, controller);
      state.result.runs.push({ label: image ? '当前图片' : '纯文本', ...first });
      renderResults();
      if (mode !== 'none') {
        status(`当前图片已完成，正在分析${mode === 'mirror' ? '镜像' : '无图'}对照（2 / 2）…`, 'loading');
        const { image: _, ...control } = body;
        if (mode === 'mirror') control.image = (await rasterize(image.url, true)).split(',')[1];
        if (controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');
        state.result.runs.push({ label: mode === 'mirror' ? '水平镜像' : '无图', ...await ask(control, controller) });
        renderResults();
      }
      const elapsed = state.result.runs.map((run) => `${run.label} ${run.elapsed_ms.toLocaleString()} ms`).join(' · ');
      status(`已完成 ${Object.keys(body.questions).length} 道题 · ${elapsed}（含网络与上传） · ${first.response.model || 'devision'}`);
    } catch (error) {
      const message = controller.signal.aborted ? (timedOut ? '等待超过 120 秒，请检查服务负载后重试。' : '已取消等待；服务端可能仍在处理本次请求。') : error.message;
      state.result.error = message;
      status(`${state.result.runs.length ? '当前图片结果已保留；对照未完成。' : ''}${message}`, 'error');
      renderResults();
    } finally {
      clearTimeout(timeout);
      state.busy = false;
      state.controller = null;
      $('#stale').hidden = revision === state.inputRevision;
      syncControls();
    }
  });
  $('#cancel').addEventListener('click', () => state.controller?.abort());
  document.addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') { event.preventDefault(); $('#workspace').requestSubmit(); }
  });
  $('#export').addEventListener('click', () => {
    if (!state.result?.runs.length) return;
    const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `devision-${state.result.created_at.replace(/[:.]/g, '-')}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  async function checkHealth() {
    $('#connection').disabled = true;
    try {
      const response = await fetch('/health', { signal: AbortSignal.timeout(8000) });
      const health = await response.json();
      if (!response.ok || health.status !== 'ok') throw new Error('unavailable');
      $('#connection').className = 'connection online';
      $('#connection-label').textContent = health.model || '模型已连接';
      $('#connection').title = `模型已连接：${health.model || 'devision'}。点击重新检查。`;
    } catch {
      $('#connection').className = 'connection offline';
      $('#connection-label').textContent = '模型未连接';
      $('#connection').title = '模型服务暂时不可用，点击重新检查。';
    } finally { $('#connection').disabled = false; }
  }
  $('#connection').addEventListener('click', checkHealth);
  async function initialize() {
    const initialRevision = state.inputRevision;
    const initialImageRevision = state.imageRevision;
    state.scenes = [...window.devisionScenes.diagrams];
    renderScenes();
    let names = [];
    try {
      const response = await fetch('/examples', { signal: AbortSignal.timeout(8000) });
      if (!response.ok) throw new Error('unavailable');
      const data = await response.json();
      if (!Array.isArray(data) || data.some((name) => typeof name !== 'string')) throw new Error('invalid');
      names = data;
    } catch { $('#scene-note').textContent = '照片示例暂时不可用。可以使用合成示例或上传图片；预测结果将在运行后展示。'; }
    state.scenes = [...names.map(window.devisionScenes.fromFile), ...window.devisionScenes.diagrams];
    renderScenes();
    if (state.inputRevision === initialRevision && state.imageRevision === initialImageRevision) await selectScene(state.scenes[0]);
  }
  addQuestion({ type: 'noul', text: 'Is there a dog in the image?' });
  checkHealth();
  initialize();
})();
