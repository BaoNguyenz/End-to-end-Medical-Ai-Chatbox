/* app.js — Enterprise RAG Frontend */

const API = '';  // same origin

const SAMPLES = [
  'How does API authentication work?',
  'What are the GDPR data privacy requirements?',
  'Who is responsible for incident response?',
  'Compare data privacy and information security policies',
  'ERR_AUTH_001',
  'TechDocs Pro pricing',
  'What is the remote work equipment policy?',
  'Which regulations does TechDocs comply with?',
];

// ── Init ──────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  renderSamples();
  checkHealth();
  loadStats();

  // Enter key submits (Shift+Enter = newline)
  document.getElementById('query-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuery();
    }
  });
});

// ── Sample queries ────────────────────────────────────────────────────────

function renderSamples() {
  const grid = document.getElementById('samples-grid');
  SAMPLES.forEach(q => {
    const chip = document.createElement('div');
    chip.className = 'sample-chip';
    chip.textContent = q;
    chip.onclick = () => {
      document.getElementById('query-input').value = q;
      submitQuery();
    };
    grid.appendChild(chip);
  });
}

// ── Health check ──────────────────────────────────────────────────────────

async function checkHealth() {
  const dot = document.getElementById('status-dot');
  try {
    const res = await fetch(`${API}/api/health`);
    const data = await res.json();
    const qdrantOk = data.components?.qdrant?.status === 'ok';
    dot.className = 'status-dot ' + (qdrantOk ? 'ok' : 'err');
    dot.title = qdrantOk ? 'All systems operational' : 'Qdrant unavailable';
  } catch {
    dot.className = 'status-dot err';
    dot.title = 'API unreachable';
  }
}

// ── Stats ─────────────────────────────────────────────────────────────────

async function loadStats() {
  try {
    const res = await fetch(`${API}/api/stats`);
    const data = await res.json();
    document.getElementById('stat-vectors').textContent =
      data.vector_store?.points_count ?? '—';
    document.getElementById('stat-bm25').textContent =
      data.bm25_corpus_size ?? '—';
    const gc = data.graph_counts ?? {};
    const totalNodes = (gc.Policy ?? 0) + (gc.Stakeholder ?? 0) +
                       (gc.Product ?? 0) + (gc.Regulation ?? 0) +
                       (gc.TechnicalDoc ?? 0);
    document.getElementById('stat-graph').textContent =
      data.graph_enabled ? totalNodes : 'Off';
    document.getElementById('stat-rels').textContent =
      data.graph_enabled ? (gc.Relationships ?? '—') : 'Off';
  } catch {
    // silently fail
  }
}

// ── Submit query ──────────────────────────────────────────────────────────

async function submitQuery() {
  const queryInput = document.getElementById('query-input');
  const query = queryInput.value.trim();
  if (!query) return;

  const mode  = document.getElementById('mode-select').value;
  const top_k = parseInt(document.getElementById('topk-slider').value);

  setLoading(true);

  try {
    const res = await fetch(`${API}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, search_mode: mode, top_k }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    renderError(err.message);
  } finally {
    setLoading(false);
  }
}

// ── Render result ─────────────────────────────────────────────────────────

function renderResult(data) {
  const area = document.getElementById('results-area');

  const totalMs = data.latency?.total ?? 0;
  const stageKeys = ['retrieval', 'graph_search', 'post_retrieval', 'generation'];
  const maxMs = Math.max(...stageKeys.map(k => data.latency?.[k] ?? 0), 1);

  const latencyBars = stageKeys
    .filter(k => data.latency?.[k] != null)
    .map(k => {
      const ms = data.latency[k];
      const pct = Math.round((ms / Math.max(totalMs, 1)) * 100);
      return `
        <div class="latency-row">
          <span class="latency-label">${k.replace('_', ' ')}</span>
          <div class="latency-bar-wrap">
            <div class="latency-bar" style="width:${pct}%"></div>
          </div>
          <span class="latency-ms">${Math.round(ms)}ms</span>
        </div>`;
    }).join('');

  const sourceBadgeClass = src =>
    ({ vector:'src-vector', bm25:'src-bm25', hybrid:'src-hybrid', graph:'src-graph' }[src] || '');

  const sourceCards = (data.sources || []).map((s, i) => `
    <div class="source-card">
      <div class="source-header">
        <span class="source-rank">${i + 1}</span>
        <span class="source-docid">${escHtml(s.doc_id)}</span>
        <span class="source-badge ${sourceBadgeClass(s.source)}">${s.source}</span>
        <span class="source-score">score: ${s.score}</span>
      </div>
      <div class="source-preview">${escHtml(s.content_preview)}${s.content_preview.length >= 200 ? '…' : ''}</div>
    </div>`).join('');

  const meta = data.metadata || {};
  const metaTags = [
    meta.search_mode && `<span class="meta-tag mode">mode: ${meta.search_mode}</span>`,
    meta.query_class && `<span class="meta-tag class">class: ${meta.query_class}</span>`,
    meta.transformation && meta.transformation !== 'none' &&
      `<span class="meta-tag transform">transform: ${meta.transformation}</span>`,
  ].filter(Boolean).join('');

  area.innerHTML = `
    <div class="answer-card">
      <div class="answer-query">Answering: <strong>${escHtml(data.query)}</strong></div>
      <div class="answer-meta">${metaTags}</div>
      <hr class="answer-divider" />
      <div class="answer-text">${escHtml(data.answer)}</div>

      <div class="latency-section">
        <div class="latency-title">Latency breakdown &mdash; total: ${Math.round(totalMs)}ms</div>
        <div class="latency-bars">${latencyBars}</div>
      </div>

      <div class="sources-section">
        <div class="sources-header" onclick="toggleSources()">
          <span class="sources-title">Retrieved Sources</span>
          <span class="sources-count">${data.sources?.length ?? 0} chunks</span>
          <span class="toggle-icon" id="toggle-icon">▼</span>
        </div>
        <div class="sources-list" id="sources-list">${sourceCards}</div>
      </div>
    </div>`;
}

function renderError(msg) {
  document.getElementById('results-area').innerHTML = `
    <div class="answer-card" style="border-color: var(--red)33">
      <div class="answer-text" style="color: var(--red)">
        Error: ${escHtml(msg)}
      </div>
    </div>`;
}

// ── UI helpers ────────────────────────────────────────────────────────────

function setLoading(on) {
  document.getElementById('loading-overlay').classList.toggle('active', on);
  document.getElementById('send-btn').disabled = on;
  if (on) {
    document.getElementById('placeholder')?.remove();
  }
}

function toggleSources() {
  const list = document.getElementById('sources-list');
  const icon = document.getElementById('toggle-icon');
  const hidden = list.style.display === 'none';
  list.style.display = hidden ? '' : 'none';
  icon.textContent = hidden ? '▼' : '▶';
  icon.style.transform = hidden ? '' : 'rotate(-90deg)';
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
