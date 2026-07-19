/* shared.js — Otter Streams Docs — cross-page enhancements
 * Additive only: does not replace any per-page inline <script> (copy-to-clipboard,
 * sidebar scrollspy, diagram zoom, etc.) already present on individual pages.
 */
(function () {
  'use strict';

  /* ────────────────────────────────────────────────────────────
   * Search index
   * Hand-curated (no build step / backend on this static site).
   * Each entry: { title, page, anchor, section }
   * ──────────────────────────────────────────────────────────── */
  var OTTER_SEARCH_INDEX = [
    { title: 'Introduction', page: 'index.html', anchor: 'intro', section: 'Documentation' },
    { title: 'Quick Start', page: 'index.html', anchor: 'quickstart', section: 'Documentation' },
    { title: 'Architecture', page: 'index.html', anchor: 'architecture', section: 'Documentation' },
    { title: 'UDF Class Reference — MLInferenceFunction', page: 'index.html', anchor: 'sql-udf', section: 'Documentation' },
    { title: 'Registering the UDF in Studio', page: 'index.html', anchor: 'sql-registration', section: 'Documentation' },
    { title: 'SQL Usage Examples', page: 'index.html', anchor: 'sql-usage', section: 'Documentation' },
    { title: 'Table Connector — ml-inference', page: 'index.html', anchor: 'connector', section: 'Documentation' },
    { title: 'Building the Shaded JAR', page: 'index.html', anchor: 'shaded-jar', section: 'Documentation' },
    { title: 'Troubleshooting', page: 'index.html', anchor: 'troubleshooting', section: 'Documentation' },
    { title: 'Error Reference', page: 'index.html', anchor: 'error-reference', section: 'Documentation' },

    { title: 'InferenceEngine<C>', page: 'api.html', anchor: 'inference-engine', section: 'API Reference' },
    { title: 'InferenceResult', page: 'api.html', anchor: 'inference-result', section: 'API Reference' },
    { title: 'InferenceException', page: 'api.html', anchor: 'inference-exception', section: 'API Reference' },
    { title: 'ModelCache', page: 'api.html', anchor: 'model-cache', section: 'API Reference' },
    { title: 'InferenceConfig', page: 'api.html', anchor: 'inference-config', section: 'API Reference' },
    { title: 'ModelConfig', page: 'api.html', anchor: 'model-config', section: 'API Reference' },
    { title: 'MLInferenceFunction', page: 'api.html', anchor: 'ml-inference-fn', section: 'API Reference' },
    { title: 'MLInferenceLookupFunction', page: 'api.html', anchor: 'lookup-fn', section: 'API Reference' },
    { title: 'MLInferenceDynamicTableFactory', page: 'api.html', anchor: 'table-factory', section: 'API Reference' },
    { title: 'SqlInferenceConfig', page: 'api.html', anchor: 'sql-config', section: 'API Reference' },
    { title: 'Connector DDL Options', page: 'api.html', anchor: 'connector-options', section: 'API Reference' },

    { title: 'Dependency Graph', page: 'modules.html', anchor: 'dependency-graph', section: 'Modules' },
    { title: 'ml-inference-core', page: 'modules.html', anchor: 'core', section: 'Modules' },
    { title: 'Runtime Layer — OtterRuntime', page: 'modules.html', anchor: 'runtime', section: 'Modules' },
    { title: 'Dynamic Loading & Rollback', page: 'modules.html', anchor: 'dynamic-loading', section: 'Modules' },
    { title: 'Shadow & Canary Deployments', page: 'modules.html', anchor: 'canary-shadow', section: 'Modules' },
    { title: 'Feature Store Providers', page: 'modules.html', anchor: 'feature-stores', section: 'Modules' },
    { title: 'otter-stream-sql', page: 'modules.html', anchor: 'sql', section: 'Modules' },
    { title: 'otter-stream-onnx', page: 'modules.html', anchor: 'onnx', section: 'Modules' },
    { title: 'otter-stream-tensorflow', page: 'modules.html', anchor: 'tensorflow', section: 'Modules' },
    { title: 'otter-streams-xgboost', page: 'modules.html', anchor: 'xgboost', section: 'Modules' },
    { title: 'otter-stream-pmml', page: 'modules.html', anchor: 'pmml', section: 'Modules' },
    { title: 'otter-stream-remote', page: 'modules.html', anchor: 'remote', section: 'Modules' },

    { title: 'Rule Engine Overview', page: 'rules.html', anchor: 'overview', section: 'Rule Engine' },
    { title: 'Rule Engine Quick Start', page: 'rules.html', anchor: 'quickstart', section: 'Rule Engine' },
    { title: 'Rule Engine Architecture', page: 'rules.html', anchor: 'architecture', section: 'Rule Engine' },
    { title: 'Rule Configuration Formats (YAML/properties/class)', page: 'rules.html', anchor: 'config-formats', section: 'Rule Engine' },
    { title: 'Rule Evaluation Modes (single/multiple/batch)', page: 'rules.html', anchor: 'evaluation-modes', section: 'Rule Engine' },
    { title: 'External Decision Engine Connectors (Drools, KIE, DMN)', page: 'rules.html', anchor: 'connectors', section: 'Rule Engine' },
    { title: 'Rule Dashboard & Metrics', page: 'rules.html', anchor: 'dashboard', section: 'Rule Engine' },
    { title: 'Publishing Decisions to Kafka', page: 'rules.html', anchor: 'kafka', section: 'Rule Engine' },

    { title: 'MinIO Pipeline Demo', page: 'examples.html', anchor: 'minio', section: 'Examples' },
    { title: 'Prerequisites', page: 'examples.html', anchor: 'prerequisites', section: 'Examples' },
    { title: 'MinIO Configuration', page: 'examples.html', anchor: 'minio-config', section: 'Examples' },
    { title: 'Model Preloading', page: 'examples.html', anchor: 'preload', section: 'Examples' },
    { title: 'Full Streaming Pipeline', page: 'examples.html', anchor: 'pipeline-sql', section: 'Examples' },
    { title: 'Fraud Detection — Extended Example', page: 'examples.html', anchor: 'fraud', section: 'Examples' },
    { title: 'Anomaly Detection — IoT Sensors', page: 'examples.html', anchor: 'anomaly', section: 'Examples' },

    { title: 'Release v0.0.4', page: 'releases.html', anchor: 'v0-0-4', section: 'Releases' },
    { title: 'Compatibility Matrix', page: 'releases.html', anchor: 'compat', section: 'Releases' },
    { title: 'Upgrade Notes', page: 'releases.html', anchor: 'upgrade', section: 'Releases' },

    { title: 'DataStream Overview', page: 'datastream.html', anchor: 'overview', section: 'DataStream API' },
    { title: 'AsyncModelInferenceFunction', page: 'datastream.html', anchor: 'async-function', section: 'DataStream API' },
    { title: 'ModelCache Integration', page: 'datastream.html', anchor: 'model-cache', section: 'DataStream API' },
    { title: 'Full DataStream Pipeline Example', page: 'datastream.html', anchor: 'full-pipeline', section: 'DataStream API' },

    { title: 'Studio Demo Overview', page: 'studio-demo.html', anchor: 'overview', section: 'Studio Demo' },
    { title: 'Feature Engineering Manager', page: 'studio-demo.html', anchor: 'feature-manager', section: 'Studio Demo' },
    { title: 'Inference Manager', page: 'studio-demo.html', anchor: 'inference-manager', section: 'Studio Demo' },
    { title: 'End-to-End Studio Walkthrough', page: 'studio-demo.html', anchor: 'e2e', section: 'Studio Demo' }
  ];

  /* ────────────────────────────────────────────────────────────
   * Search palette (⌘K / Ctrl+K)
   * ──────────────────────────────────────────────────────────── */
  function initSearch() {
    var box = document.getElementById('otter-search');
    if (!box) return;

    var overlay = document.createElement('div');
    overlay.className = 'search-overlay';
    overlay.innerHTML =
      '<div class="search-modal" role="dialog" aria-label="Search documentation">' +
      '  <div class="search-modal-input-row">' +
      '    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>' +
      '    <input type="text" class="search-modal-input" placeholder="Search documentation…" autocomplete="off" spellcheck="false">' +
      '    <kbd class="search-kbd">Esc</kbd>' +
      '  </div>' +
      '  <div class="search-modal-results"></div>' +
      '</div>';
    document.body.appendChild(overlay);

    var input = overlay.querySelector('.search-modal-input');
    var results = overlay.querySelector('.search-modal-results');
    var activeIndex = -1;
    var currentMatches = [];

    function render(query) {
      var q = query.trim().toLowerCase();
      currentMatches = !q ? [] : OTTER_SEARCH_INDEX.filter(function (item) {
        return item.title.toLowerCase().indexOf(q) !== -1 || item.section.toLowerCase().indexOf(q) !== -1;
      }).slice(0, 20);
      activeIndex = currentMatches.length ? 0 : -1;

      if (!q) {
        results.innerHTML = '<div class="search-hint">Type to search across Documentation, API Reference, Modules, Examples, Releases, DataStream API, and Studio Demo.</div>';
        return;
      }
      if (!currentMatches.length) {
        results.innerHTML = '<div class="search-hint">No results for "' + escapeHtml(query) + '"</div>';
        return;
      }
      results.innerHTML = currentMatches.map(function (item, i) {
        return '<a class="search-result' + (i === 0 ? ' active' : '') + '" href="' + item.page + '#' + item.anchor + '" data-idx="' + i + '">' +
          '<span class="search-result-section">' + escapeHtml(item.section) + '</span>' +
          '<span class="search-result-title">' + escapeHtml(item.title) + '</span>' +
          '</a>';
      }).join('');
    }

    function escapeHtml(s) {
      return s.replace(/[&<>"']/g, function (c) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
      });
    }

    function setActive(idx) {
      var items = results.querySelectorAll('.search-result');
      items.forEach(function (el) { el.classList.remove('active'); });
      if (items[idx]) {
        items[idx].classList.add('active');
        items[idx].scrollIntoView({ block: 'nearest' });
      }
      activeIndex = idx;
    }

    function open() {
      overlay.classList.add('open');
      input.value = '';
      render('');
      setTimeout(function () { input.focus(); }, 0);
    }
    function close() {
      overlay.classList.remove('open');
    }

    box.addEventListener('click', open);
    document.addEventListener('keydown', function (e) {
      var isK = (e.key === 'k' || e.key === 'K') && (e.metaKey || e.ctrlKey);
      if (isK) { e.preventDefault(); overlay.classList.contains('open') ? close() : open(); }
      else if (e.key === '/' && document.activeElement !== input && !overlay.classList.contains('open')) {
        var tag = (document.activeElement && document.activeElement.tagName) || '';
        if (tag !== 'INPUT' && tag !== 'TEXTAREA') { e.preventDefault(); open(); }
      }
    });

    overlay.addEventListener('click', function (e) { if (e.target === overlay) close(); });
    input.addEventListener('input', function () { render(input.value); });
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { close(); return; }
      if (e.key === 'ArrowDown') { e.preventDefault(); if (currentMatches.length) setActive((activeIndex + 1) % currentMatches.length); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); if (currentMatches.length) setActive((activeIndex - 1 + currentMatches.length) % currentMatches.length); }
      else if (e.key === 'Enter') {
        var item = currentMatches[activeIndex];
        if (item) { window.location.href = item.page + '#' + item.anchor; }
      }
    });
  }

  /* ────────────────────────────────────────────────────────────
   * Heading permalinks — hover a content h2 to reveal a "#" link
   * ──────────────────────────────────────────────────────────── */
  function initHeadingAnchors() {
    document.querySelectorAll('.content-body h2[id]').forEach(function (h) {
      var a = document.createElement('a');
      a.className = 'heading-anchor';
      a.href = '#' + h.id;
      a.setAttribute('aria-label', 'Link to this section');
      a.textContent = '#';
      h.appendChild(a);
    });
  }

  /* ────────────────────────────────────────────────────────────
   * TOC scrollspy — highlights the active entry in the right-hand
   * "On this page" box as you scroll (separate from, and additive
   * to, any existing sidebar-nav scrollspy already on the page).
   * ──────────────────────────────────────────────────────────── */
  function initTocScrollspy() {
    var tocLinks = document.querySelectorAll('.toc-list a[href^="#"]');
    if (!tocLinks.length) return;
    var targets = [];
    tocLinks.forEach(function (link) {
      var id = link.getAttribute('href').slice(1);
      var el = document.getElementById(id);
      if (el) targets.push({ link: link, el: el });
    });
    if (!targets.length) return;

    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var match = targets.find(function (t) { return t.el === entry.target; });
        if (!match) return;
        if (entry.isIntersecting) {
          tocLinks.forEach(function (l) { l.classList.remove('toc-active'); });
          match.link.classList.add('toc-active');
        }
      });
    }, { rootMargin: '-15% 0px -70% 0px' });
    targets.forEach(function (t) { obs.observe(t.el); });
  }

  document.addEventListener('DOMContentLoaded', function () {
    initSearch();
    initHeadingAnchors();
    initTocScrollspy();
  });
})();
