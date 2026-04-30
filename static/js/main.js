/* main.js — ChurnShield frontend logic */

/* ── Nav scroll effect ─────────────────────────────────────── */
window.addEventListener('scroll', () => {
  document.getElementById('nav')?.classList.toggle('scrolled', window.scrollY > 10);
});

/* ── Range input live value update ────────────────────────── */
document.querySelectorAll('input[type=range]').forEach(r => {
  const display = document.getElementById(r.dataset.display);
  if (display) {
    const prefix = r.dataset.prefix || '';
    const suffix = r.dataset.suffix || '';
    display.textContent = prefix + r.value + suffix;
    r.addEventListener('input', () => {
      display.textContent = prefix + r.value + suffix;
    });
  }
});

/* ── Prediction form submit ────────────────────────────────── */
const predictBtn = document.getElementById('predict-btn');
if (predictBtn) {
  predictBtn.addEventListener('click', async () => {
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="spinner"></span>Predicting…';

    const payload = {
      Gender:           document.getElementById('f-gender')?.value,
      SeniorCitizen:    document.getElementById('f-senior')?.value,
      Partner:          document.getElementById('f-partner')?.value,
      Dependents:       document.getElementById('f-dependents')?.value,
      Tenure:           document.getElementById('f-tenure')?.value,
      Age:              document.getElementById('f-age')?.value,
      PhoneService:     document.getElementById('f-phone')?.value,
      MultipleLines:    document.getElementById('f-lines')?.value,
      InternetService:  document.getElementById('f-internet')?.value,
      OnlineSecurity:   document.getElementById('f-security')?.value,
      OnlineBackup:     document.getElementById('f-backup')?.value,
      DeviceProtection: document.getElementById('f-device')?.value,
      TechSupport:      document.getElementById('f-techsupport')?.value,
      StreamingTV:      document.getElementById('f-tv')?.value,
      StreamingMovies:  document.getElementById('f-movies')?.value,
      Contract:         document.getElementById('f-contract')?.value,
      PaperlessBilling: document.getElementById('f-paperless')?.value,
      PaymentMethod:    document.getElementById('f-payment')?.value,
      MonthlyCharges:   document.getElementById('f-monthly')?.value,
      TotalCharges:     document.getElementById('f-total')?.value,
      NumComplaints:    document.getElementById('f-complaints')?.value,
      SupportCalls:     document.getElementById('f-calls')?.value,
    };

    try {
      const res  = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      renderResult(data);
    } catch (err) {
      alert('Prediction failed: ' + err.message);
    } finally {
      predictBtn.disabled = false;
      predictBtn.innerHTML = '⚡ Predict Churn Probability';
    }
  });
}

function renderResult(data) {
  const prob  = data.probability_pct;
  const high  = data.prediction === 1;
  const cls   = high ? 'high' : 'low';

  // Hero
  const hero = document.getElementById('result-hero');
  if (hero) {
    hero.className = 'result-hero fade-up ' + cls;
    document.getElementById('result-icon').textContent  = high ? '🚨' : '✅';
    const pctEl = document.getElementById('result-pct');
    pctEl.className = 'result-pct ' + cls;
    pctEl.textContent = prob + '%';
    const badge = document.getElementById('result-badge');
    badge.className = 'result-badge ' + cls;
    badge.textContent = high ? 'HIGH CHURN RISK' : 'LOW CHURN RISK';
    document.getElementById('result-sub').textContent = high
      ? `This customer has a ${prob}% probability of churning. Immediate retention action is recommended.`
      : `This customer has a ${prob}% churn probability. The relationship appears stable.`;
    document.getElementById('model-info').textContent =
      `${data.model_name} · AUC ${data.model_auc}`;
  }

  // Gauge
  const fill  = document.getElementById('gauge-fill');
  const thumb = document.getElementById('gauge-thumb');
  if (fill) {
    fill.style.width = prob + '%';
    fill.style.background = prob > 70 ? 'var(--red)' : prob > 40 ? 'var(--amber)' : 'var(--teal)';
  }
  if (thumb) {
    thumb.style.left = prob + '%';
    thumb.style.borderColor = prob > 70 ? 'var(--red)' : prob > 40 ? 'var(--amber)' : 'var(--teal)';
  }
  const gScore = document.getElementById('gauge-score');
  if (gScore) gScore.textContent = prob + '%';

  // Risk factors
  const grid = document.getElementById('risk-grid');
  if (grid && data.risk_factors) {
    grid.innerHTML = data.risk_factors.map(f => `
      <div class="risk-item ${f.risk ? 'danger' : 'safe'}">
        <div class="risk-dot ${f.risk ? 'danger' : 'safe'}"></div>
        <span>${f.label}</span>
      </div>`).join('');
  }

  // Scroll to result on mobile
  if (window.innerWidth < 960) {
    document.getElementById('result-hero')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

/* ── Customer table fetch ──────────────────────────────────── */
let currentPage = 1;
async function loadCustomers(page = 1) {
  currentPage = page;
  const churn    = document.getElementById('filter-churn')?.value || '';
  const contract = document.getElementById('filter-contract')?.value || '';
  const internet = document.getElementById('filter-internet')?.value || '';
  const search   = document.getElementById('filter-search')?.value || '';

  const params = new URLSearchParams({ page, per_page: 50, churn, contract, internet, search });
  const res  = await fetch('/api/customers?' + params);
  const data = await res.json();

  document.getElementById('row-count').textContent =
    `Showing ${Math.min(50, data.total).toLocaleString()} of ${data.total.toLocaleString()} customers`;

  const tbody = document.getElementById('table-body');
  if (!tbody) return;
  tbody.innerHTML = data.records.map(r => `
    <tr>
      <td class="mono">${r.CustomerID}</td>
      <td>${r.Contract}</td>
      <td>${r.InternetService}</td>
      <td>${r.Tenure} mo</td>
      <td>$${Number(r.MonthlyCharges).toFixed(2)}</td>
      <td>$${Number(r.TotalCharges || 0).toFixed(0)}</td>
      <td>${r.NumComplaints}</td>
      <td><span class="badge ${r.Churn ? 'yes' : 'no'}">${r.Churn ? '● Churned' : '● Retained'}</span></td>
    </tr>`).join('');

  // Pagination
  const pag = document.getElementById('pagination');
  if (pag) {
    pag.innerHTML = '';
    for (let p = 1; p <= data.total_pages; p++) {
      if (p > 8) { pag.innerHTML += `<span style="color:var(--muted);font-size:.8rem">…${data.total_pages} pages</span>`; break; }
      const btn = document.createElement('button');
      btn.className = 'page-btn' + (p === page ? ' active' : '');
      btn.textContent = p;
      btn.onclick = () => loadCustomers(p);
      pag.appendChild(btn);
    }
  }
}

/* ── EDA charts (Chart.js) ─────────────────────────────────── */
function initEDACharts(stats) {
  if (!window.Chart) return;

  const defaults = {
    responsive: true,
    plugins: { legend: { display: false } },
  };
  const gridColor = 'rgba(0,0,0,0.05)';
  const font = { family: 'Sora' };

  // Contract bar
  const cEl = document.getElementById('chart-contract');
  if (cEl) {
    const labels = Object.keys(stats.by_contract);
    const values = Object.values(stats.by_contract);
    const colors = values.map(v => v > 20 ? '#e8364a' : v > 10 ? '#f59e0b' : '#00b09b');
    new Chart(cEl, {
      type: 'bar',
      data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 8, borderSkipped: false }] },
      options: { ...defaults, scales: { x: { grid: { display: false }, ticks: { font } }, y: { grid: { color: gridColor }, ticks: { callback: v => v + '%', font }, suggestedMax: 50 } } }
    });
  }

  // Payment bar
  const pEl = document.getElementById('chart-payment');
  if (pEl) {
    const labels = Object.keys(stats.by_payment);
    const values = Object.values(stats.by_payment);
    const colors = values.map(v => v > 25 ? '#e8364a' : v > 22 ? '#f59e0b' : '#00b09b');
    new Chart(pEl, {
      type: 'bar',
      data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 8, borderSkipped: false }] },
      options: { ...defaults, scales: { x: { grid: { display: false }, ticks: { font, maxRotation: 20 } }, y: { grid: { color: gridColor }, ticks: { callback: v => v + '%', font }, suggestedMax: 40 } } }
    });
  }

  // Internet bar
  const iEl = document.getElementById('chart-internet');
  if (iEl) {
    const labels = Object.keys(stats.by_internet);
    const values = Object.values(stats.by_internet);
    const colors = ['#e8364a', '#f59e0b', '#00b09b'];
    new Chart(iEl, {
      type: 'bar',
      data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 8, borderSkipped: false }] },
      options: { ...defaults, scales: { x: { grid: { display: false }, ticks: { font } }, y: { grid: { color: gridColor }, ticks: { callback: v => v + '%', font }, suggestedMax: 40 } } }
    });
  }

  // Donut
  const dEl = document.getElementById('chart-donut');
  if (dEl) {
    new Chart(dEl, {
      type: 'doughnut',
      data: {
        labels: ['Retained', 'Churned'],
        datasets: [{ data: [stats.retained, stats.churned], backgroundColor: ['#00b09b', '#e8364a'], borderWidth: 0, hoverOffset: 6 }]
      },
      options: { responsive: true, cutout: '70%', plugins: { legend: { position: 'bottom', labels: { font, padding: 14 } } } }
    });
  }

  // Retained vs Churned comparison
  const cmpEl = document.getElementById('chart-compare');
  if (cmpEl && stats.retained_stats && stats.churned_stats) {
    new Chart(cmpEl, {
      type: 'bar',
      data: {
        labels: ['Avg Tenure (mo)', 'Monthly Charges ($)', 'Total Charges ($100)'],
        datasets: [
          { label: 'Retained', data: [stats.retained_stats.avg_tenure, stats.retained_stats.avg_monthly, (stats.retained_stats.avg_total || 0) / 100], backgroundColor: '#00b09b', borderRadius: 6 },
          { label: 'Churned',  data: [stats.churned_stats.avg_tenure,  stats.churned_stats.avg_monthly,  (stats.churned_stats.avg_total  || 0) / 100], backgroundColor: '#e8364a', borderRadius: 6 }
        ]
      },
      options: { responsive: true, plugins: { legend: { position: 'top', labels: { font } } }, scales: { x: { grid: { display: false }, ticks: { font } }, y: { grid: { color: gridColor }, ticks: { font } } } }
    });
  }
}

/* ── Model comparison charts ───────────────────────────────── */
function initModelCharts() {
  if (!window.Chart) return;
  const font = { family: 'Sora' };
  const gridColor = 'rgba(0,0,0,0.05)';

  const aucEl = document.getElementById('chart-auc');
  if (aucEl) {
    new Chart(aucEl, {
      type: 'bar',
      data: {
        labels: ['Logistic Reg.', 'Random Forest', 'Grad. Boost', 'SVM'],
        datasets: [
          { label: 'Test AUC', data: [0.775, 0.772, 0.761, 0.750], backgroundColor: ['#0b0c10','#52546a','#7f8199','#c2c4d6'], borderRadius: 6 },
          { label: 'CV AUC',   data: [0.760, 0.747, 0.727, 0.728], backgroundColor: ['#e8364a','#f59e0b','#f59e0b','#c2c4d6'], borderRadius: 6 }
        ]
      },
      options: { responsive: true, plugins: { legend: { position: 'top', labels: { font } } }, scales: { x: { grid: { display: false }, ticks: { font } }, y: { min: 0.6, grid: { color: gridColor }, ticks: { font } } } }
    });
  }

  const featEl = document.getElementById('chart-features');
  if (featEl) {
    new Chart(featEl, {
      type: 'bar',
      data: {
        labels: ['Contract M2M', 'Tenure', 'Fiber optic', 'Electronic check', 'Senior Citizen', 'Complaints', 'Support Calls', 'Paperless Billing', 'Monthly Charges', 'ServiceCount'],
        datasets: [{ data: [1.25, 0.60, 0.55, 0.45, 0.25, 0.35, 0.28, 0.30, 0.20, 0.18], backgroundColor: '#0b0c10', borderRadius: 4 }]
      },
      options: {
        indexAxis: 'y', responsive: true,
        plugins: { legend: { display: false } },
        scales: { x: { grid: { color: gridColor }, ticks: { font } }, y: { grid: { display: false }, ticks: { font: { ...font, size: 11 } } } }
      }
    });
  }
}
