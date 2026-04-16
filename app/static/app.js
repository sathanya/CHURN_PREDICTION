document.addEventListener('DOMContentLoaded', () => {
    
    // --- TOP LEVEL TAB LOGIC ---
    const navBtns = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            navBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(btn.dataset.target).classList.add('active');
        });
    });

    // --- CHART GLOBALS ---
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', sans-serif";
    
    let lstmChartInstance = null;
    let rocChartInstance = null;
    let prChartInstance = null;

    // FORMATTERS
    const moneyFmt = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
    const pctFmt = new Intl.NumberFormat('en-US', { style: 'percent', maximumFractionDigits: 1 });

    // --- TAB 1: SUMMARY ---
    async function initSummary() {
        try {
            const res = await fetch('/api/summary');
            const data = await res.json();
            
            document.getElementById('val-total-customers').innerText = data.total_users.toLocaleString();
            document.getElementById('val-leaving-rate').innerText = pctFmt.format(data.overall_leaving_rate);
            document.getElementById('val-money-at-risk').innerText = moneyFmt.format(data.valuable_at_risk);
            document.getElementById('val-risk-pool-size').innerText = `Affecting ${data.at_risk_pool_size.toLocaleString()} nodes`;

            const tbody = document.querySelector('#top-customers-table tbody');
            tbody.innerHTML = '';
            
            data.top_customers.slice(0, 10).forEach(u => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${u.CustomerID}</td>
                    <td class="text-amber"><strong>${pctFmt.format(u.Churn_Probability)}</strong></td>
                    <td>${moneyFmt.format(u.total_spent)}</td>
                    <td><button class="nav-btn" style="padding:4px 8px; font-size:0.8rem">Isolate</button></td>
                    <td>${u.days_since_last_order}</td>
                `;
                tbody.appendChild(tr);
            });
        } catch (e) {
            console.error(e);
        }
    }

    // --- TAB 2: LSTM ---
    async function initLSTM() {
        try {
            const res = await fetch('/api/lstm_risk');
            const data = await res.json();
            
            if(!data.datasets || data.datasets.length === 0) return;

            const ctx = document.getElementById('lstmChart').getContext('2d');
            
            const chartDatasets = data.datasets.map((d, i) => {
                // Generate a neon hue
                const hue = (i * 60) % 360;
                return {
                    label: `${d.id} (Risk: ${pctFmt.format(d.risk_score)})`,
                    data: d.spend_trend,
                    borderColor: `hsl(${hue}, 100%, 60%)`,
                    backgroundColor: `hsla(${hue}, 100%, 60%, 0.1)`,
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#0f172a',
                    pointBorderColor: `hsl(${hue}, 100%, 60%)`,
                    pointRadius: 4,
                    pointHoverRadius: 6
                };
            });

            lstmChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.months || ["M-5", "M-4", "M-3", "M-2", "M-1", "Now"],
                    datasets: chartDatasets
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: 'right' }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { 
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            title: { display: true, text: 'Normalized Spend Velocity' }
                        }
                    }
                }
            });
        } catch (e) { console.error(e); }
    }

    // --- TAB 3: CAUSAL NUDGE ---
    const simDisc = document.getElementById('sim-discount');
    const outDisc = document.getElementById('out-disc');

    async function updateCausalSim() {
        outDisc.innerText = simDisc.value;
        try {
            const res = await fetch(`/api/causal_nudge?discount=${simDisc.value}`);
            const data = await res.json();
            
            document.getElementById('roi-saved').innerText = `${data.rescued.toLocaleString()} / ${data.pool_size.toLocaleString()}`;
            document.getElementById('roi-gross').innerText = moneyFmt.format(data.revenue_saved);
            document.getElementById('val-uplift').innerText = moneyFmt.format(data.uplift_mean);
            document.getElementById('roi-net').innerText = moneyFmt.format(data.net_profit);
            document.getElementById('roi-cost').innerText = `-${moneyFmt.format(data.cost)} Op. Cost`;
        } catch(e) { console.error(e); }
    }

    simDisc.addEventListener('input', updateCausalSim);

    // --- TAB 4: BASKET ---
    const basketSelect = document.getElementById('basket-select');
    async function initBasket() {
        const res = await fetch('/api/market_basket/items');
        const data = await res.json();
        basketSelect.innerHTML = '<option value="">Awaiting Node Selection...</option>';
        data.items.forEach(i => {
            basketSelect.innerHTML += `<option value="${i}">${i}</option>`;
        });
    }

    basketSelect.addEventListener('change', async (e) => {
        const item = e.target.value;
        if(!item){
            document.getElementById('basket-result').style.display = 'none';
            return;
        }

        const res = await fetch(`/api/market_basket/${encodeURIComponent(item)}`);
        const data = await res.json();
        
        if (data.matches.length > 0) {
            document.getElementById('basket-result').style.display = 'block';
            const best = data.matches[0];
            document.getElementById('basket-best-match').innerText = best.consequents;
            document.getElementById('basket-conf').innerText = (best.confidence * 100).toFixed(1);
            document.getElementById('basket-lift').innerText = best.lift.toFixed(2);
            
            const tbody = document.querySelector('#basket-table tbody');
            tbody.innerHTML = '';
            data.matches.slice(0,5).forEach(m => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${m.consequents}</strong></td>
                    <td>${(m.confidence * 100).toFixed(1)}%</td>
                    <td>${m.lift.toFixed(2)}</td>
                `;
                tbody.appendChild(tr);
            });
        }
    });

    // --- TAB 5: METRICS ---
    async function initMetrics() {
        try {
            const res = await fetch('/api/metrics');
            const data = await res.json();
            
            const ctx1 = document.getElementById('rocChart').getContext('2d');
            rocChartInstance = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: data.roc.fpr,
                    datasets: [
                        {
                            label: `RF AUC = ${data.roc.auc.toFixed(2)}`,
                            data: data.roc.tpr,
                            borderColor: '#8b5cf6',
                            backgroundColor: 'rgba(139, 92, 246, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            pointRadius: 0
                        },
                        {
                            label: 'Random',
                            data: data.roc.fpr,
                            borderColor: '#4d4d4d',
                            borderDash: [5, 5],
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { grid: { color: 'rgba(255,255,255,0.05)' } }
                    }
                }
            });

            const ctx2 = document.getElementById('prChart').getContext('2d');
            prChartInstance = new Chart(ctx2, {
                type: 'line',
                data: {
                    labels: data.pr.recall,
                    datasets: [
                        {
                            label: `PR AUC = ${data.pr.auc.toFixed(2)}`,
                            data: data.pr.precision,
                            borderColor: '#06b6d4',
                            backgroundColor: 'rgba(6, 182, 212, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { grid: { color: 'rgba(255,255,255,0.05)' } }
                    }
                }
            });
        } catch (e) {
            console.error(e);
        }
    }

    // INITIALIZE APP
    initSummary();
    initLSTM();
    updateCausalSim();
    initBasket();
    initMetrics();
});
