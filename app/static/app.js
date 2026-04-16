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
                    <td><button class="nav-btn isolate-btn" data-id="${u.CustomerID}" style="padding:4px 8px; font-size:0.8rem; background:rgba(6,182,212,0.1); border:1px solid var(--cyan-glow);">Isolate</button></td>
                `;
                tbody.appendChild(tr);
            });
            
            // Bind isolate buttons
            document.querySelectorAll('.isolate-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const cid = e.target.getAttribute('data-id');
                    triggerAgent(cid);
                });
            });

        } catch (e) {
            console.error(e);
        }
    }

    async function triggerAgent(cid) {
        const terminal = document.getElementById('agent-terminal');
        terminal.innerHTML = '';
        const cursor = `<span class="pulse-dot" style="display:inline-block; margin-left:4px;"></span>`;
        terminal.innerHTML = `<div style="color:var(--text-main)">> Initiating Trace on Node ${cid}...${cursor}</div>`;
        
        try {
            const res = await fetch(`/api/generate_nudge/${cid}`);
            const data = await res.json();
            
            terminal.innerHTML = ''; // clear initial
            
            let delay = 0;
            // Type out the trace logs
            data.trace.forEach((line, index) => {
                setTimeout(() => {
                    terminal.innerHTML += `<div style="margin-bottom:4px;">${line}</div>`;
                    terminal.scrollTop = terminal.scrollHeight;
                }, delay);
                delay += (Math.random() * 400) + 200; // random hacker typing speed
            });
            
            // Output the final email payload
            setTimeout(() => {
                terminal.innerHTML += `
                <div style="margin-top:12px; padding:10px; background:rgba(16,185,129,0.1); border:1px solid var(--emerald); border-radius:4px; color:var(--text-main); white-space:pre-wrap; font-family: 'Inter', sans-serif;">${data.email}</div>
                `;
                terminal.innerHTML += `<div style="margin-top:10px; color:var(--text-muted)">> [SYSTEM] Payload delivered to outbound queue. Standing by.</div>`;
                terminal.scrollTop = terminal.scrollHeight;
            }, delay + 800);
            
        } catch(e) {
            terminal.innerHTML += `<div style="color:var(--red)">> [CRITICAL ERROR] Trace failed.</div>`;
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

    // --- STRESS TEST SLIDER ---
    const simStress = document.getElementById('sim-stress');
    const outStress = document.getElementById('out-stress');
    
    const baseChurnRate = { val: null };
    const baseCapital   = { val: null };

    async function updateStressTest() {
        const shock = parseInt(simStress.value);
        const factor = 1.0 - (shock / 100.0);
        outStress.innerText = shock === 0 ? 'Baseline (0% Drop)' : `-${shock}% Consumer Spending`;

        try {
            const res = await fetch(`/api/stress_test?factor=${factor}`);
            const data = await res.json();

            // Animate capital at risk going up
            const capEl = document.getElementById('val-money-at-risk');
            const rateEl = document.getElementById('val-leaving-rate');
            if (shock > 0) {
                capEl.innerText = moneyFmt.format(data.new_risk_capital);
                rateEl.innerText = pctFmt.format(data.new_churn_rate);
                document.getElementById('card-capital-risk').style.boxShadow = '0 0 20px rgba(239,68,68,0.5)';
                document.getElementById('card-leaving-rate').style.boxShadow = '0 0 20px rgba(239,68,68,0.5)';
            } else {
                // Reset to originals
                if (baseChurnRate.val !== null) {
                    capEl.innerText = moneyFmt.format(baseCapital.val);
                    rateEl.innerText = pctFmt.format(baseChurnRate.val);
                }
                document.getElementById('card-capital-risk').style.boxShadow = '';
                document.getElementById('card-leaving-rate').style.boxShadow = '';
            }
        } catch(e) { console.error(e); }
    }

    simStress.addEventListener('input', updateStressTest);

    // --- AFFINITY NETWORK GRAPH (vis-network) ---
    async function initAffinityNetwork() {
        try {
            const res = await fetch('/api/network_graph');
            const data = await res.json();
            if (!data.nodes || data.nodes.length === 0) return;

            const container = document.getElementById('affinity-network');

            // Format nodes with neon glow color
            const nodes = new vis.DataSet(data.nodes.map(n => ({
                id: n.id,
                label: n.label.length > 25 ? n.label.substring(0, 22) + '...' : n.label,
                title: n.label,
                color: {
                    background: '#0f172a',
                    border: '#06b6d4',
                    highlight: { background: '#0e7490', border: '#06b6d4' }
                },
                font: { color: '#e2e8f0', size: 11 },
                borderWidth: 2,
                shape: 'dot',
                size: 14
            })));

            const edges = new vis.DataSet(data.edges.map(e => ({
                from: e.from,
                to: e.to,
                title: e.title,
                width: Math.min(e.value, 5),
                color: { color: '#10b981', highlight: '#34d399' },
                arrows: { to: { enabled: true, scaleFactor: 0.5 } }
            })));

            const networkData = { nodes, edges };
            const options = {
                physics: {
                    enabled: true,
                    barnesHut: { gravitationalConstant: -4000, springLength: 150 }
                },
                interaction: { hover: true, tooltipDelay: 100 },
                layout: { randomSeed: 42 }
            };

            new vis.Network(container, networkData, options);
        } catch(e) { console.error(e); }
    }

    // --- THREAT DETECTION TABLE ---
    async function initThreatDetection() {
        try {
            const res = await fetch('/api/anomalies');
            const data = await res.json();
            
            const tbody = document.querySelector('#anomalies-table tbody');
            tbody.innerHTML = '';

            if (!data.anomalies || data.anomalies.length === 0) {
                tbody.innerHTML = '<tr><td colspan="3" style="color:var(--text-muted); text-align:center;">No threats detected in current dataset.</td></tr>';
                return;
            }

            data.anomalies.forEach(a => {
                const score = parseFloat(a.Anomaly_Score).toFixed(4);
                const tr = document.createElement('tr');
                tr.style.borderLeft = '3px solid var(--red)';
                tr.innerHTML = `
                    <td style="color:var(--red); font-weight:600;">⚠ ${a.CustomerID}</td>
                    <td style="color:var(--amber);">${score}</td>
                    <td style="font-size:0.8rem;">${a.Flag_Reason}</td>
                `;
                tbody.appendChild(tr);
            });
        } catch(e) { console.error(e); }
    }

    // --- PATCH initSummary to store baseline values ---
    const _origInitSummary = initSummary;

    // --- INITIALIZE APP ---
    initSummary().then(() => {
        // After summary loads, store the baseline numbers
        const capEl = document.getElementById('val-money-at-risk');
        const rateEl = document.getElementById('val-leaving-rate');
        // We'll just re-fetch to store them cleanly
        fetch('/api/summary').then(r => r.json()).then(d => {
            baseChurnRate.val = d.overall_leaving_rate;
            baseCapital.val   = d.valuable_at_risk;
        });
    });
    initLSTM();
    updateCausalSim();
    initBasket();
    initAffinityNetwork();
    initThreatDetection();
    initMetrics();
});
