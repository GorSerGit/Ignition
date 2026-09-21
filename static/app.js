let network = null;
let isParsing = false;

document.addEventListener('DOMContentLoaded', function() {
    // Инициализация
    loadDatabase();
    loadHyperparams();
    setInterval(refreshLogs, 5000);
    setInterval(checkParseStatus, 2000);

    // Обработчики событий
    document.getElementById('refreshStatsBtn').addEventListener('click', updateStats);
    document.getElementById('searchGraphBtn').addEventListener('click', () => loadGraph(document.getElementById('graphSearch').value));
    document.getElementById('refreshGraphBtn').addEventListener('click', () => loadGraph());
    document.getElementById('exportGraphBtn').addEventListener('click', exportGraph);
    
    document.getElementById('askBtn').addEventListener('click', askQuestion);
    document.getElementById('questionInput').addEventListener('keypress', (e) => { if(e.key==='Enter') askQuestion(); });

    document.getElementById('loadCorpusBtn').addEventListener('click', loadCorpus);
    document.getElementById('startParseBtn').addEventListener('click', startParsing);

    document.getElementById('saveHyperparamsBtn').addEventListener('click', saveHyperparams);
    document.getElementById('refreshLogBtn').addEventListener('click', refreshLogs);

    // Обработчик добавления факта
    document.getElementById('saveFactBtn').addEventListener('click', addFact);
});

// ==========================================
// УТИЛИТЫ И УВЕДОМЛЕНИЯ
// ==========================================
function showToast(message, type = 'info') {
    const toastEl = document.getElementById('appToast');
    const toastTitle = document.getElementById('toastTitle');
    const toastBody = document.getElementById('toastBody');
    
    toastBody.textContent = message;
    toastTitle.textContent = type === 'success' ? '✅ Успех' : type === 'danger' ? '❌ Ошибка' : 'ℹ️ Инфо';
    
    toastEl.className = `toast border-${type}`;
    const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
    toast.show();
}

// ==========================================
// БАЗА ДАННЫХ И СТАТИСТИКА
// ==========================================
function loadDatabase() {
    fetch('/api/load_db', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({path: ''}) })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'ok') {
            updateStatsDisplay(data.stats);
            loadGraph();
            showToast('База данных загружена', 'success');
        } else {
            showToast('Ошибка загрузки БД: ' + data.message, 'danger');
        }
    }).catch(err => showToast('Ошибка сети', 'danger'));
}

function updateStats() {
    fetch('/api/stats').then(res => res.json()).then(data => {
        if (data.status === 'ok') updateStatsDisplay(data.stats);
    });
}

function updateStatsDisplay(stats) {
    document.getElementById('statsLabel').textContent = `S=${stats.S} C=${stats.C} P=${stats.P} H=${stats.H} L=${stats.L}`;
}

// ==========================================
// ГРАФ И ВИЗУАЛИЗАЦИЯ
// ==========================================
function loadGraph(search = '', focusUid = null) {
    const url = '/api/graph' + (search ? `?search=${encodeURIComponent(search)}` : '');
    fetch(url).then(res => res.json()).then(data => {
        if (data.status === 'ok') {
            renderGraph(data.nodes, data.edges, focusUid);
        } else {
            showToast('Ошибка графа: ' + data.message, 'danger');
        }
    });
}

function renderGraph(nodes, edges, focusUid = null) {
    const container = document.getElementById('graphContainer');
    const data = {
        nodes: new vis.DataSet(nodes.map(n => ({ ...n, color: n.type === 'fact' ? '#90EE90' : '#87CEEB' }))),
        edges: new vis.DataSet(edges)
    };

    const options = {
        nodes: { shape: 'dot', size: 16, font: { size: 14, face: 'Arial', color: '#333' }, borderWidth: 2 },
        edges: { smooth: { type: 'continuous' }, arrows: { to: { enabled: true, scaleFactor: 0.5 } }, font: { size: 10, align: 'middle', strokeWidth: 2, color: '#fff' }, color: { color: '#848484', highlight: '#333' } },
        physics: { stabilization: { iterations: 150 }, barnesHut: { gravitationalConstant: -4000, springLength: 120 } },
        interaction: { hover: true, tooltipDelay: 200 }
    };

    network = new vis.Network(container, data, options);

    // 🎯 ВИЗУАЛИЗАЦИЯ НОВОГО ФАКТА: Фокусируемся на новом узле
    if (focusUid) {
        network.once('stabilizationDone', function() {
            network.focus(focusUid, { scale: 1.5, animation: { duration: 1000, easingFunction: "easeInOutQuad" } });
            network.selectNodes([focusUid]);
        });
    }
}

function exportGraph() {
    if (!network) return showToast('Граф не загружен', 'warning');
    const canvas = network.canvas.frame.canvas;
    const link = document.createElement('a');
    link.download = 'graph.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
}

// ==========================================
// ДОБАВЛЕНИЕ НОВОГО ФАКТА
// ==========================================
function addFact() {
    const predicate = document.getElementById('factPredicate').value.trim();
    const template = document.getElementById('factTemplate').value;
    const subject = document.getElementById('factSubject').value.trim();
    const object = document.getElementById('factObject').value.trim();

    if (!predicate || !subject || !object) {
        return showToast('Заполните Предикат, Субъект и Объект', 'warning');
    }

    fetch('/api/add_fact', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ predicate, template, subject, object })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'ok') {
            showToast('Факт успешно добавлен в память!', 'success');
            bootstrap.Modal.getInstance(document.getElementById('addFactModal')).hide();
            document.getElementById('addFactForm').reset();
            
            // Перезагружаем граф и фокусируемся на новом узле (Факте)
            loadGraph('', data.uid); 
            updateStats();
        } else {
            showToast('Ошибка: ' + data.message, 'danger');
        }
    })
    .catch(err => showToast('Ошибка сети', 'danger'));
}

// ==========================================
// АГЕНТ
// ==========================================
function askQuestion() {
    const question = document.getElementById('questionInput').value.trim();
    if (!question) return showToast('Введите вопрос', 'warning');

    document.getElementById('traceOutput').textContent = 'Воспламенение...';
    document.getElementById('answerOutput').innerHTML = '<span class="text-muted">Генерация ответа...</span>';

    fetch('/api/ask', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ question })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'ok') {
            document.getElementById('traceOutput').textContent = data.trace.join('\n');
            document.getElementById('answerOutput').innerHTML = `<p>${data.answer.replace(/\n/g, '<br>')}</p>`;
        } else {
            document.getElementById('answerOutput').innerHTML = `<span class="text-danger">${data.message}</span>`;
        }
    });
}

// ==========================================
// ПАРСИНГ
// ==========================================
function loadCorpus() {
    const path = document.getElementById('corpusPath').value.trim();
    if (!path) return showToast('Укажите путь', 'warning');
    
    fetch('/api/load_corpus', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({path}) })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'ok') showToast(`Корпус загружен: ${data.word_count} слов`, 'success');
        else showToast(data.message, 'danger');
    });
}

function startParsing() {
    if (isParsing) return;
    const limit = parseInt(document.getElementById('parseLimit').value) || 0;
    
    fetch('/api/parse', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text: '', limit}) })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'ok') {
            isParsing = true;
            showToast('Парсинг запущен', 'info');
        } else {
            showToast(data.message, 'danger');
        }
    });
}

function checkParseStatus() {
    if (!isParsing) return;
    fetch('/api/parse_status').then(res => res.json()).then(data => {
        const bar = document.getElementById('parseProgress');
        bar.style.width = data.progress + '%';
        bar.textContent = data.progress + '%';
        document.getElementById('parseLogOutput').textContent = data.log;
        
        if (!data.running && data.progress === 100) {
            isParsing = false;
            showToast('Парсинг завершен!', 'success');
            updateStats();
            loadGraph();
        }
    });
}

// ==========================================
// НАСТРОЙКИ И ЛОГИ
// ==========================================
function loadHyperparams() {
    fetch('/api/hyperparams').then(res => res.json()).then(data => {
        if (data.status === 'ok') {
            const container = document.getElementById('hyperparamsForm');
            container.innerHTML = '';
            for (const [key, value] of Object.entries(data.params)) {
                container.innerHTML += `
                    <div class="col-md-4">
                        <label class="form-label">${key}</label>
                        <input type="number" step="any" class="form-control" id="hp_${key}" value="${value}">
                    </div>`;
            }
        }
    });
}

function saveHyperparams() {
    const inputs = document.querySelectorAll('#hyperparamsForm input');
    const params = {};
    inputs.forEach(inp => params[inp.id.replace('hp_', '')] = parseFloat(inp.value));
    
    fetch('/api/hyperparams', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(params) })
    .then(res => res.json())
    .then(data => showToast(data.status === 'ok' ? 'Сохранено' : data.message, data.status === 'ok' ? 'success' : 'danger'));
}

function refreshLogs() {
    fetch('/api/logs').then(res => res.json()).then(data => {
        const div = document.getElementById('logOutput');
        div.textContent = data.logs.join('\n');
        div.scrollTop = div.scrollHeight;
    });
}