// ──────────────────────────────────────────────
//  STATE
// ──────────────────────────────────────────────
let socket = null;
let agents = [];
let selectedAgent1 = null;
let selectedAgent2 = null;
let selectedSize = 3;
let selectedPlayer = 1;
let gameActive = false;
let boardSize = 3;
let humanPlayer = 1;
let moveCount = 0;
let lastMoveRC = null;
let previousScores = [0, 0];

const CELL_SIZE = 60;
const DOT_SIZE = 14;

// ──────────────────────────────────────────────
//  SOCKET CONNECTION
// ──────────────────────────────────────────────
function connectSocket() {
    socket = io();

    socket.on('connect', () => {
        document.getElementById('conn-dot').className = 'conn-dot connected';
    });

    socket.on('disconnect', () => {
        document.getElementById('conn-dot').className = 'conn-dot disconnected';
    });

    socket.on('agents_list', (data) => {
        agents = data;
        buildAgentGrid();
    });

    socket.on('game_started', (data) => {
        onGameStarted(data);
    });

    socket.on('state_update', (data) => {
        onStateUpdate(data);
    });

    socket.on('game_over', (data) => {
        onGameOver(data);
    });

    socket.on('error', (data) => {
        showToast(data.message, true);
    });
}

// ──────────────────────────────────────────────
//  ARCH INFO PANEL HELPERS
// ──────────────────────────────────────────────

/**
 * Show/hide the architecture info panels for a given grid slot (1 or 2).
 * Panels: arch-info-{slot} (PatchNet), arch-info-cnn-{slot} (CNN)
 */
function updateArchPanel(agent, slot) {
    const patchPanel = document.getElementById(`arch-info-${slot}`);
    const cnnPanel = document.getElementById(`arch-info-cnn-${slot}`);

    if (patchPanel) patchPanel.classList.add('hidden');
    if (cnnPanel) cnnPanel.classList.add('hidden');

    if (!agent) return;

    if (agent.model_type === 'patch' && patchPanel) {
        patchPanel.classList.remove('hidden');
    } else if (agent.model_type === 'cnn' && cnnPanel) {
        cnnPanel.classList.remove('hidden');
    }
}

// ──────────────────────────────────────────────
//  SETUP SCREEN
// ──────────────────────────────────────────────
function filterAgents(cat, slot) {
    const gridId = slot === 1 ? 'agent-grid' : 'agent-grid-2';
    const filterId = slot === 1 ? 'category-filter-1' : 'category-filter-2';

    const grid = document.getElementById(gridId);
    const filter = document.getElementById(filterId);
    if (!grid) return;

    // Update active tab
    if (filter) {
        filter.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('selected'));
        const activeBtn = filter.querySelector(`[data-cat="${cat}"]`);
        if (activeBtn) activeBtn.classList.add('selected');
    }

    grid.querySelectorAll('.agent-option').forEach(el => {
        const agentCat = el.dataset.category;
        el.style.display = (cat === 'all' || agentCat === cat) ? '' : 'none';
    });
}

function buildAgentGrid() {
    const grid1 = document.getElementById('agent-grid');
    const grid2 = document.getElementById('agent-grid-2');
    grid1.innerHTML = '';
    if (grid2) grid2.innerHTML = '';

    agents.forEach((agent, idx) => {
        const maxDiff = 10;
        let diffDots = '';
        for (let i = 0; i < maxDiff; i++) {
            const cls = i < agent.difficulty
                ? (agent.difficulty >= 7 ? 'diff-dot filled high' : 'diff-dot filled')
                : 'diff-dot';
            diffDots += `<div class="${cls}"></div>`;
        }

        // Model-type badge
        let modelBadge = '';
        if (agent.model_type === 'cnn') {
            modelBadge = `<span class="arch-badge cnn-badge">CNN</span>`;
        } else if (agent.model_type === 'patch') {
            modelBadge = `<span class="arch-badge patch-badge">PATCH</span>`;
        }

        // Extra badge (e.g. NEW)
        let extraBadge = agent.badge ? `<span class="arch-badge new-badge">${agent.badge}</span>` : '';

        const contentHTML = `
            <div class="agent-name">${agent.name}${modelBadge}${extraBadge}</div>
            <div class="agent-desc">${agent.description}</div>
            <div class="agent-difficulty">${diffDots}</div>
        `;

        // Player 1 Grid
        const div1 = document.createElement('div');
        div1.className = 'agent-option' + (idx === 1 ? ' selected' : '');
        div1.dataset.category = agent.category || 'basic';
        div1.dataset.agentIdx = idx;
        if (idx === 1) { selectedAgent1 = agent; updateArchPanel(agent, 1); }
        div1.innerHTML = contentHTML;
        div1.onclick = () => {
            grid1.querySelectorAll('.agent-option').forEach(e => e.classList.remove('selected'));
            div1.classList.add('selected');
            selectedAgent1 = agent;
            updateArchPanel(agent, 1);
        };
        grid1.appendChild(div1);

        // Player 2 Grid (if exists)
        if (grid2) {
            const div2 = document.createElement('div');
            div2.className = 'agent-option' + (idx === 1 ? ' selected' : '');
            div2.dataset.category = agent.category || 'basic';
            div2.dataset.agentIdx = idx;
            if (idx === 1) { selectedAgent2 = agent; updateArchPanel(agent, 2); }
            div2.innerHTML = contentHTML;
            div2.onclick = () => {
                grid2.querySelectorAll('.agent-option').forEach(e => e.classList.remove('selected'));
                div2.classList.add('selected');
                selectedAgent2 = agent;
                updateArchPanel(agent, 2);
            };
            grid2.appendChild(div2);
        }
    });
}

// Size selector
document.querySelectorAll('.size-btn:not(.custom-size-box)').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.size-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');

        let r = btn.dataset.rows ? parseInt(btn.dataset.rows) : parseInt(btn.dataset.size);
        let c = btn.dataset.cols ? parseInt(btn.dataset.cols) : parseInt(btn.dataset.size);

        selectedSize = [r, c];

        const cr = document.getElementById('custom-rows');
        const cc = document.getElementById('custom-cols');
        if (cr) cr.value = '';
        if (cc) cc.value = '';
    });
});

function handleCustomSize() {
    const cr = document.getElementById('custom-rows').value;
    const cc = document.getElementById('custom-cols').value;
    if (cr && cc) {
        document.querySelectorAll('.size-btn').forEach(b => b.classList.remove('selected'));
        document.querySelector('.custom-size-box').classList.add('selected');
        let r = parseInt(cr);
        let c = parseInt(cc);
        if (r < 2) r = 2;
        if (r > 15) r = 15;
        if (c < 2) c = 2;
        if (c > 15) c = 15;
        selectedSize = [r, c];
    }
}

const crInput = document.getElementById('custom-rows');
const ccInput = document.getElementById('custom-cols');
if (crInput) crInput.addEventListener('input', handleCustomSize);
if (ccInput) ccInput.addEventListener('input', handleCustomSize);

const customBoxItem = document.querySelector('.custom-size-box');
if (customBoxItem) {
    customBoxItem.addEventListener('click', handleCustomSize);
}

function selectOrder(player) {
    selectedPlayer = player;
    document.querySelectorAll('.order-btn').forEach(b => b.classList.remove('selected'));
    document.querySelector(`.order-btn[data-player="${player}"]`).classList.add('selected');

    const card2 = document.getElementById('agent-config-card-2');
    const t1 = document.getElementById('agent1-title');
    if (player === -1) {
        if (card2) card2.classList.remove('hidden');
        if (t1) t1.textContent = 'Choose AI Player 1';
    } else {
        if (card2) card2.classList.add('hidden');
        if (t1) t1.textContent = 'Choose Your Opponent';
    }
}

// ──────────────────────────────────────────────
//  GAME ACTIONS
// ──────────────────────────────────────────────
function getModifiedSpec(spec) {
    if (!spec) return spec;
    const depth = document.getElementById('ai-depth').value;
    const sims = document.getElementById('ai-sims').value;
    const dagToggle = document.getElementById('ai-dag');
    const useDag = dagToggle ? dagToggle.checked : true;

    let [type, params] = spec.split(':');
    if (!params) params = '';

    let paramMap = {};
    if (params) {
        params.split(',').forEach(p => {
            let parts = p.split('=');
            if (parts.length === 2 && parts[0] && parts[1]) {
                paramMap[parts[0]] = parts[1];
            }
        });
    }

    if (depth && (type === 'minmax' || type === 'alphabeta')) {
        paramMap['depth'] = depth;
    }
    // Include alphazero_cnn in simulation override
    if (sims && (type === 'mcts' || type === 'alphazero_bit' || type === 'alphazero_cpp' ||
        type === 'alphazero_patch' || type === 'alphazero_cnn')) {
        paramMap['n_simulations'] = sims;
        if (type === 'mcts') paramMap['iterations'] = sims;
    }
    if (type === 'alphazero_cpp' || type === 'alphazero_patch' || type === 'alphazero_cnn') {
        paramMap['dag'] = useDag;
    }

    let newParams = Object.keys(paramMap).map(k => `${k}=${paramMap[k]}`).join(',');
    return newParams ? `${type}:${newParams}` : type;
}

function startGame() {
    if (!selectedAgent1) {
        showToast('Please select an opponent', true);
        return;
    }

    // Ensure we have an array for size
    let boardSizeArg = selectedSize;
    if (!Array.isArray(boardSizeArg)) {
        boardSizeArg = [selectedSize, selectedSize];
    }

    let payload = {
        agent_spec: getModifiedSpec(selectedAgent1.spec),
        board_size: boardSizeArg,
        human_player: selectedPlayer,
        env_type: 'classic',  // Auto fallbacks will happen on server if required
    };
    if (selectedPlayer === -1) {
        payload.agent2_spec = getModifiedSpec(selectedAgent2 ? selectedAgent2.spec : selectedAgent1.spec);
    }

    socket.emit('new_game', payload);
}

function loadArchive() {
    fetch('/api/archive')
        .then(res => res.json())
        .then(data => {
            const tbody = document.getElementById('archive-body');
            if (data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="padding: 10px; text-align: center; color: var(--text-muted);">No games played yet.</td></tr>';
                return;
            }
            tbody.innerHTML = '';
            data.forEach(game => {
                const tr = document.createElement('tr');
                tr.style.borderBottom = '1px solid var(--border)';
                tr.innerHTML = `
                    <td style="padding: 10px;">${game.date}</td>
                    <td style="padding: 10px;">${game.board_size}</td>
                    <td style="padding: 10px;">${game.matchup}</td>
                    <td style="padding: 10px;">${game.score}</td>
                    <td style="padding: 10px; font-weight: 500; color: ${game.winner.includes('Player') ? 'var(--accent)' : 'inherit'};">${game.winner}</td>
                `;
                tbody.appendChild(tr);
            });
        });
}

// Call on load
loadArchive();

function backToSetup() {
    document.getElementById('setup-screen').classList.remove('hidden');
    document.getElementById('game-screen').classList.remove('active');
    document.getElementById('game-over-overlay').classList.remove('show');
    gameActive = false;
}

function rematch() {
    document.getElementById('game-over-overlay').classList.remove('show');
    startGame();
}

// ──────────────────────────────────────────────
//  GAME EVENT HANDLERS
// ──────────────────────────────────────────────
function onGameStarted(data) {
    document.getElementById('setup-screen').classList.add('hidden');
    document.getElementById('game-screen').classList.add('active');
    document.getElementById('game-over-overlay').classList.remove('show');

    boardSize = data.board_size;
    humanPlayer = data.human_player;
    gameActive = true;
    moveCount = 0;
    lastMoveRC = null;
    previousScores = [0, 0];

    // Labels
    let p1Label = '';
    let p2Label = '';
    if (humanPlayer === -1) {
        p1Label = data.agent_name;
        p2Label = data.agent2_name || data.agent_name;
    } else {
        p1Label = humanPlayer === 1 ? 'You' : data.agent_name;
        p2Label = humanPlayer === 2 ? 'You' : data.agent_name;
    }
    document.getElementById('p1-label').textContent = `Player 1 (${p1Label})`;
    document.getElementById('p2-label').textContent = `Player 2 (${p2Label})`;

    // Stats
    document.getElementById('stat-board-size').textContent = `${boardSize[0]}×${boardSize[1]}`;
    document.getElementById('stat-moves-made').textContent = '0';
    document.getElementById('stat-agent').textContent = humanPlayer === -1 ? `${p1Label} vs ${p2Label}` : data.agent_name;
    const totalEdges = boardSize[0] * (boardSize[1] + 1) + boardSize[1] * (boardSize[0] + 1);
    document.getElementById('stat-remaining').textContent = totalEdges;

    // Clear log
    document.getElementById('move-log').innerHTML = '';

    buildBoard(data);
    updateTurnIndicator(data);
}

function onStateUpdate(data) {
    moveCount++;

    // Score bump animation
    if (data.score[0] > previousScores[0]) {
        const el = document.getElementById('score-p1');
        el.classList.remove('bump');
        void el.offsetWidth;
        el.classList.add('bump');
    }
    if (data.score[1] > previousScores[1]) {
        const el = document.getElementById('score-p2');
        el.classList.remove('bump');
        void el.offsetWidth;
        el.classList.add('bump');
    }
    previousScores = [...data.score];

    lastMoveRC = data.last_move ? { row: data.last_move.row, col: data.last_move.col } : null;

    updateBoard(data);
    updateScores(data);
    updateTurnIndicator(data);
    addMoveLogEntry(data.last_move);

    // Stats
    document.getElementById('stat-moves-made').textContent = moveCount;
    const totalEdges = boardSize[0] * (boardSize[1] + 1) + boardSize[1] * (boardSize[0] + 1);
    const placedEdges = countPlacedEdges(data);
    document.getElementById('stat-remaining').textContent = totalEdges - placedEdges;
}

function onGameOver(data) {
    gameActive = false;

    const overlay = document.getElementById('game-over-overlay');
    const icon = document.getElementById('go-icon');
    const title = document.getElementById('go-title');
    const score = document.getElementById('go-score');

    if (data.human_player === -1) {
        if (data.score[0] > data.score[1]) {
            icon.textContent = '🤖';
            title.textContent = 'Player 1 Wins!';
            title.className = 'game-over-title win';
        } else if (data.score[1] > data.score[0]) {
            icon.textContent = '👾';
            title.textContent = 'Player 2 Wins!';
            title.className = 'game-over-title lose';
        } else {
            icon.textContent = '🤝';
            title.textContent = "It's a Draw!";
            title.className = 'game-over-title draw';
        }
    } else {
        const yourScore = data.human_player === 1 ? data.score[0] : data.score[1];
        const oppScore = data.human_player === 1 ? data.score[1] : data.score[0];

        if (yourScore > oppScore) {
            icon.textContent = '🎉';
            title.textContent = 'You Win!';
            title.className = 'game-over-title win';
        } else if (oppScore > yourScore) {
            icon.textContent = '😔';
            title.textContent = 'You Lost';
            title.className = 'game-over-title lose';
        } else {
            icon.textContent = '🤝';
            title.textContent = "It's a Draw!";
            title.className = 'game-over-title draw';
        }
    }

    score.textContent = `${data.score[0]} – ${data.score[1]}`;

    // Show turn indicator as game over
    const ti = document.getElementById('turn-indicator');
    ti.className = 'turn-indicator game-over-indicator';
    ti.textContent = 'Game Over';

    setTimeout(() => overlay.classList.add('show'), 600);
}

// ──────────────────────────────────────────────
//  BOARD RENDERING
// ──────────────────────────────────────────────
function buildBoard(data) {
    const board = document.getElementById('game-board');
    board.innerHTML = '';

    const M = data.board_size[0]; // rows
    const N = data.board_size[1]; // cols
    const gridRows = 2 * M + 1;
    const gridCols = 2 * N + 1;

    board.style.gridTemplateColumns = buildGridTemplate(N);
    board.style.gridTemplateRows = buildGridTemplate(M);
    board.style.display = 'grid';

    for (let r = 0; r < gridRows; r++) {
        for (let c = 0; c < gridCols; c++) {
            const el = document.createElement('div');

            if (r % 2 === 0 && c % 2 === 0) {
                // DOT
                el.className = 'dot';
            } else if (r % 2 === 0 && c % 2 === 1) {
                // HORIZONTAL EDGE
                el.className = 'edge edge-h';
                el.dataset.row = r;
                el.dataset.col = c;
                el.innerHTML = '<div class="edge-line"></div>';
                el.onclick = () => onEdgeClick(r, c);

                const hi = r / 2;
                const hj = (c - 1) / 2;
                if (data.horizontal_edges[hi] && data.horizontal_edges[hi][hj]) {
                    el.classList.add('placed');
                }
            } else if (r % 2 === 1 && c % 2 === 0) {
                // VERTICAL EDGE
                el.className = 'edge edge-v';
                el.dataset.row = r;
                el.dataset.col = c;
                el.innerHTML = '<div class="edge-line"></div>';
                el.onclick = () => onEdgeClick(r, c);

                const vi = (r - 1) / 2;
                const vj = c / 2;
                if (data.vertical_edges[vi] && data.vertical_edges[vi][vj]) {
                    el.classList.add('placed');
                }
            } else {
                // BOX CELL
                el.className = 'box-cell';
                el.dataset.boxRow = (r - 1) / 2;
                el.dataset.boxCol = (c - 1) / 2;

                const bi = (r - 1) / 2;
                const bj = (c - 1) / 2;
                if (data.boxes[bi] && data.boxes[bi][bj] === 1) {
                    el.classList.add('p1-box', 'claimed');
                    el.innerHTML = '<span class="box-label">P1</span>';
                } else if (data.boxes[bi] && data.boxes[bi][bj] === 2) {
                    el.classList.add('p2-box', 'claimed');
                    el.innerHTML = '<span class="box-label">P2</span>';
                }
            }

            board.appendChild(el);
        }
    }

    // Disable edges if not human's turn
    if (data.current_player !== humanPlayer) {
        disableEdges();
    }
}

function buildGridTemplate(boxesCount) {
    const parts = [];
    for (let i = 0; i < 2 * boxesCount + 1; i++) {
        parts.push(i % 2 === 0 ? `${DOT_SIZE}px` : `${CELL_SIZE}px`);
    }
    return parts.join(' ');
}

function updateBoard(data) {
    const M = data.board_size[0];
    const N = data.board_size[1];
    const board = document.getElementById('game-board');

    // Remove old last-move highlights
    board.querySelectorAll('.last-move').forEach(e => e.classList.remove('last-move'));

    // Update horizontal edges
    for (let i = 0; i <= M; i++) {
        for (let j = 0; j < N; j++) {
            const r = i * 2;
            const c = j * 2 + 1;
            const el = board.querySelector(`.edge-h[data-row="${r}"][data-col="${c}"]`);
            if (!el) continue;

            if (data.horizontal_edges[i] && data.horizontal_edges[i][j]) {
                if (!el.classList.contains('placed')) {
                    el.classList.add('placed');
                }
                // Mark with player color
                if (data.last_move && data.last_move.row === r && data.last_move.col === c) {
                    el.classList.add('last-move');
                    el.classList.add(data.last_move.player === 1 ? 'p1' : 'p2');
                }
            }
        }
    }

    // Update vertical edges
    for (let i = 0; i < M; i++) {
        for (let j = 0; j <= N; j++) {
            const r = i * 2 + 1;
            const c = j * 2;
            const el = board.querySelector(`.edge-v[data-row="${r}"][data-col="${c}"]`);
            if (!el) continue;

            if (data.vertical_edges[i] && data.vertical_edges[i][j]) {
                if (!el.classList.contains('placed')) {
                    el.classList.add('placed');
                }
                if (data.last_move && data.last_move.row === r && data.last_move.col === c) {
                    el.classList.add('last-move');
                    el.classList.add(data.last_move.player === 1 ? 'p1' : 'p2');
                }
            }
        }
    }

    // Update boxes
    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            const el = board.querySelector(`.box-cell[data-box-row="${i}"][data-box-col="${j}"]`);
            if (!el) continue;

            if (data.boxes[i] && data.boxes[i][j] === 1 && !el.classList.contains('p1-box')) {
                el.classList.add('p1-box', 'claimed');
                el.innerHTML = '<span class="box-label">P1</span>';
            } else if (data.boxes[i] && data.boxes[i][j] === 2 && !el.classList.contains('p2-box')) {
                el.classList.add('p2-box', 'claimed');
                el.innerHTML = '<span class="box-label">P2</span>';
            }
        }
    }

    // Enable/disable edges based on turn
    if (!data.done && data.current_player === humanPlayer) {
        enableEdges();
    } else {
        disableEdges();
    }
}

function enableEdges() {
    document.querySelectorAll('.edge:not(.placed)').forEach(e => {
        e.classList.remove('disabled');
    });
}

function disableEdges() {
    document.querySelectorAll('.edge:not(.placed)').forEach(e => {
        e.classList.add('disabled');
    });
}

function onEdgeClick(row, col) {
    if (!gameActive) return;
    socket.emit('human_move', { row, col });
}

// ──────────────────────────────────────────────
//  UI UPDATES
// ──────────────────────────────────────────────
function updateScores(data) {
    document.getElementById('score-p1').textContent = data.score[0];
    document.getElementById('score-p2').textContent = data.score[1];

    const p1Card = document.getElementById('score-card-p1');
    const p2Card = document.getElementById('score-card-p2');
    p1Card.classList.toggle('active-turn', data.current_player === 1 && !data.done);
    p2Card.classList.toggle('active-turn', data.current_player === 2 && !data.done);
}

function updateTurnIndicator(data) {
    const ti = document.getElementById('turn-indicator');
    if (data.done) {
        ti.className = 'turn-indicator game-over-indicator';
        ti.textContent = 'Game Over';
    } else if (data.current_player === humanPlayer) {
        ti.className = 'turn-indicator your-turn';
        ti.textContent = 'Your Turn';
    } else {
        ti.className = 'turn-indicator ai-turn';
        ti.textContent = 'AI Thinking…';
    }
}

function addMoveLogEntry(move) {
    if (!move) return;
    const log = document.getElementById('move-log');

    // Remove previous latest
    log.querySelectorAll('.latest').forEach(e => e.classList.remove('latest'));

    const entry = document.createElement('div');
    const isHuman = move.player === humanPlayer;
    const playerLabel = isHuman ? 'You' : 'AI';
    const pClass = move.player === 1 ? 'p1-entry' : 'p2-entry';
    const edgeType = move.action[0] === 0 ? 'H' : 'V';

    let badges = '';
    // check if box was made by score change
    if (move.think_time !== undefined) {
        badges += `<span class="log-badge time-badge">${move.think_time}s</span>`;
    }

    entry.className = `log-entry ${pClass} latest`;
    entry.innerHTML = `
            <span class="log-num">#${moveCount}</span>
            <span class="log-text"><strong>${playerLabel}</strong> → ${edgeType}(${move.action[1]},${move.action[2]})</span>
            ${badges}
        `;

    log.appendChild(entry);
    entry.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function countPlacedEdges(data) {
    let count = 0;
    for (const row of data.horizontal_edges) {
        for (const e of row) { if (e) count++; }
    }
    for (const row of data.vertical_edges) {
        for (const e of row) { if (e) count++; }
    }
    return count;
}

// ──────────────────────────────────────────────
//  TOAST
// ──────────────────────────────────────────────
let toastTimer = null;
function showToast(msg, isError = false) {
    const toast = document.getElementById('status-toast');
    toast.textContent = msg;
    toast.className = 'status-toast show' + (isError ? ' error' : '');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 3000);
}

// ──────────────────────────────────────────────
//  INIT
// ──────────────────────────────────────────────
connectSocket();