// ==========================================================================
// Bangla Distill — Frontend Application Logic
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const tabArena = document.getElementById('tabArena');
  const tabChat = document.getElementById('tabChat');
  const viewArena = document.getElementById('viewArena');
  const viewChat = document.getElementById('viewChat');

  // Sliders
  const sliderTemp = document.getElementById('sliderTemp');
  const sliderTokens = document.getElementById('sliderTokens');
  const sliderTopP = document.getElementById('sliderTopP');
  const sliderRep = document.getElementById('sliderRep');
  const valTemp = document.getElementById('valTemp');
  const valTokens = document.getElementById('valTokens');
  const valTopP = document.getElementById('valTopP');
  const valRep = document.getElementById('valRep');
  const btnResetParams = document.getElementById('btnResetParams');

  // Arena Elements
  const arenaPromptInput = document.getElementById('arenaPromptInput');
  const arenaCharCount = document.getElementById('arenaCharCount');
  const btnRunArena = document.getElementById('btnRunArena');
  const baseOutput = document.getElementById('baseOutput');
  const studentOutput = document.getElementById('studentOutput');
  const baseMetrics = document.getElementById('baseMetrics');
  const studentMetrics = document.getElementById('studentMetrics');
  const btnCopyBase = document.getElementById('btnCopyBase');
  const btnCopyStudent = document.getElementById('btnCopyStudent');

  // Chat Elements
  const chatMessages = document.getElementById('chatMessages');
  const chatInput = document.getElementById('chatInput');
  const btnSendChat = document.getElementById('btnSendChat');
  const chatLoraToggle = document.getElementById('chatLoraToggle');

  // Telemetry
  const gpuStatusText = document.getElementById('gpuStatusText');
  const vramStatusText = document.getElementById('vramStatusText');
  const btnExport = document.getElementById('btnExport');
  const toast = document.getElementById('toast');

  // History State for Export
  let lastArenaResult = null;
  let chatHistory = [];

  // --- 1. Mode Switching ---
  tabArena.addEventListener('click', () => {
    tabArena.classList.add('active');
    tabChat.classList.remove('active');
    viewArena.classList.add('active');
    viewChat.classList.remove('active');
  });

  tabChat.addEventListener('click', () => {
    tabChat.classList.add('active');
    tabArena.classList.remove('active');
    viewChat.classList.add('active');
    viewArena.classList.remove('active');
    chatInput.focus();
  });

  // --- 2. Sliders & Parameters ---
  function updateSliderDisplay() {
    valTemp.textContent = parseFloat(sliderTemp.value).toFixed(2);
    valTokens.textContent = sliderTokens.value;
    valTopP.textContent = parseFloat(sliderTopP.value).toFixed(2);
    valRep.textContent = parseFloat(sliderRep.value).toFixed(2);
  }

  [sliderTemp, sliderTokens, sliderTopP, sliderRep].forEach(slider => {
    slider.addEventListener('input', updateSliderDisplay);
  });

  btnResetParams.addEventListener('click', () => {
    sliderTemp.value = 0.30;
    sliderTokens.value = 250;
    sliderTopP.value = 0.80;
    sliderRep.value = 1.20;
    updateSliderDisplay();
    showToast("Parameters reset to defaults");
  });

  function getParams() {
    return {
      temperature: parseFloat(sliderTemp.value),
      max_new_tokens: parseInt(sliderTokens.value),
      top_p: parseFloat(sliderTopP.value),
      repetition_penalty: parseFloat(sliderRep.value),
    };
  }

  // --- 3. Telemetry Fetch ---
  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        gpuStatusText.textContent = data.gpu_name || 'GPU Ready';
        vramStatusText.textContent = `${data.vram_allocated_mb} MB VRAM`;
      }
    } catch (err) {
      console.warn("Telemetry fetch error:", err);
    }
  }
  fetchStatus();
  setInterval(fetchStatus, 15000);

  // --- 4. Presets Click ---
  document.querySelectorAll('.preset-card').forEach(card => {
    card.addEventListener('click', () => {
      const prompt = card.getAttribute('data-prompt');
      if (viewArena.classList.contains('active')) {
        arenaPromptInput.value = prompt;
        arenaCharCount.textContent = `${prompt.length} chars`;
        arenaPromptInput.focus();
      } else {
        chatInput.value = prompt;
        chatInput.focus();
      }
      showToast("Preset prompt loaded");
    });
  });

  arenaPromptInput.addEventListener('input', () => {
    arenaCharCount.textContent = `${arenaPromptInput.value.length} chars`;
  });

  // --- 5. Arena Comparison Execution ---
  btnRunArena.addEventListener('click', runArenaComparison);
  arenaPromptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      runArenaComparison();
    }
  });

  async function runArenaComparison() {
    const prompt = arenaPromptInput.value.trim();
    if (!prompt) {
      showToast("Please enter a prompt first!");
      arenaPromptInput.focus();
      return;
    }

    // UI Loading State
    btnRunArena.disabled = true;
    btnRunArena.querySelector('.btn-text').textContent = "Running Both Models...";
    
    baseOutput.classList.remove('placeholder');
    studentOutput.classList.remove('placeholder');
    baseOutput.innerHTML = `<span style="color: #94a3b8;">Generating baseline response...</span>`;
    studentOutput.innerHTML = `<span style="color: #94a3b8;">Generating distilled student response...</span>`;

    baseMetrics.textContent = "Processing...";
    studentMetrics.textContent = "Processing...";
    btnCopyBase.disabled = true;
    btnCopyStudent.disabled = true;

    const payload = {
      prompt,
      ...getParams()
    };

    try {
      const res = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Generation failed");
      }

      const data = await res.json();
      lastArenaResult = data;

      // Base model rendering
      baseOutput.textContent = data.base_model.text || "(Empty response)";
      baseMetrics.textContent = `${data.base_model.duration_seconds}s • ${data.base_model.tokens_generated} tokens (${data.base_model.tokens_per_second} tok/s)`;
      btnCopyBase.disabled = false;

      // Student model rendering
      studentOutput.textContent = data.student_model.text || "(Empty response)";
      studentMetrics.textContent = `${data.student_model.duration_seconds}s • ${data.student_model.tokens_generated} tokens (${data.student_model.tokens_per_second} tok/s)`;
      btnCopyStudent.disabled = false;

      // Update VRAM display
      if (data.vram_allocated_mb) {
        vramStatusText.textContent = `${data.vram_allocated_mb} MB VRAM`;
      }

      showToast("Arena comparison complete!");
    } catch (err) {
      baseOutput.innerHTML = `<span style="color: #ef4444;">Error: ${err.message}</span>`;
      studentOutput.innerHTML = `<span style="color: #ef4444;">Error: ${err.message}</span>`;
      baseMetrics.textContent = "Failed";
      studentMetrics.textContent = "Failed";
    } finally {
      btnRunArena.disabled = false;
      btnRunArena.querySelector('.btn-text').textContent = "Run Arena Comparison";
    }
  }

  // --- 6. Interactive Chat Execution ---
  btnSendChat.addEventListener('click', sendChatMessage);
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });

  async function sendChatMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    chatInput.value = "";
    btnSendChat.disabled = true;

    // Append User Bubble
    appendBubble("user", text);

    // Append Assistant Bubble placeholder
    const assistantBubble = appendBubble("model", "চিন্তাভাবনা করছে...");
    const contentEl = assistantBubble.querySelector('.bubble-content');

    const payload = {
      prompt: text,
      use_lora: chatLoraToggle.checked,
      ...getParams()
    };

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Chat generation failed");
      }

      const data = await res.json();
      contentEl.textContent = data.text || "(Empty response)";
      chatHistory.push({ role: "user", content: text });
      chatHistory.push({ role: "model", content: data.text });
    } catch (err) {
      contentEl.innerHTML = `<span style="color: #ef4444;">Error: ${err.message}</span>`;
    } finally {
      btnSendChat.disabled = false;
      chatInput.focus();
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  }

  function appendBubble(role, content) {
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role === 'user' ? 'user-bubble' : 'model-bubble'}`;

    const avatar = document.createElement('div');
    avatar.className = 'bubble-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🇧🇩';

    const body = document.createElement('div');
    body.className = 'bubble-content bengali';
    body.textContent = content;

    bubble.appendChild(avatar);
    bubble.appendChild(body);
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
  }

  // --- 7. Copy Helpers ---
  btnCopyBase.addEventListener('click', () => {
    navigator.clipboard.writeText(baseOutput.textContent);
    showToast("Base model output copied!");
  });

  btnCopyStudent.addEventListener('click', () => {
    navigator.clipboard.writeText(studentOutput.textContent);
    showToast("Distilled student output copied!");
  });

  // --- 8. Export to Markdown ---
  btnExport.addEventListener('click', () => {
    let md = `# Bangla Distill — Evaluation Export\n\n`;
    md += `**Date:** ${new Date().toLocaleString()}\n`;
    md += `**Model:** Qwen/Qwen2.5-0.5B + Bangla LoRA Adapter\n`;
    md += `**Teachers:** Gemma-2-9B-IT + Qwen 3.5 9B\n\n---\n\n`;

    if (lastArenaResult) {
      md += `## Side-by-Side Arena Comparison\n\n`;
      md += `**Prompt:** ${lastArenaResult.prompt}\n\n`;
      md += `### 🔴 Vanilla Base Model (No LoRA)\n`;
      md += `- **Duration:** ${lastArenaResult.base_model.duration_seconds}s (${lastArenaResult.base_model.tokens_per_second} tok/s)\n`;
      md += `- **Tokens:** ${lastArenaResult.base_model.tokens_generated}\n\n`;
      md += `> ${lastArenaResult.base_model.text}\n\n`;
      md += `### 🟢 Distilled Student Model (With LoRA)\n`;
      md += `- **Duration:** ${lastArenaResult.student_model.duration_seconds}s (${lastArenaResult.student_model.tokens_per_second} tok/s)\n`;
      md += `- **Tokens:** ${lastArenaResult.student_model.tokens_generated}\n\n`;
      md += `> ${lastArenaResult.student_model.text}\n\n---\n\n`;
    }

    if (chatHistory.length > 0) {
      md += `## Chat Session History\n\n`;
      chatHistory.forEach(msg => {
        md += `**${msg.role === 'user' ? 'User' : 'Bangla Student'}:** ${msg.content}\n\n`;
      });
    }

    if (!lastArenaResult && chatHistory.length === 0) {
      showToast("No active comparison or chat session to export yet.");
      return;
    }

    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Bangla_Distill_Export_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Exported session to Markdown!");
  });

  // --- 9. Toast Notification ---
  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 2500);
  }
});
