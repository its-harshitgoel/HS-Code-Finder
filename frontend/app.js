/**
 * HSCodeFinder — Chat Application Logic
 *
 * Purpose: Manages the chat interface, communicates with the /api/classify endpoint,
 *          renders messages, typing indicators, and result cards.
 *
 * Session management: Tracks session_id for multi-turn conversations.
 * Renders: User messages, assistant messages with markdown-like formatting,
 *          result cards and hierarchy paths.
 */

(function () {
  "use strict";

  // ---------- DOM Elements ----------
  const welcomeScreen = document.getElementById("welcome-screen");
  const messagesContainer = document.getElementById("messages-container");
  const typingIndicator = document.getElementById("typing-indicator");
  const messageInput = document.getElementById("message-input");
  const sendBtn = document.getElementById("send-btn");
  const newChatBtn = document.getElementById("new-chat-btn");
  const chatArea = document.getElementById("chat-area");

  // ---------- State ----------
  let sessionId = null;
  let isLoading = false;

  // ---------- API ----------
  const API_URL = "/api/classify";

  async function sendMessage(message) {
    const payload = {
      session_id: sessionId,
      message: message.trim(),
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90000); // 90s

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // ---------- Message Rendering ----------

  function renderUserMessage(text) {
    const messageEl = document.createElement("div");
    messageEl.className = "message message-user";
    messageEl.innerHTML = `
            <div class="message-avatar">You</div>
            <div class="message-bubble">${escapeHtml(text)}</div>
        `;
    messagesContainer.appendChild(messageEl);
    scrollToBottom();
  }

  function renderAssistantMessage(response) {
    const messageEl = document.createElement("div");
    messageEl.className = "message message-assistant";

    let bubbleContent = "";

    if (response.type === "result" && response.final_result) {
      bubbleContent = renderResultCard(response);
    } else {
      bubbleContent = formatMessage(response.message);
    }

    messageEl.innerHTML = `
            <div class="message-avatar">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                    <line x1="16" y1="13" x2="8" y2="13"/>
                    <line x1="16" y1="17" x2="8" y2="17"/>
                </svg>
            </div>
            <div class="message-bubble">${bubbleContent}</div>
        `;

    messagesContainer.appendChild(messageEl);

    if (response.options && response.options.length >= 2) {
      renderOptionButtons(response.options);
    }

    scrollToBottom();
  }

  function renderOptionButtons(options) {
    removeOptionButtons();

    const container = document.createElement("div");
    container.className = "option-buttons";
    container.id = "option-buttons";

    options.forEach((option) => {
      const btn = document.createElement("button");
      btn.className = "option-btn";
      btn.textContent = option;
      btn.addEventListener("click", () => {
        if (isLoading) return;
        removeOptionButtons();
        messageInput.value = option;
        autoResize();
        handleSend();
      });
      container.appendChild(btn);
    });

    messagesContainer.appendChild(container);
  }

  function removeOptionButtons() {
    const existing = document.getElementById("option-buttons");
    if (existing) existing.remove();
  }

  function renderResultCard(response) {
    const result = response.final_result;
    const expl = result.explanation || "";

    // Extract and remove the classification path line so it can be styled separately.
    let hierarchyHtml = "";
    let bodyText = expl;
    const pathMatch = expl.match(/\*\*Classification path:\*\*\s*(.+)/);
    if (pathMatch) {
      const steps = pathMatch[1].split(" → ");
      hierarchyHtml = steps
        .map((step) => `<span class="hierarchy-step">${escapeHtml(step.trim())}</span>`)
        .join('<span class="hierarchy-arrow">→</span>');
      bodyText = bodyText.replace(/\*\*Classification path:\*\*.*$/s, "").trim();
    }

    const explanationHtml = bodyText
      ? `<div class="result-explanation">${formatMessage(bodyText)}</div>`
      : "";

    return `
            <div class="result-card">
                <div class="result-header">
                    <span class="result-badge">✓ Classified</span>
                </div>
                <div class="hs-code">${escapeHtml(result.hs_code)}</div>
                <div class="hs-description">${escapeHtml(result.description)}</div>
                ${explanationHtml}
                ${hierarchyHtml ? `<div class="hierarchy-path">${hierarchyHtml}</div>` : ""}
            </div>
        `;
  }

  // ---------- Text Formatting ----------

  function formatMessage(text) {
    if (!text) return "";

    // Convert markdown-like bold
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    // Convert line breaks
    html = html.replace(/\n\n/g, "<br><br>");
    html = html.replace(/\n/g, "<br>");

    return html;
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // ---------- UI Helpers ----------

  function scrollToBottom() {
    requestAnimationFrame(() => {
      chatArea.scrollTop = chatArea.scrollHeight;
    });
  }

  function showTyping() {
    typingIndicator.classList.remove("hidden");
    scrollToBottom();
  }

  function hideTyping() {
    typingIndicator.classList.add("hidden");
  }

  function setLoading(loading) {
    isLoading = loading;
    sendBtn.disabled = loading || messageInput.value.trim() === "";
    messageInput.disabled = loading;

    if (loading) {
      showTyping();
    } else {
      hideTyping();
      messageInput.focus();
    }
  }

  function hideWelcomeScreen() {
    if (welcomeScreen) {
      welcomeScreen.style.display = "none";
    }
  }

  function showWelcomeScreen() {
    if (welcomeScreen) {
      welcomeScreen.style.display = "flex";
    }
  }

  function clearChat() {
    messagesContainer.innerHTML = "";
    sessionId = null;
    showWelcomeScreen();
  }

  // ---------- Auto-resize Textarea ----------
  function autoResize() {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
  }

  // ---------- Handle Send ----------

  async function handleSend() {
    const message = messageInput.value.trim();
    if (!message || isLoading) return;

    // Clear input
    messageInput.value = "";
    autoResize();
    sendBtn.disabled = true;

    // Hide welcome screen on first message
    hideWelcomeScreen();

    // Clear any pending option buttons
    removeOptionButtons();

    // Render user message
    renderUserMessage(message);

    // Send to API
    setLoading(true);

    try {
      const response = await sendMessage(message);

      // Update session ID
      if (response.session_id) {
        sessionId = response.session_id;
      }

      // Render assistant response
      renderAssistantMessage(response);

      // If result, reset session for next query
      if (response.type === "result") {
        sessionId = null;
      }
    } catch (error) {
      console.error("API Error:", error);
      const isTimeout = error.name === "AbortError";
      renderAssistantMessage({
        type: "question",
        message: isTimeout
          ? "The request timed out. The server may be busy — please try again."
          : "Sorry, something went wrong. Please try again or rephrase your description.",
        candidates: [],
        final_result: null,
      });
    } finally {
      setLoading(false);
    }
  }

  // ---------- Event Listeners ----------

  // Send button
  sendBtn.addEventListener("click", handleSend);

  // Enter to send, Shift+Enter for newline
  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  // Enable/disable send button based on input
  messageInput.addEventListener("input", () => {
    autoResize();
    sendBtn.disabled = messageInput.value.trim() === "" || isLoading;
  });

  // New chat button
  newChatBtn.addEventListener("click", clearChat);

  // Focus input on load
  messageInput.focus();
})();
