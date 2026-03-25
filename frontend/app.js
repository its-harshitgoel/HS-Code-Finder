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

    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
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
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                </svg>
            </div>
            <div class="message-bubble">${bubbleContent}</div>
        `;

    messagesContainer.appendChild(messageEl);
    scrollToBottom();
  }

  function renderResultCard(response) {
    const result = response.final_result;

    // Parse hierarchy from explanation
    let hierarchyHtml = "";
    const pathMatch = result.explanation.match(
      /Classification path:\*\*\s*(.+)/,
    );
    if (pathMatch) {
      const steps = pathMatch[1].split(" → ");
      hierarchyHtml = steps
        .map(
          (step) =>
            `<span class="hierarchy-step">${escapeHtml(step.trim())}</span>`,
        )
        .join('<span class="hierarchy-arrow">→</span>');
    }

    return `
            <div class="result-card">
                <div class="result-header">
                    <span class="result-badge">✓ Classified</span>
                </div>
                <div class="hs-code">${escapeHtml(result.hs_code)}</div>
                <div class="hs-description">${escapeHtml(result.description)}</div>
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
      renderAssistantMessage({
        type: "question",
        message:
          "Sorry, something went wrong. Please try again or rephrase your description.",
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

  // Example chips
  document.querySelectorAll(".example-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      messageInput.value = query;
      autoResize();
      sendBtn.disabled = false;
      handleSend();
    });
  });

  // Focus input on load
  messageInput.focus();
})();
